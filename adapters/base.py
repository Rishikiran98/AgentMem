"""Uniform asynchronous interface over the memory systems under test.

Semantics the runner relies on
------------------------------
* ``reset(session_id)`` removes every memory the system holds for that session
  and nothing else.  ``reset_all()`` returns the system to its initial state.
* ``write`` submits messages and returns once the system *acknowledges* them.
  Acknowledgement is not retrievability.
* ``read`` returns the assembled memory context the downstream reader model
  would receive, produced by the system's normal retrieval path.  Adapters must
  not rerank, filter, summarise or otherwise improve on it.
* Settlement is measured with an externally observable canary: a uniquely
  identifiable fact is written through the system's own write path, and
  ``is_settled`` returns True only once the *normal read path* returns it.

Every adapter emits structured events through an ``EventSink`` so the runner
can persist them; the adapter never keeps analysis state itself.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

EventSink = Callable[[dict[str, Any]], None]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _ms(a: float | None, b: float | None) -> float | None:
    return round((b - a) * 1000.0, 3) if a is not None and b is not None else None


class AdapterError(RuntimeError):
    """Raised when the memory system reports or exhibits a failure."""


class SettlementError(RuntimeError):
    """Raised when settlement is queried without a planted canary."""


@dataclass
class WriteResult:
    session_id: str
    write_id: str
    submitted_at: str
    acknowledged_at: str
    ack_latency_ms: float
    messages: int
    memories_added: int | None = None
    memories_updated: int | None = None
    memories_deleted: int | None = None
    memories_noop: int | None = None
    memory_ids: list[str] = field(default_factory=list)
    native: Any = None
    # perf_counter stamps for lag computations (not serialised by the runner)
    submitted_perf: float = 0.0
    acknowledged_perf: float = 0.0

    def to_event_fields(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("native", None)
        d.pop("submitted_perf", None)
        d.pop("acknowledged_perf", None)
        return d


@dataclass
class ReadHit:
    memory_id: str | None
    text: str
    score: float | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None  # system-reported creation timestamp, if it exposes one


@dataclass
class ReadResult:
    session_id: str
    read_id: str
    query: str
    started_at: str
    finished_at: str
    latency_ms: float
    hits: list[ReadHit]
    context: str
    native: Any = None

    def to_event_fields(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "read_id": self.read_id,
            "query": self.query,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
            "hits": len(self.hits),
            "hit_ids": [h.memory_id for h in self.hits],
            "hit_scores": [h.score for h in self.hits],
            "context_chars": len(self.context),
        }


@dataclass
class CanaryRecord:
    session_id: str
    canary_id: str
    token: str
    text: str
    submitted_at: str
    submitted_perf: float
    acknowledged_at: str | None = None
    acknowledged_perf: float | None = None
    ack_latency_ms: float | None = None
    stored: bool | None = None  # did the system report storing it?
    memory_ids: list[str] = field(default_factory=list)
    polls: int = 0
    previous_failed_observation_perf: float | None = None
    first_retrievable_at: str | None = None
    first_retrievable_perf: float | None = None
    settled: bool = False
    timed_out: bool = False
    cleanup: str | None = None  # "deleted" | "skipped" | "failed: ..."

    def lag_fields(self) -> dict[str, Any]:
        return {
            "canary_id": self.canary_id,
            "canary_token": self.token,
            "canary_stored": self.stored,
            "canary_memory_ids": self.memory_ids,
            "submitted_at": self.submitted_at,
            "acknowledged_at": self.acknowledged_at,
            "first_retrievable_at": self.first_retrievable_at,
            "polls": self.polls,
            "poll_count": self.polls,
            "write_ack_latency_ms": self.ack_latency_ms,
            "ingestion_to_retrievability_lag_ms": _ms(self.acknowledged_perf, self.first_retrievable_perf),
            "visibility_lower_bound_ms": (
                _ms(self.acknowledged_perf, self.previous_failed_observation_perf)
                if self.previous_failed_observation_perf is not None
                else 0.0
            ),
            "visibility_upper_bound_ms": _ms(self.acknowledged_perf, self.first_retrievable_perf),
            "submission_to_retrievability_ms": _ms(self.submitted_perf, self.first_retrievable_perf),
            "settled": self.settled,
            "timed_out": self.timed_out,
            "cleanup": self.cleanup,
        }


@dataclass
class SettlementResult:
    session_id: str
    settled: bool
    timed_out: bool
    timeout_s: float
    poll_interval_s: float
    polls: int
    canary: CanaryRecord
    started_at: str
    finished_at: str

    def to_event_fields(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "settled": self.settled,
            "timed_out": self.timed_out,
            "settlement_timeout_s": self.timeout_s,
            "poll_interval_s": self.poll_interval_s,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            **self.canary.lag_fields(),
        }


@dataclass
class SettlementConfig:
    timeout_s: float = 30.0
    poll_interval_s: float = 0.25
    canary_cleanup: bool = True


class MemoryAdapter(ABC):
    """Base class.  Subclasses implement the ``_impl`` hooks; the public methods add timing and events."""

    name: str = "base"
    adapter_version: str = "0.0.0"

    def __init__(self, *, settlement: SettlementConfig | None = None, event_sink: EventSink | None = None) -> None:
        self.settlement_config = settlement or SettlementConfig()
        self._sink: EventSink = event_sink or (lambda e: None)
        self._canaries: dict[str, CanaryRecord] = {}

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def set_event_sink(self, sink: EventSink) -> None:
        self._sink = sink

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        ev = {"event_type": event_type, "timestamp": utc_now_iso(), "system": self.name, "adapter_version": self.adapter_version, **fields}
        self._sink(ev)
        return ev

    # ------------------------------------------------------------------ #
    # Public interface (spec)
    # ------------------------------------------------------------------ #
    async def reset(self, session_id: str) -> None:
        t0 = time.perf_counter()
        self.emit("RESET_START", session_id=session_id)
        try:
            await self._reset_impl(session_id)
        except Exception as exc:
            self.emit("ERROR", session_id=session_id, phase="reset", error_type=type(exc).__name__, error_message=str(exc)[:2000])
            raise
        self._canaries.pop(session_id, None)
        self.emit("RESET_END", session_id=session_id, latency_ms=_ms(t0, time.perf_counter()))

    async def write(self, session_id: str, messages: list[dict], *, metadata: dict[str, Any] | None = None) -> WriteResult:
        write_id = uuid.uuid4().hex
        submitted_at, t0 = utc_now_iso(), time.perf_counter()
        self.emit("WRITE_SUBMIT", session_id=session_id, write_id=write_id, messages=len(messages), submitted_at=submitted_at)
        try:
            partial = await self._write_impl(session_id, messages, metadata=metadata)
        except Exception as exc:
            self.emit("ERROR", session_id=session_id, write_id=write_id, phase="write", error_type=type(exc).__name__, error_message=str(exc)[:2000], latency_ms=_ms(t0, time.perf_counter()))
            raise AdapterError(f"{self.name} write failed: {type(exc).__name__}: {exc}") from exc
        t1 = time.perf_counter()
        result = WriteResult(
            session_id=session_id,
            write_id=write_id,
            submitted_at=submitted_at,
            acknowledged_at=utc_now_iso(),
            ack_latency_ms=_ms(t0, t1) or 0.0,
            messages=len(messages),
            submitted_perf=t0,
            acknowledged_perf=t1,
            **partial,
        )
        self.emit("WRITE_ACK", **result.to_event_fields())
        return result

    async def read(self, session_id: str, query: str) -> str:
        """Assembled memory context for the reader model (spec signature)."""
        return (await self.search(session_id, query)).context

    async def search(self, session_id: str, query: str, *, operation: str = "read") -> ReadResult:
        """Like ``read`` but returns the structured result; ``operation`` tags proxy attribution."""
        read_id = uuid.uuid4().hex
        started_at, t0 = utc_now_iso(), time.perf_counter()
        self.emit("READ_START", session_id=session_id, read_id=read_id, operation=operation, query_chars=len(query))
        try:
            hits, native = await self._read_impl(session_id, query, operation=operation)
        except Exception as exc:
            self.emit("ERROR", session_id=session_id, read_id=read_id, phase=operation, error_type=type(exc).__name__, error_message=str(exc)[:2000], latency_ms=_ms(t0, time.perf_counter()))
            raise AdapterError(f"{self.name} read failed: {type(exc).__name__}: {exc}") from exc
        t1 = time.perf_counter()
        result = ReadResult(
            session_id=session_id,
            read_id=read_id,
            query=query,
            started_at=started_at,
            finished_at=utc_now_iso(),
            latency_ms=_ms(t0, t1) or 0.0,
            hits=hits,
            context=self.assemble_context(hits),
            native=native,
        )
        self.emit("READ_END", operation=operation, **result.to_event_fields())
        return result

    @staticmethod
    def assemble_context(hits: list[ReadHit]) -> str:
        """Uniform context assembly: one retrieved memory per line, in the system's own order."""
        return "\n".join(h.text for h in hits)

    async def plant_canary(self, session_id: str) -> CanaryRecord:
        """Write a uniquely identifiable fact through the system's write path."""
        token = "memharness-canary-" + uuid.uuid4().hex
        text = self.canary_text(token)
        rec = CanaryRecord(session_id=session_id, canary_id=uuid.uuid4().hex, token=token, text=text, submitted_at=utc_now_iso(), submitted_perf=time.perf_counter())
        self.emit("CANARY_SUBMIT", session_id=session_id, canary_id=rec.canary_id, canary_token=token, submitted_at=rec.submitted_at)
        try:
            stored, memory_ids = await self._canary_write_impl(session_id, text)
        except Exception as exc:
            self.emit("ERROR", session_id=session_id, canary_id=rec.canary_id, phase="canary_write", error_type=type(exc).__name__, error_message=str(exc)[:2000])
            raise AdapterError(f"{self.name} canary write failed: {type(exc).__name__}: {exc}") from exc
        rec.acknowledged_perf = time.perf_counter()
        rec.acknowledged_at = utc_now_iso()
        rec.ack_latency_ms = _ms(rec.submitted_perf, rec.acknowledged_perf)
        rec.stored = stored
        rec.memory_ids = list(memory_ids)
        self._canaries[session_id] = rec
        self.emit("CANARY_ACK", session_id=session_id, canary_id=rec.canary_id, acknowledged_at=rec.acknowledged_at, ack_latency_ms=rec.ack_latency_ms, canary_stored=stored, canary_memory_ids=rec.memory_ids)
        return rec

    @staticmethod
    def canary_text(token: str) -> str:
        return f"The verification code for this session is {token}."

    async def is_settled(self, session_id: str) -> bool:
        """True only when the latest planted canary is returned by the normal read path."""
        rec = self._canaries.get(session_id)
        if rec is None:
            raise SettlementError(f"no canary planted for session {session_id!r}; call plant_canary() first")
        if rec.settled:
            return True
        rec.polls += 1
        t0 = time.perf_counter()
        result = await self.search(session_id, rec.text, operation="settle")
        visible = rec.token in result.context
        observed_perf = time.perf_counter()
        self.emit(
            "SETTLEMENT_POLL", session_id=session_id, canary_id=rec.canary_id,
            poll=rec.polls, poll_count=rec.polls, visible=visible, read_id=result.read_id,
            latency_ms=_ms(t0, observed_perf),
            observation_boundary_ms=_ms(rec.acknowledged_perf, observed_perf),
        )
        if visible:
            rec.settled = True
            rec.first_retrievable_perf = observed_perf
            rec.first_retrievable_at = utc_now_iso()
        else:
            rec.previous_failed_observation_perf = observed_perf
        return visible

    async def wait_settled(self, session_id: str, *, timeout_s: float | None = None, poll_interval_s: float | None = None) -> SettlementResult:
        """Plant a canary (if none is pending) and poll ``is_settled`` until True or timeout."""
        cfg = self.settlement_config
        timeout_s = cfg.timeout_s if timeout_s is None else timeout_s
        poll_interval_s = cfg.poll_interval_s if poll_interval_s is None else poll_interval_s
        started_at = utc_now_iso()
        rec = self._canaries.get(session_id)
        if rec is None or rec.settled:
            rec = await self.plant_canary(session_id)
        deadline = rec.acknowledged_perf + timeout_s if rec.acknowledged_perf else time.perf_counter() + timeout_s
        settled = False
        while True:
            settled = await self.is_settled(session_id)
            if settled or time.perf_counter() >= deadline:
                break
            await asyncio.sleep(poll_interval_s)
        rec.timed_out = not settled
        if cfg.canary_cleanup:
            try:
                await self._canary_cleanup_impl(session_id, rec)
                rec.cleanup = "deleted"
            except Exception as exc:  # noqa: BLE001 - cleanup failure is recorded, not fatal
                rec.cleanup = f"failed: {type(exc).__name__}: {exc}"[:500]
        else:
            rec.cleanup = "skipped"
        result = SettlementResult(session_id=session_id, settled=settled, timed_out=not settled, timeout_s=timeout_s, poll_interval_s=poll_interval_s, polls=rec.polls, canary=rec, started_at=started_at, finished_at=utc_now_iso())
        self.emit("WRITE_SETTLED" if settled else "SETTLEMENT_TIMEOUT", **result.to_event_fields())
        return result

    @abstractmethod
    def config_fingerprint(self) -> dict[str, Any]:
        ...

    async def reset_all(self) -> None:
        """Return the whole system to its initial state (default: unsupported)."""
        raise NotImplementedError(f"{self.name} does not implement reset_all")

    async def health(self) -> dict[str, Any]:
        return {"ok": True}

    async def close(self) -> None:
        return None

    # ------------------------------------------------------------------ #
    # Hooks implemented per system
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def _reset_impl(self, session_id: str) -> None: ...

    @abstractmethod
    async def _write_impl(self, session_id: str, messages: list[dict], *, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Return WriteResult fields: memories_added/updated/deleted/noop, memory_ids, native."""

    @abstractmethod
    async def _read_impl(self, session_id: str, query: str, *, operation: str) -> tuple[list[ReadHit], Any]: ...

    @abstractmethod
    async def _canary_write_impl(self, session_id: str, text: str) -> tuple[bool, list[str]]:
        """Write the canary; return (stored?, memory_ids)."""

    async def _canary_cleanup_impl(self, session_id: str, rec: CanaryRecord) -> None:
        """Remove the canary after settlement (default: nothing)."""
        return None
