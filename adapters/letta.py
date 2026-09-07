"""Letta adapter.

Which Letta
-----------
Letta's open-source Python server (the "Letta V1 API server") was retired when
github.com/letta-ai/letta became a landing page for letta-code; release 0.16.8
is the last one, kept by Letta "for reproducibility" and "unsupported".  This
adapter benchmarks that server, self-hosted (compose/letta uses the official
image, which bundles PostgreSQL + pgvector; tests run it from
scripts/setup_letta_env.sh on an embedded PostgreSQL).  Every fingerprint and
event carries ``deployment: self-hosted-letta-v1-server-archived``.

Memory model (letta 0.16.8, agent_type letta_v1_agent):
  * core memory   - in-context blocks (``human``, ``persona``) that the agent edits
                    with its ``memory_insert`` / ``memory_replace`` tools;
  * archival      - embedded passages (pgvector), written by the API
                    (``passages.create``) or by the deprecated ``archival_memory_*``
                    tools if attached; searched with ``passages.search``;
  * recall        - the message history, searchable only by the agent's
                    ``conversation_search`` tool (the ``messages/search`` API
                    requires Letta's hosted Turbopuffer backend).

Translation of benchmark semantics:

    reset(session)   -> delete agent(s) named memharness-<session>; create a fresh agent with
                        explicit llm_config / embedding_config whose endpoints are proxy
                        path-token URLs (exact attribution per session); verified empty.
    write(session)   -> agents.messages.create(agent, messages=<user/system/assistant turns>):
                        one agent step; the agent's LLM decides what to store (core-memory
                        tool calls).  Memory writes are counted from the returned tool calls.
    read(session, q) -> core memory blocks (label: value) + passages.search(q, top_k) results,
                        one item per line.  Recall memory is not reachable through the API
                        (see above) and is therefore not part of the measured context.
    canary           -> passages.create(text) (documented direct archival write, no LLM),
                        polled through the same read; deleted afterwards.  ``canary_mode:
                        message`` sends it through the agent instead (for sleep-time studies).
    reset_all()      -> delete every agent with the harness name prefix.

Session dates: Letta stamps messages at ingestion time and exposes no way to
set them; the runner's date system message is passed through as a system-role
message (accepted by the API).  This is recorded as an architectural difference.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from letta_client import AsyncLetta

from adapters.base import AdapterError, CanaryRecord, EventSink, MemoryAdapter, ReadHit, SettlementConfig
from proxy.client import ProxyClient

ADAPTER_VERSION = "0.1.0"
NAME_PREFIX = "memharness-"
MEMORY_TOOL_NAMES = ("memory_insert", "memory_replace", "memory_apply_patch", "memory_rethink", "memory", "core_memory_append", "core_memory_replace", "archival_memory_insert")


@dataclass
class LettaSettings:
    server_url: str
    llm: dict[str, Any]
    embedding: dict[str, Any]
    agent: dict[str, Any] = field(default_factory=dict)
    archival_top_k: int = 10
    include_core_blocks: bool = True
    max_steps: int = 10
    canary_mode: str = "passage"  # passage | message
    system_commit: str | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LettaSettings":
        l = cfg["letta"]
        r = cfg.get("retrieval", {})
        s = cfg.get("settlement", {})
        return cls(
            server_url=l["server_url"],
            llm=dict(l["llm"]),
            embedding=dict(l["embedding"]),
            agent=dict(l.get("agent", {})),
            archival_top_k=int(r.get("archival_top_k", 10)),
            include_core_blocks=bool(r.get("include_core_blocks", True)),
            max_steps=int(l.get("max_steps", 10)),
            canary_mode=s.get("canary_mode", "passage"),
            system_commit=cfg.get("system_commit"),
        )


class LettaAdapter(MemoryAdapter):
    name = "letta"
    adapter_version = ADAPTER_VERSION

    def __init__(self, settings: LettaSettings, *, proxy_base_url: str, configuration_id: str, seed: int, run_id: str, settlement: SettlementConfig | None = None, event_sink: EventSink | None = None) -> None:
        super().__init__(settlement=settlement, event_sink=event_sink)
        self.settings = settings
        self.proxy = ProxyClient(proxy_base_url)
        self.configuration_id = configuration_id
        self.seed = seed
        self.run_id = run_id
        self._client: AsyncLetta | None = None
        self._agents: dict[str, str] = {}  # session_id -> agent_id
        self._canary_passages: dict[str, list[str]] = {}
        self._probe: dict[str, Any] = {}
        self._started = False
        if settings.canary_mode not in ("passage", "message"):
            raise ValueError(f"unknown canary_mode {settings.canary_mode!r}")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _agent_name(self, session_id: str) -> str:
        return f"{NAME_PREFIX}{session_id}"

    def _static_tags(self, session_id: str) -> dict[str, Any]:
        return dict(system=self.name, configuration=self.configuration_id, seed=self.seed, run_id=self.run_id, session_id=session_id)

    def _llm_config(self, session_id: str) -> dict[str, Any]:
        s = self.settings.llm
        # The agent's own model calls are writes by definition (they happen inside
        # messages.create); the operation is baked into the endpoint token.
        endpoint = self.proxy.base_url(**self._static_tags(session_id), operation="write", client_id=f"letta-agent-{session_id}")
        return {
            "model": s["model"],
            "model_endpoint_type": "openai",
            "model_endpoint": endpoint,
            "context_window": int(s.get("context_window", 128000)),
            "handle": f"openai/{s['model']}",
            "temperature": float(s.get("temperature", 0.7)),
            "max_tokens": int(s.get("max_tokens", 4096)),
        }

    def _embedding_config(self, session_id: str) -> dict[str, Any]:
        e = self.settings.embedding
        # Embeddings serve both writes (passages) and reads (search): operation comes from scopes.
        endpoint = self.proxy.base_url(**self._static_tags(session_id), client_id="letta-embed")
        return {
            "embedding_model": e["model"],
            "embedding_endpoint_type": "openai",
            "embedding_endpoint": endpoint,
            "embedding_dim": int(e.get("embedding_dim", 1536)),
            "embedding_chunk_size": int(e.get("chunk_size", 300)),
            "handle": f"openai/{e['model']}",
        }

    async def start(self) -> None:
        if self._started:
            return
        self.proxy.reopen()
        self._client = AsyncLetta(base_url=self.settings.server_url, timeout=float(self.settings.agent.get("request_timeout_s", 600)))
        health = await self._health_raw()
        # Probe agent: capture the server-assigned system prompt and tool set for the fingerprint.
        probe = await self._create_agent("__probe__")
        try:
            self._probe = {
                "agent_type": str(getattr(probe, "agent_type", None)),
                "tools": sorted(t.name for t in (probe.tools or [])),
                "system_prompt_sha256": hashlib.sha256((probe.system or "").encode()).hexdigest(),
                "system_prompt_chars": len(probe.system or ""),
                "block_limits": {b.label: getattr(b, "limit", None) for b in (probe.memory.blocks if getattr(probe, "memory", None) else [])},
            }
        finally:
            await self._client.agents.delete(probe.id)
        self._probe["server_version"] = health.get("version")
        self._started = True
        self.emit("ADAPTER_START", fingerprint=self.config_fingerprint())

    async def _health_raw(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(self.settings.server_url.rstrip("/") + "/v1/health/")
            r.raise_for_status()
            return r.json()

    async def _create_agent(self, session_id: str):
        a = self.settings.agent
        blocks = [{"label": "human", "value": a.get("human_initial", "")}, {"label": "persona", "value": a.get("persona", "I am a helpful assistant with long-term memory.")}]
        kwargs: dict[str, Any] = {
            "name": self._agent_name(session_id),
            "memory_blocks": blocks,
            "llm_config": self._llm_config(session_id),
            "embedding_config": self._embedding_config(session_id),
            "include_base_tools": bool(a.get("include_base_tools", True)),
            "enable_sleeptime": bool(a.get("enable_sleeptime", False)),
            "tags": [f"run:{self.run_id}", "memharness"],
        }
        if a.get("agent_type"):
            kwargs["agent_type"] = a["agent_type"]
        if a.get("extra_tools"):
            kwargs["tools"] = list(a["extra_tools"])
        if a.get("context_window_limit"):
            kwargs["context_window_limit"] = int(a["context_window_limit"])
        return await self._client.agents.create(**kwargs)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        self._client = None
        self._started = False
        await self.proxy.aclose()

    async def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {"ok": False, "proxy": await self.proxy.healthy(), "started": self._started}
        try:
            h = await self._health_raw()
            info["server"] = h
            info["ok"] = bool(info["proxy"] and h.get("status") == "ok")
        except Exception as exc:  # noqa: BLE001
            info["server_error"] = f"{type(exc).__name__}: {exc}"
        return info

    async def _agent_for(self, session_id: str) -> str:
        if session_id not in self._agents:
            await self._reset_impl(session_id)
        return self._agents[session_id]

    @asynccontextmanager
    async def _scoped(self, op: str, session_id: str) -> AsyncIterator[AsyncLetta]:
        if not self._started:
            await self.start()
        async with self.proxy.scope("letta-embed", operation=op, session_id=session_id):
            yield self._client

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    async def _reset_impl(self, session_id: str) -> None:
        if not self._started:
            await self.start()
        async for ag in self._client.agents.list(name=self._agent_name(session_id)):
            await self._client.agents.delete(ag.id)
        agent = await self._create_agent(session_id)
        self._agents[session_id] = agent.id
        self._canary_passages.pop(session_id, None)
        passages = _page_items(await self._client.agents.passages.list(agent_id=agent.id, limit=1))
        if passages:
            raise AdapterError(f"fresh agent for {session_id!r} already has archival passages")

    async def _write_impl(self, session_id: str, messages: list[dict], *, metadata: dict[str, Any] | None) -> dict[str, Any]:
        agent_id = await self._agent_for(session_id)
        payload = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant", "system") and isinstance(m.get("content"), str) and m["content"].strip()]
        if not payload:
            return {"memories_added": 0, "memory_ids": [], "native": {"skipped": "no message content"}}
        async with self._scoped("write", session_id) as c:
            res = await c.agents.messages.create(agent_id=agent_id, messages=payload, max_steps=self.settings.max_steps)
        tool_calls: list[dict[str, Any]] = []
        returns: dict[str, str] = {}
        types: list[str] = []
        for m in res.messages:
            mt = getattr(m, "message_type", None)
            types.append(str(mt))
            if mt == "tool_call_message":
                tcs = getattr(m, "tool_calls", None) or ([getattr(m, "tool_call")] if getattr(m, "tool_call", None) else [])
                for tc in tcs:
                    tool_calls.append({"id": getattr(tc, "tool_call_id", None), "name": getattr(tc, "name", None), "arguments": (getattr(tc, "arguments", "") or "")[:500]})
            elif mt == "tool_return_message":
                returns[str(getattr(m, "tool_call_id", ""))] = str(getattr(m, "status", ""))
        memory_calls = [tc for tc in tool_calls if tc["name"] in MEMORY_TOOL_NAMES]
        ok_calls = [tc for tc in memory_calls if returns.get(str(tc["id"]), "success") != "error"]
        usage = getattr(res, "usage", None)
        stop = getattr(res, "stop_reason", None)
        native = {
            "message_types": types,
            "tool_calls": tool_calls,
            "stop_reason": getattr(stop, "stop_reason", str(stop)) if stop is not None else None,
            "steps": getattr(usage, "step_count", None),
            # System-reported usage: audit only, never used for cost analysis (the proxy is authoritative).
            "system_reported_usage": {k: getattr(usage, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if usage is not None else None,
        }
        return {"memories_added": len(ok_calls), "memories_updated": None, "memories_deleted": None, "memories_noop": None, "memory_ids": [tc["id"] for tc in ok_calls if tc["id"]], "native": native}

    async def _read_impl(self, session_id: str, query: str, *, operation: str) -> tuple[list[ReadHit], Any]:
        agent_id = await self._agent_for(session_id)
        op = "settle" if operation == "settle" else "read"
        hits: list[ReadHit] = []
        async with self._scoped(op, session_id) as c:
            if self.settings.include_core_blocks:
                blocks = [b async for b in c.agents.blocks.list(agent_id=agent_id)]
                for b in blocks:
                    if b.value and b.value.strip():
                        hits.append(ReadHit(memory_id=b.id, text=f"[{b.label}] {b.value.strip()}", score=None, metadata={"kind": "core_block", "label": b.label, "limit": getattr(b, "limit", None)}, created_at=str(getattr(b, "created_at", "") or "") or None))
            sr = await c.agents.passages.search(agent_id=agent_id, query=query, top_k=self.settings.archival_top_k)
        results = list(getattr(sr, "results", []) or [])
        for r in results:
            hits.append(ReadHit(memory_id=r.id, text=r.content, score=None, metadata={"kind": "archival", "tags": list(getattr(r, "tags", []) or [])}, created_at=str(getattr(r, "timestamp", "") or "") or None))
        return hits, {"core_blocks": sum(1 for h in hits if h.metadata and h.metadata.get("kind") == "core_block"), "archival": len(results), "top_k": self.settings.archival_top_k}

    async def _canary_write_impl(self, session_id: str, text: str) -> tuple[bool, list[str]]:
        agent_id = await self._agent_for(session_id)
        async with self._scoped("settle", session_id) as c:
            if self.settings.canary_mode == "message":
                res = await c.agents.messages.create(agent_id=agent_id, messages=[{"role": "user", "content": text}], max_steps=self.settings.max_steps)
                names = [getattr(tc, "name", None) for m in res.messages if getattr(m, "message_type", None) == "tool_call_message" for tc in (getattr(m, "tool_calls", None) or [getattr(m, "tool_call", None)]) if tc]
                stored = any(n in MEMORY_TOOL_NAMES for n in names)
                return stored, []
            created = await c.agents.passages.create(agent_id=agent_id, text=text)
        ids = [p.id for p in (created if isinstance(created, list) else [created]) if getattr(p, "id", None)]
        self._canary_passages[session_id] = ids
        return bool(ids), ids

    async def _canary_cleanup_impl(self, session_id: str, rec: CanaryRecord) -> None:
        agent_id = self._agents.get(session_id)
        if not agent_id:
            return
        async with self._scoped("settle", session_id) as c:
            for pid in self._canary_passages.pop(session_id, []) or rec.memory_ids:
                try:
                    await c.agents.passages.delete(pid, agent_id=agent_id)
                except Exception:  # noqa: BLE001
                    pass

    async def reset_all(self) -> None:
        self.emit("RESET_ALL_START")
        if not self._started:
            await self.start()
        n = 0
        async for ag in _aiter(self._client.agents.list()):
            if (ag.name or "").startswith(NAME_PREFIX):
                await self._client.agents.delete(ag.id)
                n += 1
        self._agents.clear()
        self._canary_passages.clear()
        self.emit("RESET_ALL_END", deleted_agents=n)

    # ------------------------------------------------------------------ #
    # Fingerprint
    # ------------------------------------------------------------------ #
    def config_fingerprint(self) -> dict[str, Any]:
        s = self.settings
        return {
            "system": self.name,
            "deployment": "self-hosted-letta-v1-server-archived",
            "product_note": "letta 0.16.8 is the last release of the retired Letta V1 API server (github.com/letta-ai/letta is now a landing page for letta-code); unsupported upstream",
            "system_version": self._probe.get("server_version") or "unknown-until-start",
            "system_commit": s.system_commit,
            "client_version": _dist_version("letta-client"),
            "adapter_version": ADAPTER_VERSION,
            "adapter_source_sha256": _source_hash(),
            "configuration_id": self.configuration_id,
            "memory_model": {k: v for k, v in s.llm.items() if k != "api_key"},
            "embedding_model": {k: v for k, v in s.embedding.items() if k != "api_key"},
            "agent": {**{k: v for k, v in s.agent.items()}, "probe": self._probe, "max_steps": s.max_steps},
            "write": {"path": "agents.messages.create (agent step; LLM decides memory tool calls)", "date_conveyance": "system-role message (message timestamps are ingestion time)"},
            "retrieval": {"core_blocks": s.include_core_blocks, "archival_top_k": s.archival_top_k, "recall": "not reachable via API in self-hosted 0.16.8 (messages/search needs Turbopuffer)"},
            "settlement": {"timeout_s": self.settlement_config.timeout_s, "poll_interval_s": self.settlement_config.poll_interval_s, "canary_cleanup": self.settlement_config.canary_cleanup, "canary_mode": s.canary_mode},
            "feature_flags": {"enable_sleeptime": bool(s.agent.get("enable_sleeptime", False)), "extra_tools": list(s.agent.get("extra_tools", [])), "telemetry": "none (no PostHog in letta 0.16.8; OTEL/Datadog opt-in)"},
            "prompts": {"system_prompt_sha256": self._probe.get("system_prompt_sha256")},
            "proxy": {"base_url": self.proxy.root, "client_ids": {"agent": "letta-agent-<session>", "embed": "letta-embed"}},
            "server_url": s.server_url,
            "seed": self.seed,
            "run_id": self.run_id,
            "python": platform.python_version(),
        }


def _page_items(res) -> list:
    """Items of a list response: a plain list, a page with .items, or a pydantic root list."""
    if isinstance(res, list):
        return res
    for attr in ("items", "root", "data"):
        v = getattr(res, attr, None)
        if isinstance(v, list):
            return v
    return []


async def _aiter(obj):
    """Iterate a Stainless page object (sync or async iterable)."""
    if hasattr(obj, "__aiter__"):
        async for x in obj:
            yield x
    else:
        for x in obj:
            yield x


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in ("base.py", "letta.py"):
        h.update((here / f).read_bytes())
    return h.hexdigest()
