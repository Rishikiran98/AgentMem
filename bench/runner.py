"""LongMemEval runner: ingestion -> settlement -> retrieval -> reader -> judge, one instance at a time.

The runner produces raw events only (append-only JSONL).  It never computes
metrics beyond a convenience tally in RUN_END; analysis rebuilds everything
from the events plus the proxy log.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from adapters.base import MemoryAdapter, utc_now_iso
from adapters.registry import build_adapter
from bench import BENCH_VERSION
from bench.judge import Judge, JudgeConfig
from bench.llm import ProxiedChat
from bench.longmemeval import Dataset, Instance, Session, select_instances, selection_digest
from bench.metadata import host_metadata
from bench.prompts import prompt_hashes
from bench.reader import Reader, ReaderConfig
from bench.resume import completed_cells
from proxy.logging import AppendOnlyJsonlLogger


@dataclass
class IngestConfig:
    granularity: str = "session"  # session | turn | pair (consecutive user/assistant pairs, Mem0's protocol)
    date_mode: str = "system_message"  # system_message | prefix_first_user | none
    strip_labels: bool = True
    skip_empty_pairs: bool = False  # Mem0's runner skips pairs where any message content is empty


@dataclass
class RunConfig:
    system: str
    benchmark: str
    seed: int
    config_path: str
    system_config: dict[str, Any]
    benchmark_config: dict[str, Any]
    proxy_url: str
    out_dir: Path
    limit: int | None = None
    question_ids: list[str] | None = None
    overrides: dict[str, Any] | None = None
    run_id: str = field(default_factory=lambda: "")
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = f"{self.system}-{self.benchmark}-seed{self.seed}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


def format_session_messages(session: Session, ingest: IngestConfig) -> list[dict[str, Any]]:
    """Turn a haystack session into the messages handed to the adapter.  Evidence labels never leak."""
    msgs = [{"role": t.role, "content": t.content} for t in session.turns]
    if not ingest.strip_labels:
        for m, t in zip(msgs, session.turns):
            if t.has_answer:
                m["has_answer"] = True
    if ingest.date_mode == "system_message":
        msgs.insert(0, {"role": "system", "content": f"This conversation took place on {session.date}."})
    elif ingest.date_mode == "prefix_first_user":
        for m in msgs:
            if m["role"] == "user":
                m["content"] = f"[{session.date}] {m['content']}"
                break
    elif ingest.date_mode != "none":
        raise ValueError(f"unknown ingest.date_mode {ingest.date_mode!r}")
    return msgs


def split_writes(session: Session, ingest: IngestConfig) -> list[list[dict[str, Any]]]:
    msgs = format_session_messages(session, ingest)
    if ingest.granularity == "session":
        return [msgs]
    system = [m for m in msgs if m["role"] == "system"]
    body = [m for m in msgs if m["role"] != "system"]
    if ingest.granularity == "turn":
        return [system + [m] for m in body]
    if ingest.granularity == "pair":
        pairs = [body[i: i + 2] for i in range(0, len(body), 2)]
        if ingest.skip_empty_pairs:
            pairs = [p for p in pairs if not any(not (m.get("content") or "").strip() for m in p)]
        return [system + p for p in pairs]
    raise ValueError(f"unknown ingest.granularity {ingest.granularity!r}")


class Runner:
    def __init__(self, cfg: RunConfig, dataset: Dataset) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.out_dir = Path(cfg.out_dir) / cfg.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        event_path = self.out_dir / "run.jsonl"
        if event_path.exists() and event_path.stat().st_size and not cfg.resume:
            raise ValueError(f"run {cfg.run_id!r} already exists; choose a new run_id or request exact resume")
        self._completed: set[tuple[str, str]] = set()
        if cfg.resume:
            self._completed = completed_cells(event_path, run_id=cfg.run_id, configuration_id=cfg.system_config["_configuration_id"])
        self.log = AppendOnlyJsonlLogger(event_path, fsync=True, instance_id=cfg.run_id)
        b = cfg.benchmark_config
        self.ingest = IngestConfig(**{k: v for k, v in b.get("ingest", {}).items() if k in IngestConfig.__dataclass_fields__})
        self.settle_policy = b.get("settlement", {}).get("policy", "per_instance")
        self.on_error = b.get("on_error", "continue")
        self.reader_cfg = ReaderConfig(**{k: v for k, v in b.get("reader", {}).items() if k in ReaderConfig.__dataclass_fields__})
        self.judge_cfg = JudgeConfig(**{k: v for k, v in b.get("judge", {}).items() if k in JudgeConfig.__dataclass_fields__})
        self._ctx: dict[str, Any] = {}  # instance context merged into adapter events
        self.adapter: MemoryAdapter | None = None
        self.reader: Reader | None = None
        self.judge: Judge | None = None
        self.tally = {"instances": 0, "correct": 0, "errors": 0}

    # ------------------------------------------------------------------ #
    def emit(self, event_type: str, **fields: Any) -> None:
        self.log.append({"event_type": event_type, "timestamp": utc_now_iso(), "run_id": self.cfg.run_id, "system": self.cfg.system, "seed": self.cfg.seed, **self._ctx, **fields})

    def _adapter_sink(self, ev: dict[str, Any]) -> None:
        self.log.append({**ev, "run_id": self.cfg.run_id, "seed": self.cfg.seed, "source": "adapter", **self._ctx})

    async def _proxy_info(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(self.cfg.proxy_url.rstrip("/") + "/_bench/info")
            r.raise_for_status()
            return r.json()

    # ------------------------------------------------------------------ #
    async def setup(self) -> list[Instance]:
        cfg = self.cfg
        proxy_info = await self._proxy_info()
        self.adapter = build_adapter(cfg.system_config, proxy_base_url=cfg.proxy_url, seed=cfg.seed, run_id=cfg.run_id, event_sink=self._adapter_sink, overrides=cfg.overrides)
        configuration_id = cfg.system_config["_configuration_id"]
        chat = ProxiedChat(cfg.proxy_url, system=cfg.system, configuration=configuration_id, seed=cfg.seed, run_id=cfg.run_id, client_id="harness")
        self.reader = Reader(chat, self.reader_cfg)
        self.judge = Judge(chat, self.judge_cfg)
        self._chat = chat
        instances = select_instances(self.dataset, seed=cfg.seed, limit=cfg.limit, question_ids=cfg.question_ids)
        fingerprint = self.adapter.config_fingerprint()
        manifest = {
            "run_id": cfg.run_id,
            "bench_version": BENCH_VERSION,
            "system": cfg.system,
            "benchmark": cfg.benchmark,
            "benchmark_version": cfg.benchmark_config.get("benchmark_version"),
            "protocol": cfg.benchmark_config.get("protocol", "longmemeval-official"),
            "seed": cfg.seed,
            "config_path": cfg.config_path,
            "configuration_id": configuration_id,
            "system_config": {k: v for k, v in cfg.system_config.items() if not k.startswith("_")},
            "benchmark_config": cfg.benchmark_config,
            "overrides": cfg.overrides,
            "overrides_present": bool(cfg.overrides),
            "dataset": {"name": self.dataset.name, "path": self.dataset.path, "sha256": self.dataset.sha256, "size_bytes": self.dataset.size_bytes, "instances_total": len(self.dataset.instances), "synthetic": self.dataset.synthetic},
            "selection": {"seed": cfg.seed, "limit": cfg.limit, "question_ids": cfg.question_ids, "n_selected": len(instances), "order_digest": selection_digest(instances), "ordered_question_ids": [i.question_id for i in instances]},
            "ingest": self.ingest.__dict__,
            "settlement_policy": self.settle_policy,
            "reader": self.reader_cfg.describe(),
            "judge": self.judge_cfg.describe(),
            "prompt_hashes": prompt_hashes(),
            "adapter_fingerprint": fingerprint,
            "proxy": {"url": cfg.proxy_url, "instance_id": proxy_info.get("proxy_instance_id"), "log_path": proxy_info.get("settings", {}).get("log_path"), "settings": proxy_info.get("settings"), "proxy_version": proxy_info.get("proxy_version")},
            "host": host_metadata(Path(__file__).resolve().parents[1]),
            "started_at": utc_now_iso(),
        }
        manifest_path = self.out_dir / "manifest.json"
        if cfg.resume:
            self.emit("RUN_RESUME", completed_cells=len(self._completed), configuration_id=configuration_id)
        else:
            manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
            self.emit("RUN_START", **manifest)
        await self.adapter.start()  # type: ignore[attr-defined]  (ADAPTER_START follows RUN_START)
        return instances

    async def run(self) -> dict[str, Any]:
        t_run = time.perf_counter()
        instances = await self.setup()
        try:
            for pos, inst in enumerate(instances):
                if (inst.question_id, "instance") in self._completed:
                    self.emit("CELL_SKIPPED", question_id=inst.question_id, phase="instance", reason="completed_exact_resume")
                    continue
                await self.run_instance(inst, pos)
                if self.tally["errors"] and self.on_error == "abort":
                    break
        finally:
            summary = {**self.tally, "accuracy_convenience": (self.tally["correct"] / self.tally["instances"]) if self.tally["instances"] else None, "duration_ms": _ms(t_run), "finished_at": utc_now_iso()}
            self._ctx = {}
            self.emit("RUN_END", **summary)
            await self.close()
        return summary

    async def close(self) -> None:
        if self.adapter is not None:
            await self.adapter.close()
        if getattr(self, "_chat", None) is not None:
            await self._chat.aclose()
        self.log.close()

    # ------------------------------------------------------------------ #
    async def run_instance(self, inst: Instance, position: int) -> None:
        assert self.adapter and self.reader and self.judge
        sid = inst.question_id
        self._ctx = {"question_id": sid, "question_type": inst.question_type, "position": position}
        t_inst = time.perf_counter()
        timings: dict[str, float | None] = {}
        self.emit("INSTANCE_START", n_sessions=len(inst.sessions), n_turns=inst.n_turns, is_abstention=inst.is_abstention, question_date=inst.question_date, answer_session_ids=inst.answer_session_ids)
        try:
            await self.adapter.reset(sid)

            # --- ingestion --------------------------------------------------- #
            t0 = time.perf_counter()
            n_writes = 0
            for sess in inst.sessions:
                self._ctx["lme_session_id"] = sess.session_id
                for msgs in split_writes(sess, self.ingest):
                    await self.adapter.write(sid, msgs, metadata={"lme_session_id": sess.session_id, "session_date": sess.date, "session_index": sess.index})
                    n_writes += 1
                if self.settle_policy == "per_session":
                    await self.adapter.wait_settled(sid)
            self._ctx.pop("lme_session_id", None)
            timings["ingest_ms"] = _ms(t0)
            self.emit("INGEST_END", n_sessions=len(inst.sessions), n_writes=n_writes, ingest_ms=timings["ingest_ms"])

            # --- settlement ---------------------------------------------------- #
            t0 = time.perf_counter()
            settled = None
            if self.settle_policy == "per_instance":
                st = await self.adapter.wait_settled(sid)
                settled = st.settled
            timings["settle_ms"] = _ms(t0)

            # --- retrieval ----------------------------------------------------- #
            t0 = time.perf_counter()
            rr = await self.adapter.search(sid, inst.question)
            timings["read_ms"] = rr.latency_ms
            self.emit("CONTEXT", read_id=rr.read_id, context=rr.context, context_chars=len(rr.context), hits=len(rr.hits), context_empty=(rr.context.strip() == ""))

            # --- reader --------------------------------------------------------- #
            self.emit("ANSWER_START", model=self.reader_cfg.model, context_chars=len(rr.context))
            rd = await self.reader.answer(context=rr.context, question_date=inst.question_date, question=inst.question, session_id=sid, hits=rr.hits)
            ans = rd.outcome
            timings["answer_ms"] = ans.latency_ms
            self.emit("ANSWER_END", model=ans.model, answer=rd.answer, answer_raw=rd.answer_raw if rd.answer_raw != rd.answer else None, prompt_sha256=ans.prompt_sha256, prompt_chars=len(rd.prompt), proxy_request_id=ans.proxy_request_id, upstream_request_id=ans.upstream_request_id, latency_ms=ans.latency_ms, finish_reason=ans.finish_reason)

            # --- judge ---------------------------------------------------------- #
            self.emit("JUDGE_START", model=self.judge_cfg.model)
            v = await self.judge.judge(question_type=inst.question_type, question=inst.question, answer=inst.answer, response=rd.answer, abstention=inst.is_abstention, session_id=sid)
            timings["judge_ms"] = v.outcome.latency_ms
            self.emit("JUDGE_END", model=v.outcome.model, verdict_raw=v.raw, correct=v.correct, gold_answer=inst.answer, prompt_sha256=v.outcome.prompt_sha256, proxy_request_id=v.outcome.proxy_request_id, latency_ms=v.outcome.latency_ms)

            self.tally["instances"] += 1
            self.tally["correct"] += int(v.correct)
            self.emit("INSTANCE_END", correct=v.correct, error=None, settled=settled, total_ms=_ms(t_inst), **timings)
        except Exception as exc:  # noqa: BLE001 - recorded, then continue or abort per config
            self.tally["instances"] += 1
            self.tally["errors"] += 1
            self.emit("ERROR", phase="instance", error_type=type(exc).__name__, error_message=str(exc)[:4000])
            self.emit("INSTANCE_END", correct=None, error=type(exc).__name__, total_ms=_ms(t_inst), **timings)
            if self.on_error == "abort":
                raise
        finally:
            self._ctx = {}
