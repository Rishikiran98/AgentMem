"""Mem0 (open-source ``mem0ai``) adapter.

Deployment shape: Mem0 runs *in-process* (its documented OSS mode) with a
Qdrant vector store (server via compose/mem0, or embedded for tests) and its
SQLite history database.  All model traffic goes to the benchmark proxy via
Mem0's documented ``openai_base_url`` / ``api_key`` settings.

Translation of benchmark semantics to Mem0's API (mem0ai 2.0.x):

    reset(session)   -> AsyncMemory.delete_all(user_id=session); verified empty via get_all
    write(session)   -> AsyncMemory.add(messages, user_id=session, infer=cfg.infer)
    read(session, q) -> AsyncMemory.search(q, filters={"user_id": session}, top_k, threshold, rerank=False)
                        context = retrieved memory texts, one per line, in Mem0's order
    canary           -> AsyncMemory.add([{"role":"user","content": <canary>}], user_id=session, infer=cfg.canary_infer)
                        polled through the same search() call as read(); cleaned up with delete(memory_id)
    reset_all()      -> AsyncMemory.reset() on a throw-away instance (drops collection + history)

Attribution: one AsyncMemory instance per benchmark operation type (write /
read / settle), each configured with a proxy API-key token carrying the
operation, so operation attribution is exact even under concurrency.  The
session is attached with a proxy scope around each call.  The three instances
share one Qdrant client (documented ``client`` config option) and one history
DB, so they are the same Mem0 store, not three.

Nothing here reranks, filters, deduplicates or summarises.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import platform
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

# Mem0 reads MEM0_TELEMETRY at import time; PostHog telemetry is not part of the benchmark.
os.environ.setdefault("MEM0_TELEMETRY", "false")

import httpx  # noqa: E402
from mem0 import AsyncMemory  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from adapters.base import AdapterError, CanaryRecord, EventSink, MemoryAdapter, ReadHit, SettlementConfig  # noqa: E402
from proxy.client import ProxyClient  # noqa: E402

ADAPTER_VERSION = "0.1.0"
OPERATIONS = ("write", "read", "settle")


@dataclass
class Mem0Settings:
    llm: dict[str, Any]
    embedder: dict[str, Any]
    vector_store: dict[str, Any]
    history_db_path: str
    infer: bool = True
    custom_instructions: str | None = None
    top_k: int = 20
    threshold: float | None = 0.1
    rerank: bool = False
    canary_infer: bool = False
    system_commit: str | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Mem0Settings":
        m = cfg["mem0"]
        r = cfg.get("retrieval", {})
        s = cfg.get("settlement", {})
        return cls(
            llm=dict(m["llm"]),
            embedder=dict(m["embedder"]),
            vector_store=dict(m["vector_store"]),
            history_db_path=m["history_db_path"],
            infer=bool(m.get("infer", True)),
            custom_instructions=m.get("custom_instructions"),
            top_k=int(r.get("top_k", 20)),
            threshold=r.get("threshold", 0.1),
            rerank=bool(r.get("rerank", False)),
            canary_infer=bool(s.get("canary_infer", False)),
            system_commit=cfg.get("system_commit"),
        )


class Mem0Adapter(MemoryAdapter):
    name = "mem0"
    adapter_version = ADAPTER_VERSION

    def __init__(
        self,
        settings: Mem0Settings,
        *,
        proxy_base_url: str,
        configuration_id: str,
        seed: int,
        run_id: str,
        settlement: SettlementConfig | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        super().__init__(settlement=settlement, event_sink=event_sink)
        self.settings = settings
        self.proxy = ProxyClient(proxy_base_url)
        self.configuration_id = configuration_id
        self.seed = seed
        self.run_id = run_id
        self._client: QdrantClient | None = None
        self._mem: dict[str, AsyncMemory] = {}
        self._started = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _client_id(self, op: str) -> str:
        return f"mem0-{op}"

    def _vector_store_mode(self) -> str:
        vs = self.settings.vector_store
        return "embedded" if vs.get("mode", "embedded" if vs.get("path") else "server") == "embedded" else "server"

    def _new_qdrant_client(self) -> QdrantClient:
        vs = self.settings.vector_store
        if self._vector_store_mode() == "embedded":
            Path(vs["path"]).mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=vs["path"])
        if vs.get("url"):
            return QdrantClient(url=vs["url"], api_key=vs.get("api_key"))
        return QdrantClient(host=vs.get("host", "127.0.0.1"), port=int(vs.get("port", 6333)), api_key=vs.get("api_key"))

    def _mem0_config(self, op: str, *, client: QdrantClient | None) -> dict[str, Any]:
        s = self.settings
        key = self.proxy.api_key(system=self.name, configuration=self.configuration_id, seed=self.seed, run_id=self.run_id, client_id=self._client_id(op), operation=op)
        base_url = self.proxy.base_url()
        llm_cfg = {k: v for k, v in s.llm.items() if k != "provider"}
        llm_cfg.update({"api_key": key, "openai_base_url": base_url})
        emb_cfg = {k: v for k, v in s.embedder.items() if k != "provider"}
        emb_cfg.update({"api_key": key, "openai_base_url": base_url})
        vs = s.vector_store
        vs_cfg: dict[str, Any] = {"collection_name": vs["collection_name"], "embedding_model_dims": int(vs["embedding_model_dims"]), "on_disk": bool(vs.get("on_disk", True))}
        if client is not None:
            vs_cfg["client"] = client
        elif self._vector_store_mode() == "embedded":
            vs_cfg["path"] = vs["path"]
        else:
            for k in ("host", "port", "url", "api_key"):
                if vs.get(k) is not None:
                    vs_cfg[k] = vs[k]
        cfg: dict[str, Any] = {
            "llm": {"provider": s.llm.get("provider", "openai"), "config": llm_cfg},
            "embedder": {"provider": s.embedder.get("provider", "openai"), "config": emb_cfg},
            "vector_store": {"provider": vs.get("provider", "qdrant"), "config": vs_cfg},
            "history_db_path": s.history_db_path,
        }
        if s.custom_instructions:
            cfg["custom_instructions"] = s.custom_instructions
        return cfg

    @staticmethod
    async def _from_config(cfg: dict[str, Any]) -> AsyncMemory:
        res = AsyncMemory.from_config(cfg)
        if hasattr(res, "__await__"):
            res = await res
        return res

    async def start(self) -> None:
        if self._started:
            return
        Path(self.settings.history_db_path).parent.mkdir(parents=True, exist_ok=True)
        self.proxy.reopen()
        self._client = self._new_qdrant_client()
        for op in OPERATIONS:  # sequential: SQLite migrations must not race
            self._mem[op] = await self._from_config(self._mem0_config(op, client=self._client))
        self._started = True
        self.emit("ADAPTER_START", fingerprint=self.config_fingerprint())

    async def close(self) -> None:
        for m in self._mem.values():
            try:
                m.db.close()
            except Exception:  # noqa: BLE001
                pass
        self._mem.clear()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        self._started = False
        await self.proxy.aclose()

    async def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {"ok": False, "proxy": await self.proxy.healthy(), "started": self._started}
        try:
            cols = [c.name for c in self._client.get_collections().collections] if self._client else []
            info["qdrant_collections"] = cols
            info["ok"] = info["proxy"] and self._started
        except Exception as exc:  # noqa: BLE001
            info["qdrant_error"] = f"{type(exc).__name__}: {exc}"
        return info

    @asynccontextmanager
    async def _scoped(self, op: str, session_id: str) -> AsyncIterator[AsyncMemory]:
        if not self._started:
            await self.start()
        async with self.proxy.scope(self._client_id(op), session_id=session_id):
            yield self._mem[op]

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    async def _reset_impl(self, session_id: str) -> None:
        async with self._scoped("write", session_id) as mem:
            await mem.delete_all(user_id=session_id)
            remaining = await mem.get_all(filters={"user_id": session_id}, top_k=1)
        left = remaining.get("results", [])
        if left:
            raise AdapterError(f"reset left {len(left)}+ memories for session {session_id!r}")

    async def _write_impl(self, session_id: str, messages: list[dict], *, metadata: dict[str, Any] | None) -> dict[str, Any]:
        async with self._scoped("write", session_id) as mem:
            res = await mem.add(messages, user_id=session_id, metadata=metadata, infer=self.settings.infer)
        results = res.get("results", []) if isinstance(res, dict) else []
        counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
        ids: list[str] = []
        for r in results:
            counts[r.get("event", "NONE")] = counts.get(r.get("event", "NONE"), 0) + 1
            if r.get("id"):
                ids.append(str(r["id"]))
        return {"memories_added": counts["ADD"], "memories_updated": counts["UPDATE"], "memories_deleted": counts["DELETE"], "memories_noop": counts["NONE"], "memory_ids": ids, "native": res}

    async def _read_impl(self, session_id: str, query: str, *, operation: str) -> tuple[list[ReadHit], Any]:
        op = "settle" if operation == "settle" else "read"
        kwargs: dict[str, Any] = {"filters": {"user_id": session_id}, "top_k": self.settings.top_k, "rerank": self.settings.rerank}
        if self.settings.threshold is not None:
            kwargs["threshold"] = self.settings.threshold
        async with self._scoped(op, session_id) as mem:
            res = await mem.search(query, **kwargs)
        results = res.get("results", []) if isinstance(res, dict) else []
        hits = [ReadHit(memory_id=str(r.get("id")) if r.get("id") else None, text=str(r.get("memory", "")), score=r.get("score"), metadata=r.get("metadata")) for r in results]
        return hits, res

    async def _canary_write_impl(self, session_id: str, text: str) -> tuple[bool, list[str]]:
        async with self._scoped("settle", session_id) as mem:
            res = await mem.add([{"role": "user", "content": text}], user_id=session_id, infer=self.settings.canary_infer)
        results = res.get("results", []) if isinstance(res, dict) else []
        ids = [str(r["id"]) for r in results if r.get("id")]
        return bool(ids), ids

    async def _canary_cleanup_impl(self, session_id: str, rec: CanaryRecord) -> None:
        async with self._scoped("settle", session_id) as mem:
            for mid in rec.memory_ids:
                await mem.delete(mid)

    async def reset_all(self) -> None:
        """Drop the collection and history via Mem0's own reset(), then rebuild the instances."""
        self.emit("RESET_ALL_START")
        was_started = self._started
        if was_started:
            for m in self._mem.values():
                m.db.close()
            self._mem.clear()
            if self._client is not None:
                self._client.close()
                self._client = None
            self._started = False
        tmp = await self._from_config(self._mem0_config("write", client=None))
        try:
            await tmp.reset()
            # Mem0's reset() only touches the main collection; drop the entity collection too.
            client = tmp.vector_store.client
            main = self.settings.vector_store["collection_name"]
            for col in client.get_collections().collections:
                if col.name != main and col.name.startswith(main):
                    client.delete_collection(col.name)
        finally:
            try:
                tmp.db.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                tmp.vector_store.client.close()
            except Exception:  # noqa: BLE001
                pass
        self.emit("RESET_ALL_END")
        if was_started:
            await self.start()

    # ------------------------------------------------------------------ #
    # Fingerprint
    # ------------------------------------------------------------------ #
    def config_fingerprint(self) -> dict[str, Any]:
        s = self.settings
        vs = s.vector_store
        mode = self._vector_store_mode()
        llm = {k: v for k, v in s.llm.items() if k != "api_key"}
        emb = {k: v for k, v in s.embedder.items() if k != "api_key"}
        return {
            "system": self.name,
            "deployment": "self-hosted-oss-inprocess",
            "system_version": _dist_version("mem0ai"),
            "system_commit": s.system_commit,
            "system_pipeline": "mem0 v3 additive extraction (mem0ai 2.x)",
            "adapter_version": ADAPTER_VERSION,
            "adapter_source_sha256": _source_hash(),
            "configuration_id": self.configuration_id,
            "memory_model": llm,
            "embedding_model": emb,
            "vector_store": {
                "provider": vs.get("provider", "qdrant"),
                "mode": mode,
                "collection_name": vs["collection_name"],
                "embedding_model_dims": vs["embedding_model_dims"],
                "on_disk": vs.get("on_disk", True),
                "endpoint": vs.get("url") or (f"{vs.get('host', '127.0.0.1')}:{vs.get('port', 6333)}" if mode == "server" else vs.get("path")),
                "server_version": _qdrant_server_version(vs) if mode == "server" else None,
                "client_version": _dist_version("qdrant-client"),
            },
            "history_db_path": s.history_db_path,
            "write": {"infer": s.infer, "custom_instructions": s.custom_instructions},
            "retrieval": {"top_k": s.top_k, "threshold": s.threshold, "rerank": s.rerank},
            "settlement": {"timeout_s": self.settlement_config.timeout_s, "poll_interval_s": self.settlement_config.poll_interval_s, "canary_cleanup": self.settlement_config.canary_cleanup, "canary_infer": s.canary_infer},
            "feature_flags": _feature_flags(),
            "prompts": _prompt_hashes(),
            "proxy": {"base_url": self.proxy.root, "client_ids": {op: self._client_id(op) for op in OPERATIONS}},
            "seed": self.seed,
            "run_id": self.run_id,
            "python": platform.python_version(),
        }


# ---------------------------------------------------------------------- #
# Fingerprint helpers
# ---------------------------------------------------------------------- #

def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in ("base.py", "mem0.py"):
        h.update((here / f).read_bytes())
    return h.hexdigest()


def _feature_flags() -> dict[str, Any]:
    spacy_model = False
    spacy_installed = importlib.util.find_spec("spacy") is not None
    if spacy_installed:
        try:
            import spacy  # type: ignore

            spacy_model = bool(spacy.util.is_package("en_core_web_sm"))
        except Exception:  # noqa: BLE001
            spacy_model = False
    fastembed = importlib.util.find_spec("fastembed") is not None
    return {
        "spacy_installed": spacy_installed,
        "spacy_en_core_web_sm": spacy_model,
        "entity_extraction_enabled": spacy_model,
        "fastembed_installed": fastembed,
        "bm25_hybrid_search_enabled": fastembed,  # Mem0 disables BM25 when fastembed is missing
        "telemetry": os.environ.get("MEM0_TELEMETRY", "True"),
        "posthog_installed": importlib.util.find_spec("posthog") is not None,
    }


def _prompt_hashes() -> dict[str, str | None]:
    try:
        from mem0.configs import prompts
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str | None] = {}
    for name in ("ADDITIVE_EXTRACTION_PROMPT", "FACT_RETRIEVAL_PROMPT", "DEFAULT_UPDATE_MEMORY_PROMPT", "MEMORY_ANSWER_PROMPT"):
        text = getattr(prompts, name, None)
        out[name.lower() + "_sha256"] = hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None
    return out


def _qdrant_server_version(vs: dict[str, Any]) -> str | None:
    url = vs.get("url") or f"http://{vs.get('host', '127.0.0.1')}:{vs.get('port', 6333)}"
    try:
        r = httpx.get(url, timeout=3.0)
        return r.json().get("version")
    except Exception:  # noqa: BLE001
        return None
