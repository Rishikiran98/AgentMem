"""Zep / Graphiti adapter.

Which "Zep" is benchmarked
--------------------------
Zep ships as a hosted service (Zep Cloud) whose model calls happen inside
Zep's infrastructure and therefore cannot be routed through the benchmark
proxy; Zep Community Edition is archived.  Graphiti (``graphiti-core``) is
Zep's open-source temporal knowledge-graph engine and the component the Zep
paper (arXiv 2501.13956) describes.  This adapter benchmarks **Graphiti,
self-hosted, in-process**, with a dedicated graph database (Neo4j via
compose/zep; Kuzu embedded for tests).  It is labelled ``deployment:
self-hosted-graphiti-inprocess`` everywhere so it is never conflated with the
hosted product.

Translation of benchmark semantics (graphiti-core 0.30.x):

    reset(session)   -> clear_data(driver, [session]); verified empty via *.get_by_group_ids
    write(session)   -> one add_episode() per chat message (Zep's own LongMemEval evaluation
                        ingests per message: episode_body=f"{role}: {content}",
                        source=EpisodeType.message, reference_time=session date, group_id=session)
    read(session, q) -> graphiti.search(q, group_ids=[session], num_results=top_k)   (EDGE_HYBRID_SEARCH_RRF)
                        or search_(q, config=<recipe>) when a recipe with a reranker is configured;
                        context = one fact per line, in Graphiti's order, optionally with its validity dates
    canary           -> add_triplet(): a fact inserted through the documented direct-write API (no LLM
                        extraction, so it is deterministic), polled through the same search call
    reset_all()      -> clear_data(driver) + build_indices_and_constraints(delete_existing=True)

Session dates: the runner's system message carrying the date is *not* ingested
(Graphiti has no system-message episode); the date is conveyed through the
documented ``reference_time`` parameter from the write metadata instead.

Attribution: three Graphiti objects (write / read / settle) share one graph
driver; each owns LLM/embedder/reranker clients configured with a proxy token
carrying its operation.  Sessions are attached with proxy scopes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

# Graphiti reads its telemetry switch at import time; PostHog analytics are not part of the benchmark.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import GraphitiError
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search import search_config_recipes as recipes
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from adapters.base import AdapterError, CanaryRecord, EventSink, MemoryAdapter, ReadHit, SettlementConfig
from proxy.client import ProxyClient

ADAPTER_VERSION = "0.1.0"
OPERATIONS = ("write", "read", "settle")
RECIPES = {
    "edge_hybrid_rrf": None,  # graphiti.search() default
    "edge_hybrid_cross_encoder": recipes.EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    "edge_hybrid_mmr": recipes.EDGE_HYBRID_SEARCH_MMR,
    "combined_hybrid_rrf": recipes.COMBINED_HYBRID_SEARCH_RRF,
    "combined_hybrid_cross_encoder": recipes.COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
}


@dataclass
class ZepSettings:
    graph_store: dict[str, Any]
    llm: dict[str, Any]
    embedder: dict[str, Any]
    reranker: dict[str, Any]
    episode_granularity: str = "message"  # message | batch
    source_description: str = "chat message"
    recipe: str = "edge_hybrid_rrf"
    top_k: int = 10
    fact_format: str = "fact_with_dates"  # fact_with_dates | fact_only
    canary_mode: str = "triplet"  # triplet | episode
    system_commit: str | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ZepSettings":
        g = cfg["graphiti"]
        r = cfg.get("retrieval", {})
        s = cfg.get("settlement", {})
        return cls(
            graph_store=dict(g["graph_store"]),
            llm=dict(g["llm"]),
            embedder=dict(g["embedder"]),
            reranker=dict(g.get("reranker", {"provider": "openai"})),
            episode_granularity=g.get("ingest", {}).get("episode_granularity", "message"),
            source_description=g.get("ingest", {}).get("source_description", "chat message"),
            recipe=r.get("recipe", "edge_hybrid_rrf"),
            top_k=int(r.get("top_k", 10)),
            fact_format=r.get("fact_format", "fact_with_dates"),
            canary_mode=s.get("canary_mode", "triplet"),
            system_commit=cfg.get("system_commit"),
        )


_LME_DATE = re.compile(r"^(\d{4})/(\d{2})/(\d{2}) \([A-Za-z]{3}\) (\d{2}):(\d{2})$")


def parse_reference_time(value: Any) -> datetime | None:
    """LongMemEval '2023/05/20 (Sat) 02:21' or ISO-8601 -> aware UTC datetime; None if unparseable."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    m = _LME_DATE.match(value.strip())
    if m:
        y, mo, d, h, mi = (int(x) for x in m.groups())
        return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class ZepAdapter(MemoryAdapter):
    name = "zep"
    adapter_version = ADAPTER_VERSION

    def __init__(self, settings: ZepSettings, *, proxy_base_url: str, configuration_id: str, seed: int, run_id: str, settlement: SettlementConfig | None = None, event_sink: EventSink | None = None) -> None:
        super().__init__(settlement=settlement, event_sink=event_sink)
        self.settings = settings
        self.proxy = ProxyClient(proxy_base_url)
        self.configuration_id = configuration_id
        self.seed = seed
        self.run_id = run_id
        self._driver = None
        self._g: dict[str, Graphiti] = {}
        self._started = False
        self._canary_nodes: dict[str, list[str]] = {}
        if settings.recipe not in RECIPES:
            raise ValueError(f"unknown retrieval recipe {settings.recipe!r}; choose from {sorted(RECIPES)}")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _client_id(self, op: str) -> str:
        return f"zep-{op}"

    def _new_driver(self):
        gs = self.settings.graph_store
        provider = gs.get("provider", "neo4j")
        if provider == "kuzu":
            from graphiti_core.driver.kuzu_driver import KuzuDriver

            Path(gs["path"]).parent.mkdir(parents=True, exist_ok=True)
            drv = KuzuDriver(db=gs["path"])
            if not hasattr(drv, "_database"):
                # graphiti-core 0.30.1 declares GraphDriver._database but the Kuzu driver never
                # sets it; add_episode() reads it before calling the (no-op) clone().  Tests only.
                drv._database = "kuzu"
            return drv
        if provider == "neo4j":
            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            return Neo4jDriver(uri=gs["uri"], user=gs.get("user"), password=gs.get("password"), database=gs.get("database", "neo4j"))
        if provider == "falkordb":
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            return FalkorDriver(host=gs.get("host", "127.0.0.1"), port=int(gs.get("port", 6379)), username=gs.get("user"), password=gs.get("password"), database=gs.get("database", "default_db"))
        raise ValueError(f"unsupported graph_store.provider {provider!r}")

    def _graphiti(self, op: str, driver) -> Graphiti:
        s = self.settings
        key = self.proxy.api_key(system=self.name, configuration=self.configuration_id, seed=self.seed, run_id=self.run_id, client_id=self._client_id(op), operation=op)
        base_url = self.proxy.base_url()
        llm_cfg = LLMConfig(api_key=key, base_url=base_url, model=s.llm.get("model"), small_model=s.llm.get("small_model"), temperature=float(s.llm.get("temperature", 0.0)), max_tokens=int(s.llm.get("max_tokens", 16384)))
        client_kind = s.llm.get("client", "openai_responses")
        if client_kind == "openai_responses":
            llm = OpenAIClient(config=llm_cfg)
        elif client_kind == "openai_generic":
            llm = OpenAIGenericClient(config=llm_cfg)
        else:
            raise ValueError(f"unknown llm.client {client_kind!r}")
        emb = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key=key, base_url=base_url, embedding_model=s.embedder.get("model", "text-embedding-3-small"), embedding_dim=int(s.embedder.get("embedding_dim", 1024))))
        rer_provider = s.reranker.get("provider", "openai")
        if rer_provider == "openai":
            cross = OpenAIRerankerClient(config=LLMConfig(api_key=key, base_url=base_url, model=s.reranker.get("model") or s.llm.get("small_model") or s.llm.get("model")))
        elif rer_provider == "none":
            cross = None
        else:
            raise ValueError(f"unknown reranker.provider {rer_provider!r}")
        return Graphiti(graph_driver=driver, llm_client=llm, embedder=emb, cross_encoder=cross)

    async def start(self) -> None:
        if self._started:
            return
        self.proxy.reopen()
        self._driver = self._new_driver()
        for op in OPERATIONS:
            self._g[op] = self._graphiti(op, self._driver)
        await self._g["write"].build_indices_and_constraints()
        await self._ensure_kuzu_fulltext_indexes()
        self._started = True
        self.emit("ADAPTER_START", fingerprint=self.config_fingerprint())

    async def _ensure_kuzu_fulltext_indexes(self) -> None:
        """Kuzu only (tests): graphiti-core 0.30.1 creates Kuzu FTS indexes concurrently on a
        single-query connection and logs-and-drops the failures, leaving searches to error with
        "doesn't have an index".  Re-issue Graphiti's own index DDL sequentially for any missing one."""
        if self.settings.graph_store.get("provider") != "kuzu":
            return
        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_fulltext_indices

        rows, _, _ = await self._driver.execute_query("CALL SHOW_INDEXES() RETURN *")
        present = {r.get("index_name") for r in rows or []}
        for ddl in get_fulltext_indices(GraphProvider.KUZU):
            name = ddl.split("'")[3]
            if name not in present:
                await self._driver.execute_query(ddl)

    async def close(self) -> None:
        if self._driver is not None:
            try:
                await self._driver.close()
            except Exception:  # noqa: BLE001
                pass
        self._driver = None
        self._g.clear()
        self._started = False
        await self.proxy.aclose()

    async def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {"ok": False, "proxy": await self.proxy.healthy(), "started": self._started}
        try:
            if self._driver is not None:
                await self._count_group("__health__")
                info["graph"] = "reachable"
            info["ok"] = bool(info["proxy"] and self._started)
        except Exception as exc:  # noqa: BLE001
            info["graph_error"] = f"{type(exc).__name__}: {exc}"
        return info

    @asynccontextmanager
    async def _scoped(self, op: str, session_id: str) -> AsyncIterator[Graphiti]:
        if not self._started:
            await self.start()
        async with self.proxy.scope(self._client_id(op), session_id=session_id):
            yield self._g[op]

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #
    async def _count_group(self, session_id: str) -> int:
        """Rows of any kind left in a group.  Graphiti raises Groups*NotFoundError for an empty group."""
        left = 0
        for cls in (EntityNode, EpisodicNode, EntityEdge):
            try:
                left += len(await cls.get_by_group_ids(self._driver, [session_id], limit=1))
            except GraphitiError:
                pass
        return left

    async def _reset_impl(self, session_id: str) -> None:
        if not self._started:
            await self.start()
        await clear_data(self._driver, [session_id])
        left = await self._count_group(session_id)
        if left:
            raise AdapterError(f"reset left data for session {session_id!r}")
        self._canary_nodes.pop(session_id, None)

    async def _write_impl(self, session_id: str, messages: list[dict], *, metadata: dict[str, Any] | None) -> dict[str, Any]:
        meta = metadata or {}
        ref = parse_reference_time(meta.get("session_date")) or datetime.now(timezone.utc)
        base = str(meta.get("lme_session_id") or "write")
        chat = [m for m in messages if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip()]
        episodes: list[tuple[str, str]] = []
        if self.settings.episode_granularity == "message":
            episodes = [(f"{base}:{i}", f"{m['role']}: {m['content']}") for i, m in enumerate(chat)]
        elif self.settings.episode_granularity == "batch":
            episodes = [(base, "\n".join(f"{m['role']}: {m['content']}" for m in chat))] if chat else []
        else:
            raise ValueError(f"unknown episode_granularity {self.settings.episode_granularity!r}")
        edge_ids: list[str] = []
        n_nodes = 0
        native: list[dict[str, Any]] = []
        async with self._scoped("write", session_id) as g:
            for name, body in episodes:
                res = await g.add_episode(name=name, episode_body=body, source=EpisodeType.message, source_description=self.settings.source_description, reference_time=ref, group_id=session_id)
                edge_ids += [e.uuid for e in res.edges]
                n_nodes += len(res.nodes)
                native.append({"episode_uuid": res.episode.uuid, "nodes": len(res.nodes), "edges": len(res.edges), "facts": [e.fact for e in res.edges]})
        return {"memories_added": len(edge_ids), "memories_updated": None, "memories_deleted": None, "memories_noop": None, "memory_ids": edge_ids, "native": {"episodes": native, "entity_nodes": n_nodes, "reference_time": ref.isoformat()}}

    def _format_fact(self, e: EntityEdge) -> str:
        if self.settings.fact_format == "fact_only":
            return e.fact
        va = e.valid_at.date().isoformat() if e.valid_at else "unknown"
        ia = e.invalid_at.date().isoformat() if e.invalid_at else "present"
        return f"{e.fact} (valid: {va} - {ia})"

    async def _read_impl(self, session_id: str, query: str, *, operation: str) -> tuple[list[ReadHit], Any]:
        op = "settle" if operation == "settle" else "read"
        recipe = RECIPES[self.settings.recipe]
        async with self._scoped(op, session_id) as g:
            if recipe is None:
                edges = await g.search(query, group_ids=[session_id], num_results=self.settings.top_k)
                nodes = []
            else:
                cfg = recipe.model_copy(deep=True)
                cfg.limit = self.settings.top_k
                res = await g.search_(query, config=cfg, group_ids=[session_id])
                edges, nodes = list(res.edges), list(res.nodes)
        hits = [ReadHit(memory_id=e.uuid, text=self._format_fact(e), score=None, metadata={"kind": "fact", "name": e.name, "valid_at": e.valid_at.isoformat() if e.valid_at else None, "invalid_at": e.invalid_at.isoformat() if e.invalid_at else None, "expired_at": e.expired_at.isoformat() if e.expired_at else None, "episodes": list(e.episodes)}, created_at=e.created_at.isoformat() if e.created_at else None) for e in edges]
        hits += [ReadHit(memory_id=n.uuid, text=f"{n.name}: {n.summary}" if n.summary else n.name, score=None, metadata={"kind": "entity", "labels": list(n.labels)}, created_at=n.created_at.isoformat() if n.created_at else None) for n in nodes]
        return hits, {"edges": len(edges), "nodes": len(nodes), "recipe": self.settings.recipe}

    async def _canary_write_impl(self, session_id: str, text: str) -> tuple[bool, list[str]]:
        now = datetime.now(timezone.utc)
        async with self._scoped("settle", session_id) as g:
            if self.settings.canary_mode == "episode":
                res = await g.add_episode(name="memharness-canary", episode_body=f"user: {text}", source=EpisodeType.message, source_description=self.settings.source_description, reference_time=now, group_id=session_id)
                return bool(res.edges), [e.uuid for e in res.edges]
            token = text.rsplit(" ", 1)[-1].rstrip(".")
            src = EntityNode(name="benchmark harness", group_id=session_id, labels=["Entity"], created_at=now)
            dst = EntityNode(name=token, group_id=session_id, labels=["Entity"], created_at=now)
            edge = EntityEdge(group_id=session_id, source_node_uuid=src.uuid, target_node_uuid=dst.uuid, created_at=now, name="HAS_VERIFICATION_CODE", fact=text, episodes=[], valid_at=now)
            res = await g.add_triplet(src, edge, dst)
        edge_ids = [e.uuid for e in getattr(res, "edges", [])] or [edge.uuid]
        self._canary_nodes[session_id] = [n.uuid for n in getattr(res, "nodes", [])] or [src.uuid, dst.uuid]
        return bool(edge_ids), edge_ids

    async def _canary_cleanup_impl(self, session_id: str, rec: CanaryRecord) -> None:
        async with self._scoped("settle", session_id):
            for uid in rec.memory_ids:
                try:
                    e = await EntityEdge.get_by_uuid(self._driver, uid)
                    await e.delete(self._driver)
                except Exception:  # noqa: BLE001 - already gone
                    pass
            for uid in self._canary_nodes.pop(session_id, []):
                try:
                    n = await EntityNode.get_by_uuid(self._driver, uid)
                    if n.name == "benchmark harness" or n.name.startswith("memharness-canary-"):
                        await n.delete(self._driver)
                except Exception:  # noqa: BLE001
                    pass

    async def reset_all(self) -> None:
        self.emit("RESET_ALL_START")
        if not self._started:
            await self.start()
        await clear_data(self._driver)
        await self._g["write"].build_indices_and_constraints(delete_existing=True)
        await self._ensure_kuzu_fulltext_indexes()
        self._canary_nodes.clear()
        self.emit("RESET_ALL_END")

    # ------------------------------------------------------------------ #
    # Fingerprint
    # ------------------------------------------------------------------ #
    def config_fingerprint(self) -> dict[str, Any]:
        s = self.settings
        gs = s.graph_store
        provider = gs.get("provider", "neo4j")
        store: dict[str, Any] = {"provider": provider, "deprecated_backend": provider == "kuzu"}
        if provider == "kuzu":
            store.update({"path": gs.get("path"), "kuzu_version": _dist_version("kuzu")})
        elif provider == "neo4j":
            store.update({"uri": gs.get("uri"), "database": gs.get("database", "neo4j"), "neo4j_driver_version": _dist_version("neo4j"), "server_version": _neo4j_server_version(gs)})
        else:
            store.update({k: gs.get(k) for k in ("host", "port", "database")})
        return {
            "system": self.name,
            "deployment": "self-hosted-graphiti-inprocess",
            "product_note": "Graphiti (getzep/graphiti) is Zep's open-source engine; Zep Cloud is not benchmarked (model calls cannot be proxied)",
            "system_version": _dist_version("graphiti-core"),
            "system_commit": s.system_commit,
            "adapter_version": ADAPTER_VERSION,
            "adapter_source_sha256": _source_hash(),
            "configuration_id": self.configuration_id,
            "memory_model": {k: v for k, v in s.llm.items() if k != "api_key"},
            "embedding_model": {k: v for k, v in s.embedder.items() if k != "api_key"},
            "reranker": dict(s.reranker),
            "graph_store": store,
            "write": {"episode_granularity": s.episode_granularity, "source": "message", "source_description": s.source_description, "date_conveyance": "reference_time from write metadata"},
            "retrieval": {"recipe": s.recipe, "top_k": s.top_k, "fact_format": s.fact_format},
            "settlement": {"timeout_s": self.settlement_config.timeout_s, "poll_interval_s": self.settlement_config.poll_interval_s, "canary_cleanup": self.settlement_config.canary_cleanup, "canary_mode": s.canary_mode},
            "feature_flags": {"SEMAPHORE_LIMIT": os.environ.get("SEMAPHORE_LIMIT"), "USE_PARALLEL_RUNTIME": os.environ.get("USE_PARALLEL_RUNTIME"), "llm_client": s.llm.get("client", "openai_responses"), "telemetry": os.environ.get("GRAPHITI_TELEMETRY_ENABLED", "true")},
            "prompts": {"graphiti_prompts_sha256": _graphiti_prompt_hash()},
            "proxy": {"base_url": self.proxy.root, "client_ids": {op: self._client_id(op) for op in OPERATIONS}},
            "seed": self.seed,
            "run_id": self.run_id,
            "python": platform.python_version(),
        }


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in ("base.py", "zep.py"):
        h.update((here / f).read_bytes())
    return h.hexdigest()


def _graphiti_prompt_hash() -> str | None:
    try:
        import graphiti_core.prompts as p

        h = hashlib.sha256()
        for f in sorted(Path(p.__file__).parent.glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _neo4j_server_version(gs: dict[str, Any]) -> str | None:
    try:
        from neo4j import GraphDatabase

        with GraphDatabase.driver(gs["uri"], auth=(gs.get("user"), gs.get("password"))) as d:
            rec = d.execute_query("CALL dbms.components() YIELD name, versions RETURN name, versions[0] AS v").records
            return ", ".join(f"{r['name']} {r['v']}" for r in rec) or None
    except Exception:  # noqa: BLE001
        return None
