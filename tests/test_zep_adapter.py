"""Milestone 5 validation: Zep/Graphiti adapter reset, write, read and settlement over repeated sessions,
with every model call (Responses API, embeddings, reranker) routed through the proxy with attribution."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("graphiti_core", reason="zep extra not installed (pip install -e '.[zep]')")
pytest.importorskip("kuzu", reason="zep extra not installed (embedded Kuzu driver for tests)")

from adapters.base import AdapterError, SettlementConfig
from adapters.registry import build_adapter, load_config
from adapters.zep import ZepAdapter, ZepSettings, parse_reference_time
from proxy.fake_upstream import EMBED_DIM
from proxy.logging import read_events

ROOT = Path(__file__).resolve().parents[1]


def zep_settings(tmp_path: Path, **over) -> ZepSettings:
    kw = dict(
        graph_store={"provider": "kuzu", "path": str(tmp_path / "graph.kuzu")},
        llm={"client": "openai_responses", "model": "fake-model", "small_model": "fake-model", "temperature": 0.0, "max_tokens": 2000},
        embedder={"model": "fake-embed", "embedding_dim": EMBED_DIM},
        reranker={"provider": "openai", "model": "fake-model"},
        top_k=10,
    )
    kw.update(over)
    return ZepSettings(**kw)


@pytest.fixture
async def adapter(proxy_server, tmp_path):
    events: list[dict] = []
    a = ZepAdapter(zep_settings(tmp_path), proxy_base_url=proxy_server.url, configuration_id="cfg-zep", seed=3, run_id="run-zep", settlement=SettlementConfig(timeout_s=15, poll_interval_s=0.05), event_sink=events.append)
    a.events = events  # type: ignore[attr-defined]
    await a.start()
    yield a
    await a.close()


def model_calls(log: Path) -> list[dict]:
    return [e for e in read_events(log) if e["event_type"] == "MODEL_CALL"]


def test_reference_time_parsing():
    assert parse_reference_time("2023/05/20 (Sat) 02:21") == datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc)
    assert parse_reference_time("2023-05-20T02:21:00Z") == datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc)
    assert parse_reference_time("nonsense") is None and parse_reference_time(None) is None


async def test_reset_write_read_settle_repeated_sessions(adapter):
    for i in range(3):
        sid = f"zsession-{i}"
        await adapter.reset(sid)
        assert await adapter.read(sid, "anything") == ""
        fact = f"My favourite colour in round {i} is teal"
        msgs = [{"role": "system", "content": "This conversation took place on 2023/05/20 (Sat) 02:21."}, {"role": "user", "content": fact}, {"role": "assistant", "content": "Noted, thank you"}]
        wr = await adapter.write(sid, msgs, metadata={"lme_session_id": f"s{i}", "session_date": "2023/05/20 (Sat) 02:21"})
        assert wr.messages == 3 and wr.ack_latency_ms > 0
        assert wr.memories_added >= 1 and len(wr.memory_ids) == wr.memories_added
        assert len(wr.native["episodes"]) == 2, "system message must not become an episode"
        assert wr.native["reference_time"].startswith("2023-05-20T02:21:00")
        ctx = await adapter.read(sid, fact)
        assert fact in ctx and "(valid:" in ctx
        st = await adapter.wait_settled(sid)
        assert st.settled and not st.timed_out and st.canary.stored and st.canary.memory_ids
        assert st.canary.submitted_perf < st.canary.acknowledged_perf <= st.canary.first_retrievable_perf
        assert st.canary.cleanup == "deleted"
        assert st.canary.token not in await adapter.read(sid, st.canary.text)
        for j in range(i):
            other = await adapter.read(f"zsession-{j}", f"My favourite colour in round {j} is teal")
            assert f"round {j}" in other and f"round {i}" not in other
    await adapter.reset("zsession-1")
    assert await adapter.read("zsession-1", "colour") == ""
    assert "round 0" in await adapter.read("zsession-0", "My favourite colour in round 0 is teal")


async def test_every_model_call_goes_through_proxy_with_attribution(adapter, proxy_log):
    sid = "zattrib"
    await adapter.reset(sid)
    await adapter.write(sid, [{"role": "user", "content": "I play the cello on Tuesdays"}], metadata={"session_date": "2023/05/20 (Sat) 02:21"})
    await adapter.read(sid, "What instrument do I play?")
    await adapter.wait_settled(sid)
    await asyncio.sleep(0.3)
    calls = model_calls(proxy_log)
    assert calls
    for c in calls:
        assert c["system"] == "zep" and c["configuration"] == "cfg-zep" and c["seed"] == 3 and c["run_id"] == "run-zep", c
        assert c["session_id"] == sid and c["attribution_sources"]["session_id"] == "scope"
        assert c["operation"] in ("write", "read", "settle") and c["client_id"] == f"zep-{c['operation']}"
        assert c["status"] == "success" and c["usage_source"] == "upstream" and c["prompt_tokens"] > 0
    write_calls = [c for c in calls if c["operation"] == "write"]
    assert any(c["endpoint"] == "responses" for c in write_calls), "Graphiti's default client uses the Responses API"
    assert any(c["endpoint"] == "embeddings" for c in write_calls)
    resp = next(c for c in write_calls if c["endpoint"] == "responses")
    assert resp["request_params"]["response_format"]["type"] == "json_schema" and resp["request_params"]["response_format"]["name"]
    assert resp["usage_details"]["input_tokens"] == resp["prompt_tokens"] and resp["usage_details"]["output_tokens"] == resp["completion_tokens"]
    assert all(c["endpoint"] == "embeddings" for c in calls if c["operation"] == "read"), "RRF search makes no LLM calls"
    assert {c["endpoint"] for c in calls if c["operation"] == "settle"} <= {"embeddings", "responses"}


async def test_cross_encoder_recipe_calls_reranker_through_proxy(proxy_server, proxy_log, tmp_path):
    a = ZepAdapter(zep_settings(tmp_path, recipe="edge_hybrid_cross_encoder"), proxy_base_url=proxy_server.url, configuration_id="cfg", seed=1, run_id="r")
    await a.start()
    try:
        await a.write("ce", [{"role": "user", "content": "My sister Ines lives in Coimbra"}, {"role": "user", "content": "I take the 7:10 train to work"}])
        ctx = await a.read("ce", "Where does my sister live?")
        assert "Coimbra" in ctx
    finally:
        await a.close()
    await asyncio.sleep(0.3)
    reads = [c for c in model_calls(proxy_log) if c["operation"] == "read"]
    assert any(c["endpoint"] == "chat" and c["request_params"].get("max_tokens") == 1 for c in reads), "reranker True/False logprob calls must be visible as read cost"


async def test_write_failure_raised_and_logged(adapter, proxy_log):
    sid = "zfail"
    await adapter.reset(sid)
    adapter.settings.llm["model"] = "fail-500"
    adapter.settings.llm["small_model"] = "fail-500"
    await adapter.close()
    await adapter.start()
    with pytest.raises(AdapterError):
        await adapter.write(sid, [{"role": "user", "content": "this will fail"}])
    assert any(e["event_type"] == "ERROR" and e["phase"] == "write" for e in adapter.events)
    await asyncio.sleep(0.3)
    errs = [c for c in model_calls(proxy_log) if c["status"] == "error"]
    assert errs and errs[-1]["operation"] == "write" and errs[-1]["http_status"] == 500


async def test_reset_all_and_fingerprint(adapter):
    await adapter.write("a", [{"role": "user", "content": "alpha fact about Lisbon"}])
    await adapter.reset_all()
    assert await adapter.read("a", "alpha fact about Lisbon") == ""
    await adapter.write("a", [{"role": "user", "content": "alpha again about Porto"}])
    assert "Porto" in await adapter.read("a", "alpha again about Porto")
    fp = adapter.config_fingerprint()
    for k in ("system", "deployment", "system_version", "adapter_source_sha256", "memory_model", "embedding_model", "reranker", "graph_store", "write", "retrieval", "settlement", "feature_flags", "prompts", "proxy"):
        assert k in fp, k
    assert fp["deployment"] == "self-hosted-graphiti-inprocess" and fp["graph_store"]["provider"] == "kuzu" and fp["graph_store"]["deprecated_backend"] is True
    assert fp["prompts"]["graphiti_prompts_sha256"] and fp["system_version"]
    json.dumps(fp)


async def test_committed_config_loads_and_builds(proxy_server, tmp_path):
    cfg = load_config(ROOT / "configs" / "zep.yaml")
    assert cfg["system"] == "zep" and cfg["graphiti"]["llm"]["client"] == "openai_responses"
    overrides = {"graphiti": {"graph_store": {"provider": "kuzu", "path": str(tmp_path / "g.kuzu")}, "llm": {"model": "fake-model", "small_model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dim": EMBED_DIM}, "reranker": {"model": "fake-model"}}}
    a = build_adapter(cfg, proxy_base_url=proxy_server.url, seed=1, run_id="r", overrides=overrides)
    assert isinstance(a, ZepAdapter) and a.settings.fact_format == "fact_with_dates" and a.settlement_config.timeout_s == 60
    await a.start()
    try:
        await a.write("s", [{"role": "user", "content": "config test with Lisbon"}])
        assert "Lisbon" in await a.read("s", "config test with Lisbon")
        assert a.config_fingerprint()["configuration_id"] == cfg["_configuration_id"]
    finally:
        await a.close()
