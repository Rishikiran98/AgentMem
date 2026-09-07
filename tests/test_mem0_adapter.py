"""Milestone 2 validation: reset, write, read and settlement work reliably for repeated sessions,
and every model call is routed through the proxy with correct attribution."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from adapters.base import AdapterError, SettlementConfig, SettlementError
from adapters.mem0 import Mem0Adapter, Mem0Settings
from adapters.registry import build_adapter, load_config
from proxy.logging import read_events

ROOT = Path(__file__).resolve().parents[1]


def mem0_settings(tmp_path: Path) -> Mem0Settings:
    return Mem0Settings(
        llm={"provider": "openai", "model": "fake-model", "temperature": 0.1, "max_tokens": 2000, "top_p": 0.1},
        embedder={"provider": "openai", "model": "fake-embed"},  # no embedding_dims: fake upstream returns 8-d vectors
        vector_store={"provider": "qdrant", "mode": "embedded", "path": str(tmp_path / "qdrant"), "collection_name": "memharness_test", "embedding_model_dims": 8, "on_disk": True},
        history_db_path=str(tmp_path / "history.db"),
        infer=True,
        top_k=20,
        threshold=0.1,
    )


@pytest.fixture
async def adapter(proxy_server, tmp_path):
    events: list[dict] = []
    a = Mem0Adapter(mem0_settings(tmp_path), proxy_base_url=proxy_server.url, configuration_id="cfg-test", seed=7, run_id="run-test", settlement=SettlementConfig(timeout_s=10, poll_interval_s=0.05), event_sink=events.append)
    a.events = events  # type: ignore[attr-defined]
    await a.start()
    yield a
    await a.close()


def model_calls(log: Path) -> list[dict]:
    return [e for e in read_events(log) if e["event_type"] == "MODEL_CALL"]


async def test_reset_write_read_settle_repeated_sessions(adapter, proxy_log):
    """The Milestone 2 success criterion, three sessions in a row, with cross-session isolation."""
    for i in range(3):
        sid = f"session-{i}"
        await adapter.reset(sid)
        assert await adapter.read(sid, "anything") == ""

        fact_a = f"My favourite colour in round {i} is teal"
        fact_b = f"I adopted a dog named Biscuit{i} last spring"
        wr = await adapter.write(sid, [{"role": "user", "content": fact_a}, {"role": "assistant", "content": "Noted"}, {"role": "user", "content": fact_b}])
        assert wr.session_id == sid and wr.ack_latency_ms > 0 and wr.messages == 3
        assert wr.memories_added == 3 and len(wr.memory_ids) == 3  # fake extractor keeps every message verbatim

        ctx = await adapter.read(sid, fact_a)
        assert fact_a in ctx
        lines = ctx.split("\n")
        assert lines[0] == fact_a  # identical text -> identical fake embedding -> top hit

        st = await adapter.wait_settled(sid)
        assert st.settled and not st.timed_out and st.polls >= 1
        c = st.canary
        assert c.stored and c.memory_ids and c.first_retrievable_at is not None
        assert c.submitted_perf < c.acknowledged_perf <= c.first_retrievable_perf
        assert c.cleanup == "deleted"
        assert c.token not in await adapter.read(sid, c.text)  # cleaned up, no contamination

        # Earlier sessions are untouched and isolated.
        for j in range(i):
            other = await adapter.read(f"session-{j}", f"My favourite colour in round {j} is teal")
            assert f"round {j}" in other and f"round {i}" not in other

    # Reset removes exactly the target session.
    await adapter.reset("session-1")
    assert await adapter.read("session-1", "colour") == ""
    assert "round 0" in await adapter.read("session-0", "My favourite colour in round 0 is teal")
    assert "round 2" in await adapter.read("session-2", "My favourite colour in round 2 is teal")


async def test_reset_is_idempotent_and_verified(adapter):
    await adapter.reset("fresh")
    await adapter.reset("fresh")
    await adapter.write("fresh", [{"role": "user", "content": "I live in Lisbon"}])
    assert "Lisbon" in await adapter.read("fresh", "I live in Lisbon")
    await adapter.reset("fresh")
    assert await adapter.read("fresh", "I live in Lisbon") == ""
    kinds = [e["event_type"] for e in adapter.events]
    assert kinds.count("RESET_START") == 3 and kinds.count("RESET_END") == 3


async def test_every_model_call_goes_through_proxy_with_attribution(adapter, proxy_log):
    sid = "attrib"
    await adapter.reset(sid)
    await adapter.write(sid, [{"role": "user", "content": "I play the cello on Tuesdays"}])
    await adapter.read(sid, "What instrument do I play?")
    await adapter.wait_settled(sid)
    await asyncio.sleep(0.2)  # proxy logs after the response is sent

    calls = model_calls(proxy_log)
    assert calls, "no model calls reached the proxy"
    for c in calls:
        assert c["system"] == "mem0" and c["configuration"] == "cfg-test" and c["seed"] == 7 and c["run_id"] == "run-test", c
        assert c["session_id"] == sid and c["attribution_sources"]["session_id"] == "scope", c
        assert c["operation"] in ("write", "read", "settle") and c["attribution_sources"]["operation"] == "api_key_token", c
        assert c["client_id"] == f"mem0-{c['operation']}"
        assert c["unattributed"] is False and c["status"] == "success"
        assert c["usage_source"] == "upstream" and c["prompt_tokens"] > 0
    ops = {op: [c for c in calls if c["operation"] == op] for op in ("write", "read", "settle")}
    # write: at least one chat (extraction) + embeddings; read: embeddings only; settle: embeddings only (infer=False)
    assert any(c["endpoint"] == "chat" for c in ops["write"]) and any(c["endpoint"] == "embeddings" for c in ops["write"])
    assert ops["read"] and all(c["endpoint"] == "embeddings" for c in ops["read"])
    assert ops["settle"] and all(c["endpoint"] == "embeddings" for c in ops["settle"])
    # Scope events bracket the calls in the same log.
    kinds = [e["event_type"] for e in read_events(proxy_log)]
    assert "SCOPE_ENTER" in kinds and "SCOPE_EXIT" in kinds


async def test_adapter_events_are_complete_and_ordered(adapter):
    sid = "events"
    await adapter.reset(sid)
    await adapter.write(sid, [{"role": "user", "content": "I collect vintage maps"}])
    await adapter.read(sid, "hobby")
    st = await adapter.wait_settled(sid)
    kinds = [e["event_type"] for e in adapter.events]
    expected_order = ["RESET_START", "RESET_END", "WRITE_SUBMIT", "WRITE_ACK", "READ_START", "READ_END", "CANARY_SUBMIT", "CANARY_ACK", "READ_START", "READ_END", "SETTLEMENT_POLL", "WRITE_SETTLED"]
    idx = 0
    for k in kinds:
        if idx < len(expected_order) and k == expected_order[idx]:
            idx += 1
    assert idx == len(expected_order), kinds
    settled = next(e for e in adapter.events if e["event_type"] == "WRITE_SETTLED")
    for key in ("write_ack_latency_ms", "ingestion_to_retrievability_lag_ms", "submission_to_retrievability_ms", "polls", "submitted_at", "acknowledged_at", "first_retrievable_at"):
        assert settled[key] is not None, key
    assert settled["ingestion_to_retrievability_lag_ms"] >= 0
    assert settled["submission_to_retrievability_ms"] >= settled["ingestion_to_retrievability_lag_ms"]
    assert settled["polls"] == st.polls
    json.dumps(adapter.events)  # everything must be serialisable for the runner


async def test_is_settled_requires_canary(adapter):
    await adapter.reset("nocanary")
    with pytest.raises(SettlementError):
        await adapter.is_settled("nocanary")


async def test_settlement_timeout_is_reported_not_hidden(adapter, monkeypatch):
    sid = "timeout"
    await adapter.reset(sid)
    await adapter.write(sid, [{"role": "user", "content": "I like tea"}])
    rec = await adapter.plant_canary(sid)
    # Simulate a system whose write path acks but whose read path never surfaces the fact.
    monkeypatch.setattr(adapter, "canary_text", staticmethod(lambda token: "unrelated"))
    rec.text = "completely unrelated query text"
    st = await adapter.wait_settled(sid, timeout_s=0.3, poll_interval_s=0.05)
    assert st.timed_out and not st.settled and st.polls >= 2
    assert st.canary.first_retrievable_at is None
    assert any(e["event_type"] == "SETTLEMENT_TIMEOUT" for e in adapter.events)
    assert st.canary.cleanup == "deleted"


async def test_write_failure_is_raised_and_logged(adapter, proxy_log):
    sid = "fail"
    await adapter.reset(sid)
    adapter.settings.llm["model"] = "fail-500"  # affects only newly built instances
    await adapter.close()
    await adapter.start()
    with pytest.raises(AdapterError):
        await adapter.write(sid, [{"role": "user", "content": "this will fail"}])
    assert any(e["event_type"] == "ERROR" and e["phase"] == "write" for e in adapter.events)
    await asyncio.sleep(0.2)
    errs = [c for c in model_calls(proxy_log) if c["status"] == "error"]
    assert errs and errs[-1]["operation"] == "write" and errs[-1]["http_status"] == 500


async def test_reset_all_wipes_everything_and_rebuilds(adapter):
    await adapter.write("a", [{"role": "user", "content": "alpha fact"}])
    await adapter.write("b", [{"role": "user", "content": "beta fact"}])
    await adapter.reset_all()
    assert await adapter.read("a", "alpha fact") == "" and await adapter.read("b", "beta fact") == ""
    await adapter.write("a", [{"role": "user", "content": "alpha again"}])
    assert "alpha again" in await adapter.read("a", "alpha again")


async def test_config_fingerprint_is_complete(adapter):
    fp = adapter.config_fingerprint()
    for key in ("system", "system_version", "system_commit", "adapter_version", "adapter_source_sha256", "configuration_id", "memory_model", "embedding_model", "vector_store", "retrieval", "write", "settlement", "feature_flags", "prompts", "proxy", "seed", "run_id"):
        assert key in fp, key
    assert fp["system_version"] is not None
    assert fp["memory_model"]["model"] == "fake-model" and "api_key" not in fp["memory_model"]
    assert fp["vector_store"]["mode"] == "embedded" and fp["vector_store"]["client_version"]
    assert fp["prompts"]["additive_extraction_prompt_sha256"]
    assert set(fp["feature_flags"]) >= {"spacy_en_core_web_sm", "bm25_hybrid_search_enabled", "telemetry"}
    assert fp["feature_flags"]["telemetry"] == "false"
    json.dumps(fp)


async def test_committed_config_loads_and_builds(proxy_server, tmp_path):
    cfg = load_config(ROOT / "configs" / "mem0.yaml")
    assert cfg["system"] == "mem0" and len(cfg["_configuration_id"]) == 16
    overrides = {"mem0": {"llm": {"model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dims": None}, "vector_store": {"mode": "embedded", "path": str(tmp_path / "q"), "embedding_model_dims": 8}, "history_db_path": str(tmp_path / "h.db")}}
    a = build_adapter(cfg, proxy_base_url=proxy_server.url, seed=1, run_id="r", overrides=overrides)
    assert isinstance(a, Mem0Adapter) and a.settlement_config.timeout_s == 30
    await a.start()
    try:
        await a.write("s", [{"role": "user", "content": "config test"}])
        assert "config test" in await a.read("s", "config test")
        assert a.config_fingerprint()["configuration_id"] == cfg["_configuration_id"]
    finally:
        await a.close()
