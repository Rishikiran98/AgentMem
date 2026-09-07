"""Milestone 6 validation: Letta adapter reset, write, read and settlement over repeated sessions,
against the retired Letta V1 server on embedded PostgreSQL, with all model calls through the proxy."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("letta_client", reason="letta extra not installed (pip install -e '.[letta]')")

from adapters.base import AdapterError, SettlementConfig
from adapters.letta import LettaAdapter, LettaSettings
from adapters.registry import build_adapter, load_config
from proxy.fake_upstream import EMBED_DIM
from proxy.logging import read_events
from tests.letta_server import SERVER_ENV

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.usefixtures("letta_server")


def letta_settings(server_url: str, **over) -> LettaSettings:
    kw = dict(
        server_url=server_url,
        llm={"model": "fake-model", "temperature": 0.0, "max_tokens": 512, "context_window": 32000},
        embedding={"model": "fake-embed", "embedding_dim": EMBED_DIM, "chunk_size": 300},
        agent={"include_base_tools": True, "persona": "I am a helpful assistant."},
        archival_top_k=10,
        server_env=dict(SERVER_ENV),  # the letta_server fixture runs with exactly this env
        server_runtime={"event_loop": "uvloop"},
    )
    kw.update(over)
    return LettaSettings(**kw)


@pytest.fixture
async def adapter(proxy_server, letta_server):
    events: list[dict] = []
    a = LettaAdapter(letta_settings(letta_server.url), proxy_base_url=proxy_server.url, configuration_id="cfg-letta", seed=5, run_id="run-letta", settlement=SettlementConfig(timeout_s=20, poll_interval_s=0.1), event_sink=events.append)
    a.events = events  # type: ignore[attr-defined]
    await a.start()
    yield a
    await a.reset_all()
    await a.close()


def model_calls(log: Path) -> list[dict]:
    return [e for e in read_events(log) if e["event_type"] == "MODEL_CALL"]


async def test_reset_write_read_settle_repeated_sessions(adapter):
    for i in range(3):
        sid = f"lsession-{i}"
        await adapter.reset(sid)
        assert await adapter.read(sid, "anything") == "" or "[persona]" in await adapter.read(sid, "anything")
        fact = f"My favourite colour in round {i} is teal"
        wr = await adapter.write(sid, [{"role": "system", "content": "This conversation took place on 2023/05/20 (Sat) 02:21."}, {"role": "user", "content": fact}, {"role": "assistant", "content": "Noted, thank you"}])
        assert wr.messages == 3 and wr.ack_latency_ms > 0
        assert wr.memories_added >= 1, wr.native  # the (fake) agent stored the fact with a memory tool
        assert wr.native["stop_reason"] == "end_turn" and any(tc["name"] in ("memory_insert", "core_memory_append", "archival_memory_insert") for tc in wr.native["tool_calls"])
        ctx = await adapter.read(sid, fact)
        assert fact in ctx and "[human]" in ctx
        st = await adapter.wait_settled(sid)
        assert st.settled and not st.timed_out and st.canary.stored and st.canary.memory_ids
        assert st.canary.submitted_perf < st.canary.acknowledged_perf <= st.canary.first_retrievable_perf
        assert st.canary.cleanup == "deleted"
        assert st.canary.token not in await adapter.read(sid, st.canary.text)
        for j in range(i):
            other = await adapter.read(f"lsession-{j}", f"My favourite colour in round {j} is teal")
            assert f"round {j}" in other and f"round {i}" not in other
    await adapter.reset("lsession-1")
    assert "round 1" not in await adapter.read("lsession-1", "colour")
    assert "round 0" in await adapter.read("lsession-0", "My favourite colour in round 0 is teal")


async def test_every_model_call_goes_through_proxy_with_attribution(adapter, proxy_log):
    sid = "lattrib"
    await adapter.reset(sid)
    await adapter.write(sid, [{"role": "user", "content": "I play the cello on Tuesdays"}])
    await adapter.read(sid, "What instrument do I play?")
    await adapter.wait_settled(sid)
    await asyncio.sleep(0.3)
    calls = model_calls(proxy_log)
    assert calls
    for c in calls:
        assert c["system"] == "letta" and c["configuration"] == "cfg-letta" and c["seed"] == 5 and c["run_id"] == "run-letta", c
        assert c["session_id"] == sid, c
        assert c["operation"] in ("write", "read", "settle"), c
        assert c["status"] == "success" and c["usage_source"] == "upstream" and c["prompt_tokens"] > 0
    agent_calls = [c for c in calls if c["client_id"] == f"letta-agent-{sid}"]
    assert agent_calls and all(c["endpoint"] == "chat" and c["operation"] == "write" and c["attribution_sources"]["operation"] == "path_token" for c in agent_calls)
    embed_calls = [c for c in calls if c["client_id"] == "letta-embed"]
    assert embed_calls and all(c["endpoint"] == "embeddings" and c["attribution_sources"]["operation"] == "scope" for c in embed_calls)
    assert {c["operation"] for c in embed_calls} >= {"read", "settle"}


async def test_write_failure_raised_and_logged(adapter, proxy_log):
    sid = "lfail"
    adapter.settings.llm["model"] = "fail-500"
    await adapter.reset(sid)  # new agent picks up the failing model
    with pytest.raises(AdapterError):
        await adapter.write(sid, [{"role": "user", "content": "this will fail"}])
    adapter.settings.llm["model"] = "fake-model"
    assert any(e["event_type"] == "ERROR" and e["phase"] == "write" for e in adapter.events)
    await asyncio.sleep(0.3)
    errs = [c for c in model_calls(proxy_log) if c["status"] == "error"]
    assert errs and errs[-1]["operation"] == "write" and errs[-1]["http_status"] == 500 and errs[-1]["client_id"] == f"letta-agent-{sid}"


async def test_reset_all_and_fingerprint(adapter):
    await adapter.write("la", [{"role": "user", "content": "alpha fact about Lisbon"}])
    await adapter.reset_all()
    assert not adapter._agents
    await adapter.write("la", [{"role": "user", "content": "alpha again about Porto"}])
    assert "Porto" in await adapter.read("la", "alpha again about Porto")
    fp = adapter.config_fingerprint()
    for k in ("system", "deployment", "system_version", "client_version", "adapter_source_sha256", "memory_model", "embedding_model", "agent", "write", "retrieval", "settlement", "feature_flags", "prompts", "proxy"):
        assert k in fp, k
    assert fp["deployment"] == "self-hosted-letta-v1-server-archived" and fp["system_version"] == "0.16.8"
    assert fp["agent"]["probe"]["agent_type"].endswith("letta_v1_agent") and "memory_insert" in fp["agent"]["probe"]["tools"]
    assert fp["prompts"]["system_prompt_sha256"] and fp["agent"]["probe"]["system_prompt_chars"] > 500
    json.dumps(fp)


async def test_committed_config_loads_and_builds(proxy_server, letta_server):
    cfg = load_config(ROOT / "configs" / "letta.yaml")
    assert cfg["system"] == "letta" and cfg["letta"]["agent"]["agent_type"] == "letta_v1_agent"
    overrides = {"letta": {"server_url": letta_server.url, "llm": {"model": "fake-model", "context_window": 32000}, "embedding": {"model": "fake-embed", "embedding_dim": EMBED_DIM}}}
    a = build_adapter(cfg, proxy_base_url=proxy_server.url, seed=1, run_id="r", overrides=overrides)
    assert isinstance(a, LettaAdapter) and a.settings.archival_top_k == 10 and a.settlement_config.timeout_s == 60
    await a.start()
    try:
        await a.write("s", [{"role": "user", "content": "config test with Lisbon"}])
        assert "Lisbon" in await a.read("s", "config test with Lisbon")
        assert a.config_fingerprint()["configuration_id"] == cfg["_configuration_id"]
    finally:
        await a.reset_all()
        await a.close()


def test_server_env_frozen_and_consistent(letta_server):
    """The deployment env is declared once per surface and all three agree (no silent substitution);
    the launcher verified the uvloop runtime the official image has."""
    from tests.letta_server import REQUIRED_RUNTIME

    cfg = load_config(ROOT / "configs" / "letta.yaml")
    assert cfg["server_env"] == SERVER_ENV
    compose = (ROOT / "compose" / "letta" / "docker-compose.yml").read_text()
    for k, v in SERVER_ENV.items():
        assert f"- {k}={v}" in compose, k
    assert "LETTA_TRACK_PROVIDER_TRACE" not in compose and "LETTA_TRACK_PROVIDER_TRACE" not in SERVER_ENV
    assert cfg["server_runtime"]["event_loop"] == REQUIRED_RUNTIME["event_loop"] == "uvloop"
    assert letta_server.runtime["event_loop"] == "uvloop" and letta_server.runtime["uvloop"]
    s = LettaSettings.from_config(cfg)
    assert s.server_env == SERVER_ENV and s.server_runtime == cfg["server_runtime"]


async def test_consecutive_writes_have_no_server_stall(adapter):
    """Regression for the 60 s per-step stall (OpenAI client teardown deregistering a reused fd,
    killing a NullPool asyncpg connect) that the server shows on the stdlib selector loop.

    On uvloop, as in the official image, every consecutive write acks within seconds.
    """
    await adapter.reset("stall")
    latencies = []
    for i in range(6):
        wr = await adapter.write("stall", [{"role": "user", "content": f"I keep note number {i} about Lisbon."}, {"role": "assistant", "content": "Noted."}])
        latencies.append(wr.ack_latency_ms)
    assert max(latencies) < 30_000, latencies
    fp = adapter.config_fingerprint()
    assert fp["deployment_env"] == SERVER_ENV and fp["deployment_runtime"]["event_loop"] == "uvloop"
