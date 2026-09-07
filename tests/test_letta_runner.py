"""The LongMemEval runner drives the Letta adapter unchanged."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

pytest.importorskip("letta_client", reason="letta extra not installed (pip install -e '.[letta]')")

from adapters.registry import load_config
from bench.longmemeval import make_synthetic_dataset
from bench.runner import RunConfig, Runner
from proxy.fake_upstream import EMBED_DIM
from proxy.logging import read_events

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.usefixtures("letta_server")


async def test_letta_vertical_slice(proxy_server, proxy_log, letta_server, tmp_path):
    ds = make_synthetic_dataset(tmp_path / "syn.json", n_instances=4, seed=13)
    system_cfg = load_config(ROOT / "configs/letta.yaml")
    bcfg = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s.yaml").read_text())
    bcfg["reader"]["model"] = "fake-model"
    bcfg["judge"]["model"] = "fake-model"
    overrides = {"letta": {"server_url": letta_server.url, "llm": {"model": "fake-model", "context_window": 32000}, "embedding": {"model": "fake-embed", "embedding_dim": EMBED_DIM}}}
    inst = next(i for i in ds.instances if not i.is_abstention)
    cfg = RunConfig(system="letta", benchmark="longmemeval_s", seed=42, config_path="configs/letta.yaml", system_config=system_cfg, benchmark_config=bcfg, proxy_url=proxy_server.url, out_dir=tmp_path / "runs", question_ids=[inst.question_id], overrides=overrides)
    runner = Runner(cfg, ds)
    summary = await runner.run()
    assert summary["instances"] == 1 and summary["errors"] == 0
    events = list(read_events(runner.out_dir / "run.jsonl"))
    kinds = [e["event_type"] for e in events]
    for k in ("WRITE_ACK", "CONTEXT", "ANSWER_END", "JUDGE_END", "INSTANCE_END"):
        assert k in kinds, k
    # accuracy runs plant no canaries (settlement.policy: none); visibility experiments call wait_settled explicitly
    assert not {"CANARY_SUBMIT", "CANARY_ACK", "SETTLEMENT_POLL", "WRITE_SETTLED"} & set(kinds)
    assert events[0]["adapter_fingerprint"]["deployment"] == "self-hosted-letta-v1-server-archived"
    acks = [e for e in events if e["event_type"] == "WRITE_ACK"]
    assert len(acks) == len(inst.sessions) and any(a["memories_added"] >= 1 for a in acks)  # repeated filler is deduplicated by the (fake) agent
    ctx = next(e for e in events if e["event_type"] == "CONTEXT")
    assert "[human]" in ctx["context"] and inst.answer in ctx["context"]
    assert summary["correct"] == 1
    await asyncio.sleep(0.3)
    calls = [e for e in read_events(proxy_log) if e["event_type"] == "MODEL_CALL"]
    assert {c["operation"] for c in calls} >= {"write", "read", "answer", "judge"} and "settle" not in {c["operation"] for c in calls}
    assert all(c["system"] == "letta" and c["session_id"] == inst.question_id for c in calls)
