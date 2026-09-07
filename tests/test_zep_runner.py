"""The LongMemEval runner drives the Zep/Graphiti adapter unchanged: dates via reference_time, no leaked system episodes."""
from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from adapters.registry import load_config
from bench.longmemeval import make_synthetic_dataset
from bench.runner import RunConfig, Runner
from proxy.fake_upstream import EMBED_DIM
from proxy.logging import read_events

ROOT = Path(__file__).resolve().parents[1]


async def test_zep_vertical_slice(proxy_server, proxy_log, tmp_path):
    ds = make_synthetic_dataset(tmp_path / "syn.json", n_instances=4, seed=11)
    system_cfg = load_config(ROOT / "configs/zep.yaml")
    bcfg = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s.yaml").read_text())
    bcfg["reader"]["model"] = "fake-model"
    bcfg["judge"]["model"] = "fake-model"
    overrides = {"graphiti": {"graph_store": {"provider": "kuzu", "path": str(tmp_path / "g.kuzu")}, "llm": {"model": "fake-model", "small_model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dim": EMBED_DIM}, "reranker": {"model": "fake-model"}}}
    inst = next(i for i in ds.instances if not i.is_abstention)
    cfg = RunConfig(system="zep", benchmark="longmemeval_s", seed=42, config_path="configs/zep.yaml", system_config=system_cfg, benchmark_config=bcfg, proxy_url=proxy_server.url, out_dir=tmp_path / "runs", question_ids=[inst.question_id], overrides=overrides)
    runner = Runner(cfg, ds)
    summary = await runner.run()
    assert summary["instances"] == 1 and summary["errors"] == 0
    events = list(read_events(runner.out_dir / "run.jsonl"))
    kinds = [e["event_type"] for e in events]
    for k in ("WRITE_ACK", "CONTEXT", "ANSWER_END", "JUDGE_END", "INSTANCE_END"):
        assert k in kinds, k
    # accuracy runs plant no canaries (settlement.policy: none); visibility experiments call wait_settled explicitly
    assert not {"CANARY_SUBMIT", "CANARY_ACK", "SETTLEMENT_POLL", "WRITE_SETTLED"} & set(kinds)
    start = events[0]
    assert start["adapter_fingerprint"]["system"] == "zep" and start["adapter_fingerprint"]["deployment"] == "self-hosted-graphiti-inprocess"
    acks = [e for e in events if e["event_type"] == "WRITE_ACK"]
    assert len(acks) == len(inst.sessions)
    # every session's date reached Graphiti as reference_time; the date system message was not an episode
    for ack, sess in zip(acks, inst.sessions):
        assert ack["memories_added"] >= 1
    ctx = next(e for e in events if e["event_type"] == "CONTEXT")
    assert inst.answer in ctx["context"] and "(valid:" in ctx["context"]
    assert summary["correct"] == 1
    await asyncio.sleep(0.3)
    calls = [e for e in read_events(proxy_log) if e["event_type"] == "MODEL_CALL"]
    assert {c["operation"] for c in calls} >= {"write", "read", "answer", "judge"} and "settle" not in {c["operation"] for c in calls}
    assert all(c["system"] == "zep" and c["session_id"] == inst.question_id for c in calls)
    assert any(c["endpoint"] == "responses" for c in calls)
