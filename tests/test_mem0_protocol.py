"""Milestone 4 Arm B: Mem0's published protocol runs through the same instrumented path."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import yaml

from adapters.registry import load_config
from bench.longmemeval import Session, Turn, make_synthetic_dataset
from bench.prompts import MEM0_ANSWER_PROMPT, MEM0_JUDGE_PROMPT, mem0_format_memories, mem0_human_date, mem0_judge_label, mem0_strip_answer, prompt_hashes, sha
from bench.runner import IngestConfig, RunConfig, Runner, split_writes
from proxy.fake_upstream import EMBED_DIM
from proxy.logging import read_events

ROOT = Path(__file__).resolve().parents[1]


def test_mem0_ports_match_source_behaviour():
    assert mem0_human_date("2023/05/01 (Mon) 21:05") == "Monday, May 01, 2023"
    assert mem0_human_date("garbage") == "garbage"
    assert mem0_strip_answer("<mem_thinking>x</mem_thinking>\nfoo ANSWER: teal") == "teal"
    assert mem0_strip_answer("[mem_thinking]x[/mem_thinking] plain") == "plain"
    assert mem0_judge_label("<judge_thinking>maybe no</judge_thinking>\nyes") is True
    assert mem0_judge_label("<judge_thinking>x</judge_thinking>\nNo") is False
    assert mem0_judge_label("Verdict: yes indeed") is True  # last yes/no token
    assert mem0_judge_label("") is False
    block = mem0_format_memories([("a", "2023-05-10T10:00:00"), ("b", "2023-05-10T11:00:00"), ("c", "2023-05-11T10:00:00"), ("d", None)])
    assert block.startswith("--- Wednesday, May 10, 2023 ---\n- a\n- b\n\n--- Thursday, May 11, 2023 ---\n- c\n- d")
    assert mem0_format_memories([]) == "(No relevant memories found)"
    h = prompt_hashes()
    assert h["reader_mem0_v3"] == sha(MEM0_ANSWER_PROMPT) and h["judge_mem0_unified"] == sha(MEM0_JUDGE_PROMPT)
    assert "chandelier counts as jewelry" in MEM0_ANSWER_PROMPT and "lean toward \"yes\"" in MEM0_JUDGE_PROMPT


def test_pair_granularity_matches_pair_turns():
    sess = Session(session_id="s", date="2023/05/01 (Mon) 21:05", index=0, turns=[Turn("user", "u1", True), Turn("assistant", "a1"), Turn("user", ""), Turn("assistant", "a2"), Turn("user", "u3")])
    writes = split_writes(sess, IngestConfig(granularity="pair", date_mode="none", skip_empty_pairs=True))
    assert [[m["content"] for m in w] for w in writes] == [["u1", "a1"], ["u3"]]
    assert all("has_answer" not in m for w in writes for m in w)
    writes_keep = split_writes(sess, IngestConfig(granularity="pair", date_mode="none", skip_empty_pairs=False))
    assert len(writes_keep) == 3


def test_arm_b_configs_are_consistent():
    sys_cfg = load_config(ROOT / "configs/mem0-published-protocol.yaml")
    assert sys_cfg["retrieval"]["top_k"] == 200 and sys_cfg["system"] == "mem0"
    b = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s_mem0protocol.yaml").read_text())
    assert b["protocol"] == "mem0-memory-benchmarks-oss" and b["ingest"]["granularity"] == "pair" and b["ingest"]["date_mode"] == "none"
    assert b["reader"]["prompt"] == "mem0_v3" and b["judge"]["style"] == "mem0_unified"
    assert b["published_reference"]["mem0"]["value"] == 0.944 and b["published_reference"]["mem0"]["correct"] == 472
    a = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s.yaml").read_text())
    assert a["published_reference"]["mem0"]["protocol_matches_this_config"] is False
    assert abs(sum(a["published_reference"]["mem0"]["per_type"].values()) / 6 - 0.9535) < 0.01


async def test_arm_b_end_to_end_and_report(proxy_server, proxy_log, tmp_path):
    ds = make_synthetic_dataset(tmp_path / "syn.json", n_instances=5, seed=9)
    sys_cfg = load_config(ROOT / "configs/mem0-published-protocol.yaml")
    bcfg = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s_mem0protocol.yaml").read_text())
    bcfg["reader"]["model"] = "fake-model"
    bcfg["judge"]["model"] = "fake-model"
    overrides = {"mem0": {"llm": {"model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dims": None}, "vector_store": {"mode": "embedded", "path": str(tmp_path / "q"), "embedding_model_dims": EMBED_DIM}, "history_db_path": str(tmp_path / "h.db")}}
    cfg = RunConfig(system="mem0", benchmark="longmemeval_s", seed=42, config_path="configs/mem0-published-protocol.yaml", system_config=sys_cfg, benchmark_config=bcfg, proxy_url=proxy_server.url, out_dir=tmp_path / "runs", limit=3, overrides=overrides)
    runner = Runner(cfg, ds)
    summary = await runner.run()
    assert summary["errors"] == 0 and summary["instances"] == 3
    events = list(read_events(runner.out_dir / "run.jsonl"))
    start = events[0]
    assert start["protocol"] == "mem0-memory-benchmarks-oss" and start["ingest"]["granularity"] == "pair" and start["ingest"]["date_mode"] == "none"
    assert start["reader"]["prompt"] == "mem0_v3" and start["reader"]["prompt_template_sha256"] == sha(MEM0_ANSWER_PROMPT)
    assert start["judge"]["style"] == "mem0_unified" and start["adapter_fingerprint"]["retrieval"]["top_k"] == 200
    # pair ingestion: writes = number of non-empty pairs, no system messages
    inst0 = next(i for i in ds.instances if i.question_id == start["selection"]["ordered_question_ids"][0])
    n_pairs = sum((len(s.turns) + 1) // 2 for s in inst0.sessions)
    writes0 = [e for e in events if e["event_type"] == "WRITE_SUBMIT" and e["question_id"] == inst0.question_id]
    assert len(writes0) == n_pairs and all(e["messages"] <= 2 for e in writes0)
    ans = [e for e in events if e["event_type"] == "ANSWER_END"]
    assert all(e["answer_raw"] and "<mem_thinking>" in e["answer_raw"] and "<mem_thinking>" not in e["answer"] for e in ans)
    jd = [e for e in events if e["event_type"] == "JUDGE_END"]
    assert all("<judge_thinking>" in e["verdict_raw"] for e in jd) and all(isinstance(e["correct"], bool) for e in jd)
    assert summary["correct"] >= 1  # the fake reader/judge recover at least one needle under Arm B too

    await asyncio.sleep(0.3)
    out_dir = tmp_path / "summaries"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_mem0.py"), "--run", str(runner.out_dir), "--proxy-log", str(proxy_log), "--out-dir", str(out_dir), "--stamp", "test"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rep = json.loads((out_dir / "mem0-reproduction-test.json").read_text())
    assert len(rep["reports"]) == 1
    rp = rep["reports"][0]
    assert rp["protocol"] == "mem0-memory-benchmarks-oss" and rp["verdict"]["status"] == "not_applicable_synthetic_dataset"
    items = {row["item"]: row for row in rp["checklist"]}
    assert items["reader_prompt"]["match"] is True and items["judge_prompt"]["match"] is True and items["retrieval_top_k"]["match"] is True and items["ingest_granularity"]["match"] is True
    assert items["deployment"]["match"] is False and items["overrides_present"]["match"] is False and items["dataset"]["match"] is None
    assert items["bm25_hybrid"]["match"] is False  # fastembed absent in this environment; recorded, not hidden
    assert rp["summary"]["cost"]["available"] and rp["summary"]["cost"]["unattributed"] == 0
    md = (out_dir / "mem0-reproduction-test.md").read_text()
    assert "Discrepancy checklist" in md and "not_applicable_synthetic_dataset" in md


def test_verdict_rule():
    sys.path.insert(0, str(ROOT / "scripts"))
    from reproduce_mem0 import verdict_for

    ref = {"value": 0.944, "tolerance_abs": 0.03, "protocol_matches_this_config": True}
    from analysis.statistics import wilson_interval

    def acc(correct, n):
        return {"accuracy_judged": wilson_interval(correct, n).as_dict()}

    assert verdict_for(acc(460, 500), ref)["status"] == "reproduced"       # 92.0: within 3 pts
    assert verdict_for(acc(455, 500), ref)["status"] == "not_reproduced"   # 91.0: outside tolerance and CI
    assert verdict_for(acc(27, 30), ref)["status"] == "reproduced"         # 90.0 but wide CI contains 0.944
    assert verdict_for(acc(0, 0), ref)["status"] == "no_reference_or_no_data"
    assert verdict_for(acc(10, 10), {})["status"] == "no_reference_or_no_data"
