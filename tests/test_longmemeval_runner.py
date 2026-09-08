"""Milestone 3: one complete LongMemEval instance runs end to end and every stage is on record."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from adapters.registry import load_config
from bench.longmemeval import DatasetError, load_longmemeval, make_synthetic_dataset, select_instances, selection_digest
from bench.prompts import JUDGE_DEFAULT, judge_label, judge_prompt, prompt_hashes, reader_prompt
from bench.runner import IngestConfig, RunConfig, Runner, format_session_messages, split_writes
from proxy.fake_upstream import EMBED_DIM
from proxy.logging import read_events

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    return make_synthetic_dataset(tmp_path / "syn.json", n_instances=7, seed=3)


def test_synthetic_dataset_has_official_schema(dataset, tmp_path):
    raw = json.loads((tmp_path / "syn.json").read_text())
    for k in ("question_id", "question_type", "question", "answer", "question_date", "haystack_session_ids", "haystack_dates", "haystack_sessions", "answer_session_ids"):
        assert k in raw[0]
    assert dataset.synthetic and len(dataset.instances) == 7 and dataset.sha256
    assert any(i.is_abstention for i in dataset.instances)
    assert any(t.has_answer for i in dataset.instances for s in i.sessions for t in s.turns)


def test_loader_rejects_bad_schema(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"question_id": "x"}]))
    with pytest.raises(DatasetError):
        load_longmemeval(p)


def test_seeded_selection_is_reproducible_and_limit_is_prefix(dataset):
    a = [i.question_id for i in select_instances(dataset, seed=42)]
    b = [i.question_id for i in select_instances(dataset, seed=42)]
    c = [i.question_id for i in select_instances(dataset, seed=43)]
    assert a == b and a != c and sorted(a) == sorted(c)
    assert [i.question_id for i in select_instances(dataset, seed=42, limit=3)] == a[:3]
    assert selection_digest(select_instances(dataset, seed=42)) == selection_digest(select_instances(dataset, seed=42))
    sub = select_instances(dataset, seed=1, question_ids=["syn_1", "syn_0"])
    assert {i.question_id for i in sub} == {"syn_0", "syn_1"}
    with pytest.raises(DatasetError):
        select_instances(dataset, seed=1, question_ids=["nope"])


def test_ingest_formatting_never_leaks_labels(dataset):
    inst = next(i for i in dataset.instances if not i.is_abstention)
    sess = next(s for s in inst.sessions if any(t.has_answer for t in s.turns))
    msgs = format_session_messages(sess, IngestConfig())
    assert msgs[0]["role"] == "system" and sess.date in msgs[0]["content"]
    assert all("has_answer" not in m for m in msgs)
    assert [m["content"] for m in msgs[1:]] == [t.content for t in sess.turns]
    prefixed = format_session_messages(sess, IngestConfig(date_mode="prefix_first_user"))
    assert prefixed[0]["role"] == "user" and prefixed[0]["content"].startswith(f"[{sess.date}] ")
    none = format_session_messages(sess, IngestConfig(date_mode="none"))
    assert all(m["role"] != "system" for m in none)
    per_turn = split_writes(sess, IngestConfig(granularity="turn"))
    assert len(per_turn) == len(sess.turns) and all(w[0]["role"] == "system" and len(w) == 2 for w in per_turn)


def test_official_prompts_are_verbatim_ports():
    p = judge_prompt("multi-session", "Q?", "A", "R", abstention=False)
    assert p == JUDGE_DEFAULT.format("Q?", "A", "R")
    assert p.endswith("Is the model response correct? Answer yes or no only.")
    assert judge_prompt("x", "Q?", "E", "R", abstention=True).startswith("I will give you an unanswerable question")
    assert "off-by-one" in judge_prompt("temporal-reasoning", "Q", "A", "R", abstention=False)
    assert "Rubric" in judge_prompt("single-session-preference", "Q", "A", "R", abstention=False)
    with pytest.raises(NotImplementedError):
        judge_prompt("unknown-type", "Q", "A", "R", abstention=False)
    assert judge_label("Yes.") and judge_label(" yes") and not judge_label("No") and judge_label("The answer is yes")  # official rule: substring
    r = reader_prompt("longmemeval_facts", "fact one\nfact two", "2023/06/01 (Thu) 09:00", "What?")
    assert r.startswith("I will give you several facts extracted from history chats") and r.endswith("Question: What?\nAnswer:")
    assert "Current Date: 2023/06/01 (Thu) 09:00" in r
    h = prompt_hashes()
    assert len(h) >= 7 and all(len(v) == 64 for v in h.values()) and {"judge_default", "judge_abstention", "reader_longmemeval_facts"} <= set(h)


async def test_vertical_slice_one_instance(proxy_server, proxy_log, dataset, tmp_path):
    """One complete LongMemEval instance: ingestion -> settlement -> retrieval -> reader -> judge."""
    system_cfg = load_config(ROOT / "configs/mem0.yaml")
    bcfg = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s.yaml").read_text())
    bcfg["reader"]["model"] = "fake-model"
    bcfg["judge"]["model"] = "fake-model"
    overrides = {"mem0": {"llm": {"model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dims": None}, "vector_store": {"mode": "embedded", "path": str(tmp_path / "q"), "embedding_model_dims": EMBED_DIM}, "history_db_path": str(tmp_path / "h.db")}}
    inst = next(i for i in dataset.instances if not i.is_abstention)
    cfg = RunConfig(system="mem0", benchmark="longmemeval_s", seed=42, config_path="configs/mem0.yaml", system_config=system_cfg, benchmark_config=bcfg, proxy_url=proxy_server.url, out_dir=tmp_path / "runs", question_ids=[inst.question_id], overrides=overrides)
    runner = Runner(cfg, dataset)
    summary = await runner.run()
    assert summary["instances"] == 1 and summary["errors"] == 0
    assert summary["correct"] == 1, "fake reader + judge should recover the needle when retrieval works"

    events = list(read_events(runner.out_dir / "run.jsonl"))
    kinds = [e["event_type"] for e in events]
    assert kinds[0] == "RUN_START" and kinds[-1] == "RUN_END"
    for k in ("INSTANCE_START", "RESET_START", "RESET_END", "WRITE_SUBMIT", "WRITE_ACK", "INGEST_END", "READ_START", "READ_END", "CONTEXT", "ANSWER_START", "ANSWER_END", "JUDGE_START", "JUDGE_END", "INSTANCE_END"):
        assert k in kinds, k
    # Accuracy reproduction deliberately uses the non-contaminating
    # synchronization policy: no artificial canary enters the memory store.
    for k in ("CANARY_SUBMIT", "CANARY_ACK", "SETTLEMENT_POLL", "WRITE_SETTLED"):
        assert k not in kinds, k
    # order of the pipeline stages
    first = {k: kinds.index(k) for k in ("INSTANCE_START", "INGEST_END", "CONTEXT", "ANSWER_END", "JUDGE_END", "INSTANCE_END")}
    assert list(first.values()) == sorted(first.values())
    assert kinds.count("WRITE_SUBMIT") == len(inst.sessions) == kinds.count("WRITE_ACK")
    # every event is attributable
    for e in events[1:-1]:
        assert e["run_id"] == cfg.run_id and e["seed"] == 42 and e["system"] == "mem0"
        if e["event_type"] not in ("RUN_START", "RUN_END", "ADAPTER_START"):
            assert e.get("question_id") == inst.question_id, e["event_type"]
    run_start = events[0]
    for k in ("configuration_id", "dataset", "selection", "adapter_fingerprint", "reader", "judge", "prompt_hashes", "proxy", "host", "ingest", "settlement_policy"):
        assert k in run_start, k
    assert run_start["dataset"]["sha256"] == dataset.sha256 and run_start["dataset"]["synthetic"] is True
    assert run_start["overrides_present"] is True
    assert run_start["host"]["git"]["commit"] and run_start["host"]["python"]
    assert run_start["proxy"]["instance_id"]
    ctx = next(e for e in events if e["event_type"] == "CONTEXT")
    assert inst.answer in ctx["context"]
    ans = next(e for e in events if e["event_type"] == "ANSWER_END")
    assert ans["proxy_request_id"] and inst.answer in ans["answer"]
    jd = next(e for e in events if e["event_type"] == "JUDGE_END")
    assert jd["correct"] is True and jd["verdict_raw"] == "yes" and jd["proxy_request_id"]
    end = next(e for e in events if e["event_type"] == "INSTANCE_END")
    assert end["correct"] is True and end["settled"] is None and all(end[k] is not None for k in ("ingest_ms", "settle_ms", "read_ms", "answer_ms", "judge_ms"))
    assert (runner.out_dir / "manifest.json").exists()

    # Proxy side: reader and judge calls are tagged answer/judge under system=mem0 with the question as session.
    import asyncio

    await asyncio.sleep(0.3)
    calls = [e for e in read_events(proxy_log) if e["event_type"] == "MODEL_CALL"]
    ops = {c["operation"] for c in calls}
    assert {"write", "read", "answer", "judge"} <= ops
    assert "settle" not in ops
    for c in calls:
        assert c["system"] == "mem0" and c["run_id"] == cfg.run_id and c["session_id"] == inst.question_id and c["configuration"] == system_cfg["_configuration_id"], c
    assert [c["proxy_request_id"] if False else c["request_id"] for c in calls if c["operation"] == "answer"] == [ans["proxy_request_id"]]
    assert [c["request_id"] for c in calls if c["operation"] == "judge"] == [jd["proxy_request_id"]]
    # Seed/order reproducibility is on record.
    assert run_start["selection"]["ordered_question_ids"] == [inst.question_id]


async def test_multi_instance_run_records_errors_and_continues(proxy_server, dataset, tmp_path):
    system_cfg = load_config(ROOT / "configs/mem0.yaml")
    bcfg = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s.yaml").read_text())
    bcfg["reader"]["model"] = "fake-model"
    bcfg["judge"]["model"] = "fake-model"
    bcfg["settlement"]["policy"] = "none"
    overrides = {"mem0": {"llm": {"model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dims": None}, "vector_store": {"mode": "embedded", "path": str(tmp_path / "q"), "embedding_model_dims": EMBED_DIM}, "history_db_path": str(tmp_path / "h.db")}}
    cfg = RunConfig(system="mem0", benchmark="longmemeval_s", seed=7, config_path="configs/mem0.yaml", system_config=system_cfg, benchmark_config=bcfg, proxy_url=proxy_server.url, out_dir=tmp_path / "runs", limit=3, overrides=overrides)
    runner = Runner(cfg, dataset)
    # Break the reader for the second instance only.
    original = runner.reader_cfg.model
    calls = {"n": 0}

    async def flaky_answer(**kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated reader outage")
        return await real_answer(**kw)

    instances = select_instances(dataset, seed=7, limit=3)
    summary = None
    from bench.reader import Reader

    real_answer = None

    async def run():
        nonlocal real_answer
        insts = await runner.setup()
        real_answer = runner.reader.answer
        runner.reader.answer = flaky_answer  # type: ignore[method-assign]
        try:
            for pos, inst in enumerate(insts):
                await runner.run_instance(inst, pos)
        finally:
            runner._ctx = {}
            runner.emit("RUN_END", **runner.tally)
            await runner.close()
        return dict(runner.tally)

    summary = await run()
    assert summary["instances"] == 3 and summary["errors"] == 1
    events = list(read_events(runner.out_dir / "run.jsonl"))
    ends = [e for e in events if e["event_type"] == "INSTANCE_END"]
    assert [e["question_id"] for e in ends] == [i.question_id for i in instances]
    assert [e["error"] for e in ends].count("RuntimeError") == 1
    assert any(e["event_type"] == "ERROR" and "simulated reader outage" in e["error_message"] for e in events)
    assert original == runner.reader_cfg.model
