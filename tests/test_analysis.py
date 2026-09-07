"""Analysis layer: statistics are correct on known values and metrics derive from raw events only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from adapters.registry import load_config
from analysis.ingest import load_run
from analysis.metrics import accuracy, cost, interval_str, latency, settlement, summarize
from analysis.statistics import bootstrap_mean_interval, percentile, two_proportion_z, wilson_interval
from bench.longmemeval import make_synthetic_dataset
from bench.runner import RunConfig, Runner
from proxy.fake_upstream import EMBED_DIM

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_known_values():
    iv = wilson_interval(472, 500)
    assert abs(iv.point - 0.944) < 1e-9
    assert 0.920 < iv.low < 0.923 and 0.960 < iv.high < 0.963  # Wilson 95% for 472/500 ≈ [0.9208, 0.9614]
    z = wilson_interval(0, 10)
    assert z.point == 0.0 and z.low == 0.0 and 0.27 < z.high < 0.28
    full = wilson_interval(10, 10)
    assert abs(full.high - 1.0) < 1e-9 and 0.72 < full.low < 0.73
    assert wilson_interval(0, 0).n == 0
    assert interval_str(iv.as_dict()).startswith("94.4 [92.")


def test_percentile_and_bootstrap_and_ztest():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == 50 and percentile(vals, 99) == 99 and percentile(vals, 100) == 100 and percentile([], 50) is None
    b = bootstrap_mean_interval([1.0] * 20)
    assert b.low == b.high == 1.0
    z = two_proportion_z(472, 500, 400, 500)
    assert z["z"] > 3 and z["p_value"] < 0.01
    assert two_proportion_z(50, 100, 50, 100)["p_value"] > 0.99


@pytest.fixture
async def synthetic_run(proxy_server, proxy_log, tmp_path):
    ds = make_synthetic_dataset(tmp_path / "syn.json", n_instances=6, seed=5)
    system_cfg = load_config(ROOT / "configs/mem0.yaml")
    bcfg = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s.yaml").read_text())
    bcfg["reader"]["model"] = "fake-model"
    bcfg["judge"]["model"] = "fake-model"
    overrides = {"mem0": {"llm": {"model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dims": None}, "vector_store": {"mode": "embedded", "path": str(tmp_path / "q"), "embedding_model_dims": EMBED_DIM}, "history_db_path": str(tmp_path / "h.db")}}
    cfg = RunConfig(system="mem0", benchmark="longmemeval_s", seed=1, config_path="configs/mem0.yaml", system_config=system_cfg, benchmark_config=bcfg, proxy_url=proxy_server.url, out_dir=tmp_path / "runs", limit=4, overrides=overrides)
    runner = Runner(cfg, ds)
    summary = await runner.run()
    import asyncio

    await asyncio.sleep(0.3)
    return runner.out_dir, proxy_log, summary


async def test_metrics_from_raw_traces(synthetic_run):
    run_dir, proxy_log, summary = synthetic_run
    trace = load_run(run_dir, [proxy_log])
    acc = accuracy(trace)
    assert acc["instances"] == 4 and acc["judged"] == 4 and acc["errored_instances"] == 0
    assert acc["correct"] == summary["correct"]
    assert acc["accuracy_judged"]["n"] == 4 and sum(v["n"] for v in acc["per_type"].values()) == 4
    st = settlement(trace)
    assert st["n_settled"] == 4 and st["n_timeouts"] == 0 and st["lag_ms"]["n"] == 4 and st["lag_ms"]["p50"] is not None
    lat = latency(trace)
    assert lat["write_ack_ms"]["n"] == 12 and lat["read_ms"]["n"] == 4 and lat["answer_ms"]["n"] == 4
    c = cost(trace)
    assert c["available"] and c["unattributed"] == 0 and c["errors"] == 0
    assert set(c["by_operation"]) == {"write", "read", "settle", "answer", "judge"}
    assert c["by_operation"]["answer"]["calls"] == 4 and c["by_operation"]["judge"]["calls"] == 4
    assert c["total_tokens"] == sum(d["total_tokens"] for d in c["by_operation"].values()) > 0
    s = summarize(trace)
    json.dumps(s)
    assert s["run_id"] == trace.run_id and s["dataset"]["synthetic"] is True


async def test_load_run_ignores_other_runs_in_proxy_log(synthetic_run, tmp_path):
    run_dir, proxy_log, _ = synthetic_run
    # Append a foreign MODEL_CALL row; it must not be attributed to this run.
    with open(proxy_log, "a") as fh:
        fh.write(json.dumps({"event_type": "MODEL_CALL", "run_id": "other-run", "session_id": "x", "operation": "write", "status": "success", "total_tokens": 999999, "prompt_tokens": 1, "completion_tokens": 1, "unattributed": False, "model": "m", "endpoint": "chat", "duration_ms": 1, "upstream_duration_ms": 1}) + "\n")
    trace = load_run(run_dir, [proxy_log])
    assert (trace.model_calls["run_id"] == trace.run_id).all()
    assert cost(trace)["total_tokens"] < 999999
