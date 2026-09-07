"""Metric derivations from raw traces.  Every number here is traceable to event rows."""
from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.ingest import RunTrace
from analysis.statistics import Interval, percentile, wilson_interval
from load.histogram import quantile

QUESTION_TYPES = ("single-session-user", "single-session-assistant", "single-session-preference", "temporal-reasoning", "knowledge-update", "multi-session")


def accuracy(trace: "RunTrace | list[bool]") -> dict[str, Any]:
    """Overall and per-type accuracy from JUDGE_END events, with Wilson 95% intervals.

    Given a plain list of per-item outcomes instead of a trace, returns the
    list-level form ``{"n", "accuracy", "ci95"}`` (see ``accuracy_of``).

    Instances that errored before judging are reported separately and excluded
    from the accuracy denominator only when `count_errors_as_wrong` is False in
    the caller; here both views are returned.
    """
    if not isinstance(trace, RunTrace):
        return accuracy_of(list(trace))
    judged = trace.of("JUDGE_END")
    ends = trace.of("INSTANCE_END")
    errored = ends[ends["error"].notna()] if "error" in ends else ends.iloc[0:0]
    n_judged = len(judged)
    n_correct = int(judged["correct"].sum()) if n_judged else 0
    n_total = len(ends)
    per_type: dict[str, dict[str, Any]] = {}
    for qt, grp in judged.groupby("question_type"):
        ci = wilson_interval(int(grp["correct"].sum()), len(grp))
        per_type[str(qt)] = ci.as_dict()
    abst = judged[judged["question_id"].astype(str).str.endswith("_abs")]
    return {
        "judged": n_judged,
        "correct": n_correct,
        "errored_instances": int(len(errored)),
        "instances": n_total,
        "accuracy_judged": wilson_interval(n_correct, n_judged).as_dict(),
        "accuracy_errors_as_wrong": wilson_interval(n_correct, n_total).as_dict(),
        "per_type": per_type,
        "abstention": wilson_interval(int(abst["correct"].sum()), len(abst)).as_dict() if len(abst) else None,
    }


def settlement(trace: RunTrace) -> dict[str, Any]:
    settled = trace.of("WRITE_SETTLED")
    timeouts = trace.of("SETTLEMENT_TIMEOUT")
    lag = settled["ingestion_to_retrievability_lag_ms"].dropna().tolist() if len(settled) else []
    ack = settled["write_ack_latency_ms"].dropna().tolist() if len(settled) else []
    polls = settled["polls"].tolist() if len(settled) else []
    return {
        "n_settled": int(len(settled)),
        "n_timeouts": int(len(timeouts)),
        "first_poll_fraction": (sum(1 for p in polls if p == 1) / len(polls)) if polls else None,
        "lag_ms": {"p50": percentile(lag, 50), "p95": percentile(lag, 95), "p99": percentile(lag, 99), "n": len(lag), "note": "upper bound when polls == 1: true lag <= one read latency"},
        "canary_ack_ms": {"p50": percentile(ack, 50), "p95": percentile(ack, 95), "n": len(ack)},
    }


def latency(trace: "RunTrace | list[float]") -> dict[str, Any]:
    out: dict[str, Any] = {}
    acks = trace.of("WRITE_ACK")
    reads = trace.of("READ_END")
    reads = reads[reads["operation"] == "read"] if "operation" in reads else reads
    for name, series in (("write_ack_ms", acks["ack_latency_ms"] if len(acks) else pd.Series(dtype=float)), ("read_ms", reads["latency_ms"] if len(reads) else pd.Series(dtype=float))):
        vals = series.dropna().tolist()
        out[name] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "p99": percentile(vals, 99), "n": len(vals)}
    ends = trace.of("INSTANCE_END")
    for col in ("ingest_ms", "settle_ms", "answer_ms", "judge_ms"):
        vals = ends[col].dropna().tolist() if col in ends else []
        out[col] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "n": len(vals)}
    return out


def cost(trace: RunTrace) -> dict[str, Any]:
    """Token consumption and call counts by operation from the proxy log (authoritative)."""
    if not isinstance(trace, RunTrace):
        return latency_of(list(trace))
    calls = trace.model_calls
    if calls.empty:
        return {"available": False, "note": "no proxy MODEL_CALL rows found for this run_id"}
    ok = calls[calls["status"] == "success"]
    by_op = {}
    for op, grp in calls.groupby("operation", dropna=False):
        g_ok = grp[grp["status"] == "success"]
        by_op[str(op)] = {
            "calls": int(len(grp)),
            "errors": int((grp["status"] != "success").sum()),
            "prompt_tokens": int(g_ok["prompt_tokens"].fillna(0).sum()),
            "completion_tokens": int(g_ok["completion_tokens"].fillna(0).sum()),
            "total_tokens": int(g_ok["total_tokens"].fillna(0).sum()),
            "upstream_ms_p50": percentile(g_ok["upstream_duration_ms"].dropna().tolist(), 50),
        }
    n_instances = max(1, len(trace.of("INSTANCE_END")))
    per_instance = ok.groupby("session_id")["total_tokens"].sum() if len(ok) else pd.Series(dtype=float)
    return {
        "available": True,
        "calls": int(len(calls)),
        "errors": int((calls["status"] != "success").sum()),
        "unattributed": int(calls["unattributed"].sum()) if "unattributed" in calls else None,
        "total_tokens": int(ok["total_tokens"].fillna(0).sum()),
        "tokens_per_instance_mean": float(ok["total_tokens"].fillna(0).sum() / n_instances),
        "tokens_per_instance_p50": percentile(per_instance.tolist(), 50),
        "by_operation": by_op,
        "models": sorted(str(m) for m in calls["model"].dropna().unique()),
    }


def summarize(trace: RunTrace) -> dict[str, Any]:
    m = trace.manifest
    return {
        "run_id": trace.run_id,
        "system": m["system"],
        "seed": m["seed"],
        "configuration_id": m["configuration_id"],
        "git_commit": m.get("host", {}).get("git", {}).get("commit"),
        "dataset": m["dataset"],
        "overrides_present": m.get("overrides_present"),
        "accuracy": accuracy(trace),
        "settlement": settlement(trace),
        "latency": latency(trace),
        "cost": cost(trace),
    }


def interval_str(iv: dict[str, Any] | Interval, pct: bool = True) -> str:
    d = iv.as_dict() if isinstance(iv, Interval) else iv
    if d is None or d.get("n", 0) == 0:
        return "n/a"
    f = 100 if pct else 1
    return f"{d['point'] * f:.1f} [{d['low'] * f:.1f}, {d['high'] * f:.1f}] (n={d['n']})"


# --------------------------------------------------------------------------- #
# List-level helpers for the operational experiments (no trace object needed).
# --------------------------------------------------------------------------- #
def latency_of(values: list[float]) -> dict[str, Any]:
    """Median / IQR / p95 / p99 over raw samples; p99 is withheld below 100 samples."""
    xs = sorted(values)
    q1, q3 = quantile(xs, 0.25), quantile(xs, 0.75)
    return {"n": len(xs), "median_ms": quantile(xs, 0.5), "iqr_ms": None if q1 is None else q3 - q1, "p95_ms": quantile(xs, 0.95), "p99_ms": quantile(xs, 0.99) if len(xs) >= 100 else None}


def accuracy_of(correct: list[bool]) -> dict[str, Any]:
    """Proportion correct with a Wilson 95% interval, over per-item outcomes."""
    n = len(correct)
    if not n:
        return {"n": 0, "accuracy": None, "ci95": [None, None]}
    iv = wilson_interval(sum(bool(c) for c in correct), n)
    return {"n": n, "accuracy": iv.point, "ci95": [iv.low, iv.high]}


def visibility_intervals(events: list[dict]) -> list[tuple[float, float]]:
    """(lower, upper) retrievability bounds of settled canaries: the last failed poll and the first visible one."""
    return [(e["visibility_lower_bound_ms"], e["visibility_upper_bound_ms"]) for e in events if e.get("settled") and e.get("visibility_upper_bound_ms") is not None]
