"""Milestone 4: reproduction report for Mem0 on LongMemEval-S.

Consumes completed run directories (results/raw/runs/<run_id>) and proxy logs;
never runs a memory system.  Produces results/summaries/mem0-reproduction-<stamp>.{json,md}
with accuracy + Wilson intervals, the per-type comparison against the
pre-registered published reference, the verdict under the pre-registered
tolerance rule, and the configuration-discrepancy checklist derived from the
run manifest versus Mem0's published setup.

    python scripts/reproduce_mem0.py --run results/raw/runs/<armA_run_id> --run results/raw/runs/<armB_run_id> \
        --proxy-log results/raw/proxy/campaign.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ingest import load_run  # noqa: E402
from analysis.metrics import interval_str, summarize  # noqa: E402
from bench.prompts import prompt_hashes  # noqa: E402

# What Mem0's published numbers were obtained with (see docs/milestone4_reproduction.md for sources).
PUBLISHED_SETUP = {
    "deployment": "Mem0 managed platform (proprietary optimizations; vendor: OSS will not match)",
    "pipeline": "v3 ADD-only extraction + entity linking + BM25 + temporal scoring",
    "extraction_model": "unspecified 'production-representative model stack' (OSS default in their repo: gpt-4o-mini)",
    "embedding_model": "unspecified for platform (OSS default: text-embedding-3-small; OSS table: Qwen 600M via SageMaker)",
    "retrieval_top_k": 200,
    "retrieval_threshold": "n/a for platform (OSS: mem0ai default 0.1)",
    "reader_prompt": "Mem0 ANSWER_GENERATION_PROMPT (mem0_v3) with <mem_thinking>, benchmark-specific rules",
    "reader_model": "README: gpt-4o; run.py default: gpt-5; OSS table: GPT-5",
    "judge_prompt": "Mem0 unified JUDGE_PROMPT (mem0_unified) with <judge_thinking>",
    "judge_model": "README: gpt-4o; run.py default: gpt-5",
    "judge_label_rule": "last yes/no line after </judge_thinking>",
    "ingest_granularity": "pair (one add per user/assistant pair)",
    "date_conveyance": "platform: timestamp=session date; OSS server: not forwarded",
    "dataset": "longmemeval_s_cleaned.json (HF xiaowu0162/longmemeval-cleaned)",
    "feature_flags": "spaCy en_core_web_sm installed in their image; fastembed BM25 (Qdrant) expected",
    "software": "mem0ai @ git+...@feat/v3-pipeline (branch no longer exists; exact commit unrecorded)",
}


def verdict_for(acc: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered rule (docs/milestone4_reproduction.md, section 3)."""
    iv = acc["accuracy_judged"]
    if not ref or ref.get("value") is None or iv["n"] == 0:
        return {"status": "no_reference_or_no_data"}
    diff = iv["point"] - ref["value"]
    within_tol = abs(diff) <= ref.get("tolerance_abs", 0.03)
    inside_ci = iv["low"] <= ref["value"] <= iv["high"]
    status = "reproduced" if (within_tol or inside_ci) else "not_reproduced"
    return {"status": status, "published": ref["value"], "measured": iv["point"], "diff": diff, "ci95": [iv["low"], iv["high"]], "within_tolerance": within_tol, "published_inside_ci": inside_ci, "tolerance_abs": ref.get("tolerance_abs", 0.03), "n": iv["n"], "protocol_matches": ref.get("protocol_matches_this_config", None)}


def discrepancy_checklist(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    fp = manifest.get("adapter_fingerprint", {})
    bc = manifest.get("benchmark_config", {})
    reader = manifest.get("reader", {})
    judge = manifest.get("judge", {})
    ph = prompt_hashes()
    flags = fp.get("feature_flags", {})
    rows = [
        ("deployment", fp.get("deployment"), PUBLISHED_SETUP["deployment"], False, "platform-only optimizations are not reproducible by construction"),
        ("system_version", f"mem0ai {fp.get('system_version')} (PyPI)", PUBLISHED_SETUP["software"], False, "published build pinned an unrecorded commit of a now-deleted branch"),
        ("extraction_model", fp.get("memory_model", {}).get("model"), PUBLISHED_SETUP["extraction_model"], None, "affects which facts are stored"),
        ("embedding_model", fp.get("embedding_model", {}).get("model"), PUBLISHED_SETUP["embedding_model"], None, "affects retrieval quality"),
        ("retrieval_top_k", fp.get("retrieval", {}).get("top_k"), PUBLISHED_SETUP["retrieval_top_k"], fp.get("retrieval", {}).get("top_k") == 200, "published: top_200 budget"),
        ("retrieval_threshold", fp.get("retrieval", {}).get("threshold"), PUBLISHED_SETUP["retrieval_threshold"], None, "mem0ai 2.x default 0.1 drops low-similarity memories"),
        ("reader_prompt", reader.get("prompt"), "mem0_v3", reader.get("prompt_template_sha256") == ph["reader_mem0_v3"], "Mem0's prompt embeds LongMemEval-specific rules; official prompt is generic"),
        ("reader_model", reader.get("model"), PUBLISHED_SETUP["reader_model"], None, "vendor docs disagree (gpt-4o vs gpt-5)"),
        ("judge_prompt", judge.get("style"), "mem0_unified", judge.get("style") == "mem0_unified", "Mem0's judge instructs leniency ('lean toward yes'); official judge is 10-token yes/no"),
        ("judge_model", judge.get("model"), PUBLISHED_SETUP["judge_model"], None, "vendor states +/-1 point judge inconsistency"),
        ("ingest_granularity", manifest.get("ingest", {}).get("granularity"), "pair", manifest.get("ingest", {}).get("granularity") == "pair", "granularity changes extraction context per call"),
        ("date_conveyance", manifest.get("ingest", {}).get("date_mode"), PUBLISHED_SETUP["date_conveyance"], None, "mem0ai OSS rejects timestamp; platform accepts it; their OSS server sends nothing"),
        ("dataset", f"{manifest.get('dataset', {}).get('name')} sha256={str(manifest.get('dataset', {}).get('sha256'))[:12]} synthetic={manifest.get('dataset', {}).get('synthetic')}", PUBLISHED_SETUP["dataset"], (not manifest.get("dataset", {}).get("synthetic")) or None, "must be the cleaned 2025-09 file"),
        ("spacy_entity_extraction", flags.get("spacy_en_core_web_sm"), True, flags.get("spacy_en_core_web_sm") is True, "off -> no entity boosting"),
        ("bm25_hybrid", flags.get("bm25_hybrid_search_enabled"), True, flags.get("bm25_hybrid_search_enabled") is True, "off -> semantic-only retrieval"),
        ("settlement_check", manifest.get("settlement_policy"), "none (their runner has no settlement step)", None, "harness addition; does not change accuracy for a synchronous store"),
        ("overrides_present", manifest.get("overrides_present"), False, manifest.get("overrides_present") is False, "paper runs must have no overrides"),
    ]
    return [{"item": i, "this_run": a, "published": b, "match": m, "note": n} for i, a, b, m, n in rows]


def render_md(reports: list[dict[str, Any]]) -> str:
    out = ["# Mem0 LongMemEval-S reproduction report", "", f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} from raw traces only.", ""]
    for r in reports:
        s = r["summary"]
        acc = s["accuracy"]
        out += [f"## Run `{s['run_id']}`", "", f"- protocol: **{r['protocol']}**; system config `{s['configuration_id']}`; git `{s['git_commit']}`; seed {s['seed']}", f"- dataset: `{s['dataset']['name']}` sha256 `{s['dataset']['sha256'][:16]}` synthetic={s['dataset']['synthetic']}; instances {acc['instances']} (judged {acc['judged']}, errored {acc['errored_instances']})", f"- overrides present: {s['overrides_present']}", "", f"**Accuracy (judged):** {interval_str(acc['accuracy_judged'])}  ", f"**Accuracy (errors counted wrong):** {interval_str(acc['accuracy_errors_as_wrong'])}", ""]
        v = r["verdict"]
        if v.get("status") in ("reproduced", "not_reproduced"):
            out += [f"**Verdict vs published {v['published']:.3f} (tolerance ±{v['tolerance_abs']:.2f}):** `{v['status']}` (diff {v['diff'] * 100:+.1f} pts; published inside 95% CI: {v['published_inside_ci']}; protocol matches: {v['protocol_matches']})", ""]
        else:
            out += [f"**Verdict:** `{v.get('status')}`", ""]
        out += ["| question type | this run | published | n |", "|---|---|---|---|"]
        ref_types = (r.get("reference") or {}).get("per_type", {}) or {}
        for qt, iv in sorted(acc["per_type"].items()):
            pub = ref_types.get(qt)
            out.append(f"| {qt} | {interval_str(iv)} | {pub * 100:.1f} | {iv['n']} |" if pub is not None else f"| {qt} | {interval_str(iv)} | – | {iv['n']} |")
        out += ["", "### Cost and latency (proxy log)"]
        c = s["cost"]
        if c.get("available"):
            out += [f"- model calls {c['calls']} (errors {c['errors']}, unattributed {c['unattributed']}); total tokens {c['total_tokens']}; tokens/instance mean {c['tokens_per_instance_mean']:.0f}, p50 {c['tokens_per_instance_p50']}", "", "| operation | calls | prompt tok | completion tok | total tok | upstream p50 ms |", "|---|---|---|---|---|---|"]
            for op, d in sorted(c["by_operation"].items()):
                out.append(f"| {op} | {d['calls']} | {d['prompt_tokens']} | {d['completion_tokens']} | {d['total_tokens']} | {d['upstream_ms_p50']} |")
        else:
            out.append(f"- {c.get('note')}")
        lat, st = s["latency"], s["settlement"]
        out += ["", f"- write ack p50/p95/p99 ms: {lat['write_ack_ms']['p50']}/{lat['write_ack_ms']['p95']}/{lat['write_ack_ms']['p99']} (n={lat['write_ack_ms']['n']}); read p50/p95/p99 ms: {lat['read_ms']['p50']}/{lat['read_ms']['p95']}/{lat['read_ms']['p99']} (n={lat['read_ms']['n']})", f"- settlement: {st['n_settled']} settled, {st['n_timeouts']} timeouts; lag p50/p95 ms ≤ {st['lag_ms']['p50']}/{st['lag_ms']['p95']} (first-poll fraction {st['first_poll_fraction']})", "", "### Discrepancy checklist (this run vs Mem0's published setup)", "", "| item | this run | published | match | note |", "|---|---|---|---|---|"]
        for row in r["checklist"]:
            m = {True: "yes", False: "NO", None: "?"}[row["match"]]
            out.append(f"| {row['item']} | {row['this_run']} | {row['published']} | {m} | {row['note']} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="run directory (repeatable)")
    ap.add_argument("--proxy-log", action="append", default=None, help="proxy JSONL (repeatable); default: path recorded in the manifest")
    ap.add_argument("--out-dir", default=str(ROOT / "results/summaries"))
    ap.add_argument("--stamp", default=None)
    args = ap.parse_args()

    reports = []
    for rd in args.run:
        trace = load_run(rd, args.proxy_log)
        summary = summarize(trace)
        bc = trace.manifest.get("benchmark_config", {})
        protocol = trace.manifest.get("protocol") or bc.get("protocol", "longmemeval-official")
        ref = (bc.get("published_reference") or {}).get("mem0")
        if trace.manifest.get("dataset", {}).get("synthetic"):
            verdict = {"status": "not_applicable_synthetic_dataset"}
        else:
            verdict = verdict_for(summary["accuracy"], ref)
        reports.append({"run_dir": str(rd), "protocol": protocol, "summary": summary, "reference": ref, "verdict": verdict, "checklist": discrepancy_checklist(trace.manifest)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (out_dir / f"mem0-reproduction-{stamp}.json").write_text(json.dumps({"generated_at": stamp, "published_setup": PUBLISHED_SETUP, "reports": reports}, indent=1, default=str))
    md = render_md(reports)
    (out_dir / f"mem0-reproduction-{stamp}.md").write_text(md)
    print(md)
    print(f"\nwrote {out_dir / f'mem0-reproduction-{stamp}.md'} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
