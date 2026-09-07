"""Milestone 3 demonstration: the LongMemEval vertical slice through `python -m bench.run`.

Starts the fake upstream and the proxy as subprocesses, generates a synthetic
LongMemEval-schema dataset (unless --dataset points at the real file), runs
`bench.run` for Mem0 with an embedded Qdrant, and summarises the run log.

    python scripts/demo_milestone3.py --limit 4
    python scripts/demo_milestone3.py --upstream https://api.openai.com/v1 --dataset data/longmemeval_s_cleaned.json --limit 1 --qdrant server
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.longmemeval import make_synthetic_dataset  # noqa: E402
from proxy.fake_upstream import EMBED_DIM  # noqa: E402
from proxy.logging import read_events  # noqa: E402

PY = sys.executable


def wait_healthy(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"{url} did not become healthy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream")
    ap.add_argument("--upstream-api-key", default=os.environ.get("PROXY_UPSTREAM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--dataset", default=None, help="real LongMemEval file; default: synthetic")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qdrant", choices=["embedded", "server"], default="embedded")
    ap.add_argument("--proxy-port", type=int, default=8811)
    ap.add_argument("--fake-port", type=int, default=8899)
    ap.add_argument("--out", default=str(ROOT / "results/raw/demo-m3"))
    args = ap.parse_args()

    real = bool(args.upstream)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proxy_log = out / "proxy.jsonl"
    if proxy_log.exists():
        proxy_log.unlink()
    state = Path(tempfile.mkdtemp(prefix="memharness-m3-"))
    dataset = args.dataset or str(state / "synthetic_longmemeval.json")
    if not args.dataset:
        make_synthetic_dataset(dataset, n_instances=max(args.limit, 4), seed=args.seed)

    overrides: dict = {"mem0": {"history_db_path": str(state / "history.db")}}
    if not real:
        overrides["mem0"].update({"llm": {"model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dims": None}, "vector_store": {"embedding_model_dims": EMBED_DIM}})
    if args.qdrant == "embedded":
        overrides["mem0"].setdefault("vector_store", {}).update({"mode": "embedded", "path": str(state / "qdrant")})
    bcfg_path = state / "longmemeval_s.yaml"
    import yaml

    bcfg = yaml.safe_load((ROOT / "configs/benchmarks/longmemeval_s.yaml").read_text())
    if not real:
        bcfg["reader"]["model"] = "fake-model"
        bcfg["judge"]["model"] = "fake-model"
    bcfg_path.write_text(yaml.safe_dump(bcfg))

    upstream = args.upstream or f"http://127.0.0.1:{args.fake_port}/v1"
    env = {**os.environ, "PROXY_LOG_PATH": str(proxy_log), "PROXY_UPSTREAM_BASE_URL": upstream, "PROXY_REQUIRE_ATTRIBUTION": "1"}
    if args.upstream_api_key:
        env["PROXY_UPSTREAM_API_KEY"] = args.upstream_api_key
    procs: list[subprocess.Popen] = []
    try:
        if not real:
            procs.append(subprocess.Popen([PY, "-m", "proxy.fake_upstream", "--port", str(args.fake_port)], cwd=ROOT))
            wait_healthy(f"http://127.0.0.1:{args.fake_port}/v1/models")
        procs.append(subprocess.Popen([PY, "-m", "proxy", "--port", str(args.proxy_port), "--log-level", "warning"], cwd=ROOT, env=env))
        proxy_url = f"http://127.0.0.1:{args.proxy_port}"
        wait_healthy(f"{proxy_url}/healthz")
        cmd = [PY, "-m", "bench.run", "--system", "mem0", "--benchmark", "longmemeval_s", "--seed", str(args.seed), "--config", "configs/mem0.yaml", "--benchmark-config", str(bcfg_path), "--dataset", dataset, "--proxy", proxy_url, "--out", str(out / "runs"), "--limit", str(args.limit), "--override-json", json.dumps(overrides)]
        if not args.dataset:
            cmd.append("--allow-synthetic")
        print("$", " ".join(cmd[:12]), "...")
        rc = subprocess.call(cmd, cwd=ROOT, env={**os.environ, "MEM0_TELEMETRY": "false"})
    finally:
        for p in reversed(procs):
            p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    runs = sorted((out / "runs").glob("*/run.jsonl"), key=lambda p: p.stat().st_mtime)
    if not runs:
        print("no run log produced")
        return 1
    events = list(read_events(runs[-1]))
    kinds = collections.Counter(e["event_type"] for e in events)
    print(f"\nrun log: {runs[-1]} ({len(events)} events)")
    print("event counts:", dict(kinds))
    print(f"\n{'question_id':14} {'type':26} {'ingest_ms':>9} {'settle_ms':>9} {'read_ms':>8} {'answer_ms':>9} {'judge_ms':>8} correct")
    for e in events:
        if e["event_type"] == "INSTANCE_END":
            print(f"{e['question_id']:14} {e['question_type']:26} {e.get('ingest_ms') or 0:9.1f} {e.get('settle_ms') or 0:9.1f} {e.get('read_ms') or 0:8.1f} {e.get('answer_ms') or 0:9.1f} {e.get('judge_ms') or 0:8.1f} {e['correct']} {e['error'] or ''}")
    calls = [e for e in read_events(proxy_log) if e["event_type"] == "MODEL_CALL"]
    by_op = collections.Counter(c["operation"] for c in calls)
    tok = collections.Counter()
    for c in calls:
        tok[c["operation"]] += c.get("total_tokens") or 0
    print(f"\nproxy: {len(calls)} model calls by operation {dict(by_op)}; tokens {dict(tok)}; unattributed {sum(1 for c in calls if c['unattributed'])}")
    end = next(e for e in events if e["event_type"] == "RUN_END")
    print(f"RUN_END: {json.dumps({k: end[k] for k in ('instances', 'correct', 'errors', 'accuracy_convenience')})}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
