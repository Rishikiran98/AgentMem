"""Milestone 4 demonstration: both pre-registered arms through the CLI, then the reproduction report.

Offline (default) this uses a synthetic LongMemEval-schema dataset and the fake
provider, so it validates the apparatus and the report, never the number.

    python scripts/demo_milestone4.py --limit 4
    python scripts/demo_milestone4.py --upstream https://api.openai.com/v1 --dataset data/longmemeval_s_cleaned.json --qdrant server --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.longmemeval import make_synthetic_dataset  # noqa: E402
from proxy.fake_upstream import EMBED_DIM  # noqa: E402

PY = sys.executable
ARMS = {"A": ("configs/mem0.yaml", "configs/benchmarks/longmemeval_s.yaml"), "B": ("configs/mem0-published-protocol.yaml", "configs/benchmarks/longmemeval_s_mem0protocol.yaml")}


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
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--qdrant", choices=["embedded", "server"], default="embedded")
    ap.add_argument("--proxy-port", type=int, default=8811)
    ap.add_argument("--fake-port", type=int, default=8899)
    ap.add_argument("--out", default=str(ROOT / "results/raw/demo-m4"))
    args = ap.parse_args()

    real = bool(args.upstream)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proxy_log = out / "proxy.jsonl"
    if proxy_log.exists():
        proxy_log.unlink()
    state = Path(tempfile.mkdtemp(prefix="memharness-m4-"))
    dataset = args.dataset or str(state / "synthetic_longmemeval.json")
    if not args.dataset:
        make_synthetic_dataset(dataset, n_instances=max(args.limit, 4), seed=args.seed)

    upstream = args.upstream or f"http://127.0.0.1:{args.fake_port}/v1"
    env = {**os.environ, "PROXY_LOG_PATH": str(proxy_log), "PROXY_UPSTREAM_BASE_URL": upstream, "PROXY_REQUIRE_ATTRIBUTION": "1"}
    if args.upstream_api_key:
        env["PROXY_UPSTREAM_API_KEY"] = args.upstream_api_key
    procs: list[subprocess.Popen] = []
    run_dirs: list[Path] = []
    try:
        if not real:
            procs.append(subprocess.Popen([PY, "-m", "proxy.fake_upstream", "--port", str(args.fake_port)], cwd=ROOT))
            wait_healthy(f"http://127.0.0.1:{args.fake_port}/v1/models")
        procs.append(subprocess.Popen([PY, "-m", "proxy", "--port", str(args.proxy_port), "--log-level", "warning"], cwd=ROOT, env=env))
        proxy_url = f"http://127.0.0.1:{args.proxy_port}"
        wait_healthy(f"{proxy_url}/healthz")
        for arm in args.arms.split(","):
            sys_cfg, bcfg_src = ARMS[arm.strip()]
            overrides: dict = {"mem0": {"history_db_path": str(state / f"history-{arm}.db")}}
            if not real:
                overrides["mem0"].update({"llm": {"model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dims": None}, "vector_store": {"embedding_model_dims": EMBED_DIM}})
            if args.qdrant == "embedded":
                overrides["mem0"].setdefault("vector_store", {}).update({"mode": "embedded", "path": str(state / f"qdrant-{arm}")})
            bcfg = yaml.safe_load((ROOT / bcfg_src).read_text())
            bcfg_path = ROOT / bcfg_src
            if not real:
                bcfg["reader"]["model"] = "fake-model"
                bcfg["judge"]["model"] = "fake-model"
                bcfg_path = state / f"bench-{arm}.yaml"
                bcfg_path.write_text(yaml.safe_dump(bcfg))
            run_out = out / "runs"
            before = set(p.name for p in run_out.glob("*")) if run_out.exists() else set()
            cmd = [PY, "-m", "bench.run", "--system", "mem0", "--benchmark", "longmemeval_s", "--seed", str(args.seed), "--config", sys_cfg, "--benchmark-config", str(bcfg_path), "--dataset", dataset, "--proxy", proxy_url, "--out", str(run_out), "--limit", str(args.limit), "--override-json", json.dumps(overrides)]
            if not args.dataset:
                cmd.append("--allow-synthetic")
            print(f"\n=== Arm {arm}: {sys_cfg} + {bcfg_src}")
            rc = subprocess.call(cmd, cwd=ROOT, env={**os.environ, "MEM0_TELEMETRY": "false"})
            new = [p for p in run_out.glob("*") if p.name not in before]
            run_dirs += new
            if rc != 0:
                print(f"arm {arm} exited {rc}")
    finally:
        for p in reversed(procs):
            p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    if not run_dirs:
        return 1
    cmd = [PY, str(ROOT / "scripts/reproduce_mem0.py"), "--proxy-log", str(proxy_log), "--out-dir", str(out / "summaries")]
    for d in run_dirs:
        cmd += ["--run", str(d)]
    print("\n=== report")
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
