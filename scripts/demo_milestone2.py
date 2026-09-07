"""Milestone 2 demonstration: Mem0 adapter reset/write/read/settlement over repeated sessions.

Runs the fake upstream and the proxy as subprocesses (or targets a real provider
with --upstream), drives the Mem0 adapter through N sessions, and prints the
per-session timings plus the proxy's attribution breakdown.

    python scripts/demo_milestone2.py --sessions 5
    python scripts/demo_milestone2.py --upstream https://api.openai.com/v1 --config configs/mem0.yaml
"""
from __future__ import annotations

import argparse
import asyncio
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

from adapters.registry import build_adapter, load_config  # noqa: E402
from proxy.fake_upstream import EMBED_DIM  # noqa: E402
from proxy.logging import AppendOnlyJsonlLogger, read_events  # noqa: E402

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


async def run(proxy_url: str, cfg: dict, overrides: dict, sessions: int, adapter_log: Path) -> int:
    logger = AppendOnlyJsonlLogger(adapter_log)
    adapter = build_adapter(cfg, proxy_base_url=proxy_url, seed=42, run_id=f"demo-m2-{int(time.time())}", event_sink=logger.append, overrides=overrides)
    await adapter.start()
    failures = 0
    print(f"\nfingerprint: {json.dumps({k: adapter.config_fingerprint()[k] for k in ('system', 'system_version', 'vector_store', 'feature_flags')}, indent=None)[:600]}\n")
    print(f"{'session':10} {'ack_ms':>8} {'read_ms':>8} {'hits':>5} {'canary_ack':>11} {'lag_ms':>8} {'polls':>5} {'settled':>7}")
    try:
        for i in range(sessions):
            sid = f"demo-{i}"
            await adapter.reset(sid)
            facts = [f"I moved to apartment {100 + i} on Maple Street", f"My cat is called Pixel{i}", f"I take the 7:{10 + i:02d} train to work"]
            wr = await adapter.write(sid, [{"role": "user", "content": facts[0]}, {"role": "assistant", "content": "Got it."}, {"role": "user", "content": facts[1]}, {"role": "user", "content": facts[2]}])
            rr = await adapter.search(sid, facts[1])
            ok_read = facts[1] in rr.context
            st = await adapter.wait_settled(sid)
            lag = st.canary.lag_fields()["ingestion_to_retrievability_lag_ms"]
            if not (ok_read and st.settled):
                failures += 1
            print(f"{sid:10} {wr.ack_latency_ms:8.1f} {rr.latency_ms:8.1f} {len(rr.hits):5d} {st.canary.ack_latency_ms:11.1f} {lag if lag is not None else float('nan'):8.1f} {st.polls:5d} {str(st.settled):>7}")
            # cross-session isolation check
            if i > 0:
                other = await adapter.read(f"demo-{i - 1}", f"My cat is called Pixel{i - 1}")
                if f"Pixel{i - 1}" not in other or f"Pixel{i}" in other:
                    print(f"  ISOLATION FAILURE between demo-{i - 1} and demo-{i}")
                    failures += 1
        await adapter.reset("demo-0")
        if await adapter.read("demo-0", "apartment") != "":
            print("  RESET FAILURE: demo-0 not empty")
            failures += 1
    finally:
        await adapter.close()
        logger.close()
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream")
    ap.add_argument("--upstream-api-key", default=os.environ.get("PROXY_UPSTREAM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--config", default=str(ROOT / "configs/mem0.yaml"))
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--qdrant", choices=["embedded", "server"], default="embedded", help="embedded Qdrant (no docker) or the compose server")
    ap.add_argument("--proxy-port", type=int, default=8811)
    ap.add_argument("--fake-port", type=int, default=8899)
    ap.add_argument("--out", default=str(ROOT / "results/raw/demo-m2"))
    args = ap.parse_args()

    real = bool(args.upstream)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proxy_log = out / "proxy.jsonl"
    adapter_log = out / "adapter.jsonl"
    for p in (proxy_log, adapter_log):
        if p.exists():
            p.unlink()
    state = Path(tempfile.mkdtemp(prefix="memharness-m2-"))
    cfg = load_config(args.config)
    overrides: dict = {"mem0": {"history_db_path": str(state / "history.db")}}
    if not real:
        overrides["mem0"].update({"llm": {"model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dims": None}})
        overrides["mem0"]["vector_store"] = {"embedding_model_dims": EMBED_DIM}
    if args.qdrant == "embedded":
        overrides["mem0"].setdefault("vector_store", {}).update({"mode": "embedded", "path": str(state / "qdrant")})

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
        print(f"proxy {proxy_url} -> {upstream}; qdrant={args.qdrant}; logs in {out}")
        failures = asyncio.run(run(proxy_url, cfg, overrides, args.sessions, adapter_log))
    finally:
        for p in reversed(procs):
            p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    calls = [e for e in read_events(proxy_log) if e["event_type"] == "MODEL_CALL"]
    by = collections.Counter((c["operation"], c["endpoint"], c["status"]) for c in calls)
    tokens = collections.Counter()
    for c in calls:
        tokens[c["operation"]] += c.get("total_tokens") or 0
    unattributed = sum(1 for c in calls if c["unattributed"] or c["session_id"] is None)
    print("\nproxy attribution (operation, endpoint, status) -> calls")
    for k, v in sorted(by.items()):
        print(f"  {k}: {v}")
    print(f"total tokens by operation: {dict(tokens)}")
    print(f"model calls: {len(calls)}, without full attribution: {unattributed}")
    adapter_events = collections.Counter(e["event_type"] for e in read_events(adapter_log))
    print(f"adapter events: {dict(adapter_events)}")
    print(f"\n{failures} failure(s)")
    return 1 if failures or unattributed else 0


if __name__ == "__main__":
    sys.exit(main())
