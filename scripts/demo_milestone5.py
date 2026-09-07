"""Milestone 5 demonstration: Zep/Graphiti adapter reset/write/read/settlement over repeated sessions.

Offline (default): fake upstream + proxy as subprocesses, Graphiti with an
embedded Kuzu graph.  Real: --upstream + a Neo4j from compose/zep.

    python scripts/demo_milestone5.py --sessions 3
    python scripts/demo_milestone5.py --upstream https://api.openai.com/v1 --graph neo4j
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
    adapter = build_adapter(cfg, proxy_base_url=proxy_url, seed=42, run_id=f"demo-m5-{int(time.time())}", event_sink=logger.append, overrides=overrides)
    await adapter.start()
    failures = 0
    fp = adapter.config_fingerprint()
    print(f"\nfingerprint: {json.dumps({k: fp[k] for k in ('system', 'deployment', 'system_version', 'graph_store', 'retrieval')})}\n")
    print(f"{'session':10} {'ack_ms':>8} {'facts':>5} {'read_ms':>8} {'hits':>5} {'canary_ack':>11} {'lag_ms':>8} {'polls':>5} {'settled':>7}")
    try:
        for i in range(sessions):
            sid = f"demo-{i}"
            await adapter.reset(sid)
            facts = [f"I moved to apartment {100 + i} on Maple Street in Lisbon", f"My cat is called Pixel{i}", f"I take the 7:{10 + i:02d} train to work"]
            msgs = [{"role": "user", "content": facts[0]}, {"role": "assistant", "content": "Got it."}, {"role": "user", "content": facts[1]}, {"role": "user", "content": facts[2]}]
            wr = await adapter.write(sid, msgs, metadata={"lme_session_id": f"s{i}", "session_date": "2023/05/20 (Sat) 02:21"})
            rr = await adapter.search(sid, facts[1])
            ok_read = facts[1] in rr.context
            st = await adapter.wait_settled(sid)
            lag = st.canary.lag_fields()["ingestion_to_retrievability_lag_ms"]
            if not (ok_read and st.settled):
                failures += 1
            print(f"{sid:10} {wr.ack_latency_ms:8.1f} {wr.memories_added:5d} {rr.latency_ms:8.1f} {len(rr.hits):5d} {st.canary.ack_latency_ms:11.1f} {lag if lag is not None else float('nan'):8.1f} {st.polls:5d} {str(st.settled):>7}")
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
    ap.add_argument("--config", default=str(ROOT / "configs/zep.yaml"))
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--graph", choices=["kuzu", "neo4j"], default="kuzu")
    ap.add_argument("--proxy-port", type=int, default=8811)
    ap.add_argument("--fake-port", type=int, default=8899)
    ap.add_argument("--out", default=str(ROOT / "results/raw/demo-m5"))
    args = ap.parse_args()

    real = bool(args.upstream)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proxy_log, adapter_log = out / "proxy.jsonl", out / "adapter.jsonl"
    for p in (proxy_log, adapter_log):
        if p.exists():
            p.unlink()
    state = Path(tempfile.mkdtemp(prefix="memharness-m5-"))
    cfg = load_config(args.config)
    overrides: dict = {"graphiti": {}}
    if not real:
        overrides["graphiti"].update({"llm": {"model": "fake-model", "small_model": "fake-model"}, "embedder": {"model": "fake-embed", "embedding_dim": EMBED_DIM}, "reranker": {"model": "fake-model"}})
    if args.graph == "kuzu":
        overrides["graphiti"]["graph_store"] = {"provider": "kuzu", "path": str(state / "graph.kuzu")}

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
        print(f"proxy {proxy_url} -> {upstream}; graph={args.graph}; logs in {out}")
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
    print(f"adapter events: {dict(collections.Counter(e['event_type'] for e in read_events(adapter_log)))}")
    print(f"\n{failures} failure(s)")
    return 1 if failures or unattributed else 0


if __name__ == "__main__":
    sys.exit(main())
