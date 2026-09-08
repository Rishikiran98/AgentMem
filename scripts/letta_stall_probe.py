"""Rule out (or reproduce) the Letta 0.16.8 per-step stall (evidence for README, Milestone 6).

Runs the offline stack (fake provider, proxy, retired Letta V1 server on embedded
PostgreSQL) and sends N single-turn writes to one agent, printing the ack latency of
each.  On CPython's stdlib selector event loop the 2nd/3rd write stalls for ~60 s
(asyncpg's default connect timeout): the teardown of the server's per-request OpenAI
HTTP client deregisters an already-reused file descriptor and kills the connect of
the NullPool asyncpg socket that got it.  On uvloop, which the official image runs on
(all extras), no write exceeds a few seconds.  The launcher refuses an environment
without uvloop; to reproduce the stall anyway pass --allow-selector-loop after
``VIRTUAL_ENV=.venv-letta uv pip uninstall uvloop``.

    .venv/bin/python scripts/letta_stall_probe.py --writes 10
    .venv/bin/python scripts/letta_stall_probe.py --writes 6 --env LETTA_TRACK_PROVIDER_TRACE=false
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.registry import build_adapter, load_config  # noqa: E402
from proxy.app import create_app  # noqa: E402
from proxy.fake_upstream import EMBED_DIM, create_fake_upstream  # noqa: E402
from proxy.settings import ProxySettings  # noqa: E402
from tests.conftest import LiveServer, _free_port  # noqa: E402
import tests.letta_server as letta_server_mod  # noqa: E402
from tests.letta_server import LettaServerProcess, letta_env_available, letta_env_runtime  # noqa: E402


async def run(proxy_url: str, server_url: str, writes: int, stall_s: float) -> int:
    cfg = load_config(ROOT / "configs" / "letta.yaml")
    overrides = {"letta": {"server_url": server_url, "llm": {"model": "fake-model", "context_window": 32000}, "embedding": {"model": "fake-embed", "embedding_dim": EMBED_DIM}}}
    adapter = build_adapter(cfg, proxy_base_url=proxy_url, seed=1, run_id=f"stall-probe-{int(time.time())}", event_sink=lambda e: None, overrides=overrides)
    await adapter.start()
    stalls = 0
    try:
        await adapter.reset("probe")
        for i in range(writes):
            msgs = [{"role": "user", "content": f"My favourite number is {40 + i} and I live in city {i}."}, {"role": "assistant", "content": "Noted."}]
            wr = await adapter.write("probe", msgs, metadata={})
            flag = "  <-- stall" if wr.ack_latency_ms >= stall_s * 1000 else ""
            stalls += bool(flag)
            print(f"write {i}: {wr.ack_latency_ms / 1000:7.2f}s  memory_writes={wr.memories_added} steps={wr.native.get('steps')} stop={wr.native.get('stop_reason')}{flag}", flush=True)
    finally:
        await adapter.reset_all()
        await adapter.close()
    return stalls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--writes", type=int, default=8)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="server environment override (repeatable)")
    ap.add_argument("--allow-selector-loop", action="store_true", help="run even if uvloop is absent from the Letta environment (reproduces the stall)")
    ap.add_argument("--stall-threshold-s", type=float, default=30.0)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    if not letta_env_available():
        print("Letta server environment not built (run scripts/setup_letta_env.sh)", file=sys.stderr)
        return 2
    work = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="letta-stall-probe-"))
    fake = LiveServer(create_fake_upstream(), _free_port()).start("/v1/models")
    proxy = LiveServer(create_app(ProxySettings(upstream_base_url=fake.url + "/v1", upstream_api_key="fake", log_path=work / "proxy.jsonl", upstream_timeout_s=30.0)), _free_port()).start("/healthz")
    env = dict(kv.split("=", 1) for kv in a.env)
    runtime = letta_env_runtime()
    if a.allow_selector_loop:
        letta_server_mod.REQUIRED_RUNTIME = {"event_loop": runtime["event_loop"]}
    print(f"server runtime: {runtime}; env override: {env or 'none (harness default)'}; workdir {work}")
    t0 = time.time()
    srv = LettaServerProcess(work, openai_base_url=fake.url + "/v1", server_env=env).start()
    print(f"server up in {time.time() - t0:.1f}s at {srv.url}")
    try:
        stalls = asyncio.run(run(proxy.url, srv.url, a.writes, a.stall_threshold_s))
    finally:
        srv.stop()
        proxy.stop()
        fake.stop()
    log = (work / "letta-server.log").read_text()
    print(f"\nstalls (>= {a.stall_threshold_s:.0f}s): {stalls}/{a.writes};  server log: 'Failed to write to PostgresProviderTraceBackend' x{log.count('Failed to write to PostgresProviderTraceBackend')}")
    return 1 if stalls else 0


if __name__ == "__main__":
    sys.exit(main())
