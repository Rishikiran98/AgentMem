"""Milestone 1 demonstration: drive the proxy end to end and verify the trace.

By default this starts the deterministic fake upstream and the proxy as real
uvicorn subprocesses, then exercises them with the official ``openai`` SDK and
``httpx`` exactly as the systems under test will.  With ``--upstream`` and an
API key it runs against a real provider instead (the fake upstream is skipped).

    python scripts/demo_milestone1.py
    python scripts/demo_milestone1.py --upstream https://api.openai.com/v1 --chat-model gpt-4o-mini --embed-model text-embedding-3-small

Exit status is non-zero if any verification fails.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proxy.client import ProxyClient  # noqa: E402
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


def check(cond: bool, msg: str, failures: list[str]) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


async def run_demo(proxy_url: str, chat_model: str, embed_model: str, fail_model: str, log_path: Path, real: bool) -> int:
    from openai import AsyncOpenAI

    failures: list[str] = []
    pc = ProxyClient(proxy_url)
    run_id = f"demo-m1-{int(time.time())}"
    # A system under test is configured exactly like this: base_url + api key carrying static tags.
    static = dict(system="mem0", configuration="demo-cfg", seed=42, run_id=run_id, client_id="mem0")
    oai = AsyncOpenAI(base_url=pc.base_url(), api_key=pc.api_key(**static))
    ids: dict[str, str] = {}

    print("\n[1] chat completion via openai SDK (static token + per-request operation header)")
    resp = await oai.chat.completions.with_raw_response.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Say the word hello and nothing else."}],
        max_tokens=5,
        temperature=0,
        extra_headers=pc.headers(operation="write", session_id="sess-1"),
    )
    ids["chat"] = resp.headers["X-Bench-Request-Id"]
    parsed = resp.parse()
    print(f"     -> {parsed.choices[0].message.content!r} usage={parsed.usage.model_dump() if parsed.usage else None}")
    expected_chat_usage = parsed.usage.model_dump() if parsed.usage else None

    print("\n[2] embedding via openai SDK")
    eresp = await oai.embeddings.with_raw_response.create(model=embed_model, input=["alpha beta gamma", "delta"], extra_headers=pc.headers(operation="embed", session_id="sess-1"))
    ids["embed"] = eresp.headers["X-Bench-Request-Id"]
    e = eresp.parse()
    print(f"     -> {len(e.data)} vectors x {len(e.data[0].embedding)} dims, usage={e.usage.model_dump()}")
    expected_embed_usage = e.usage.model_dump()

    print("\n[3] streaming chat via openai SDK (client does NOT request usage; proxy must still record it)")
    stream = await oai.chat.completions.create(model=chat_model, messages=[{"role": "user", "content": "Count from one to five."}], max_tokens=8, stream=True, extra_headers=pc.headers(operation="read", session_id="sess-1"))
    text, saw_usage_chunk = "", False
    async for chunk in stream:
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""
        if getattr(chunk, "usage", None):
            saw_usage_chunk = True
    print(f"     -> streamed {text!r}; usage chunk visible to client: {saw_usage_chunk}")

    print("\n[4] scope-based attribution (system client carries only the static token)")
    async with httpx.AsyncClient(timeout=30) as raw:
        auth = {"Authorization": f"Bearer {pc.api_key(**static)}"}
        body = {"model": chat_model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 3}
        async with pc.scope("mem0", operation="consolidate", session_id="sess-2") as scope_id:
            r = await raw.post(f"{proxy_url}/v1/chat/completions", json=body, headers=auth)
        ids["scoped"] = r.headers["X-Bench-Request-Id"]
        print(f"     -> HTTP {r.status_code} inside scope {scope_id}")

        print("\n[5] 50 concurrent requests with distinct session/operation tags")
        specs = [{"operation": ["write", "read", "answer", "judge"][i % 4], "session_id": f"conc-{i}", "seed": i} for i in range(50)]

        async def one(spec):
            rr = await raw.post(f"{proxy_url}/v1/chat/completions", json={"model": chat_model, "messages": [{"role": "user", "content": f"Reply with the number {spec['seed']}."}], "max_tokens": 3}, headers={**auth, **pc.headers(**spec)})
            return rr.headers["X-Bench-Request-Id"], spec, rr.status_code

        t0 = time.perf_counter()
        conc = await asyncio.gather(*(one(s) for s in specs))
        print(f"     -> {sum(1 for _, _, sc in conc if sc == 200)}/50 succeeded in {time.perf_counter() - t0:.2f}s")

        print("\n[6] intentionally failing request")
        rf = await raw.post(f"{proxy_url}/v1/chat/completions", json={"model": fail_model, "messages": [{"role": "user", "content": "x"}]}, headers={**auth, **pc.headers(operation="write", session_id="sess-3")})
        ids["fail"] = rf.headers["X-Bench-Request-Id"]
        print(f"     -> HTTP {rf.status_code}: {rf.text[:120]}")

    await pc.aclose()
    await oai.close()

    print("\n=== verifying the append-only trace:", log_path)
    evs = list(read_events(log_path))
    calls = {ev["request_id"]: ev for ev in evs if ev["event_type"] == "MODEL_CALL"}
    types = [ev["event_type"] for ev in evs]
    check(types[0] == "PROXY_START", "first event is PROXY_START", failures)
    check([ev["seq"] for ev in evs] == list(range(1, len(evs) + 1)), f"seq is contiguous 1..{len(evs)}", failures)

    c = calls[ids["chat"]]
    check(c["prompt_tokens"] == expected_chat_usage["prompt_tokens"] and c["completion_tokens"] == expected_chat_usage["completion_tokens"] and c["total_tokens"] == expected_chat_usage["total_tokens"], f"chat tokens recorded = upstream usage {expected_chat_usage}", failures)
    check(c["usage_source"] == "upstream", "chat usage_source == upstream", failures)
    check((c["system"], c["configuration"], c["seed"], c["session_id"], c["operation"], c["run_id"]) == ("mem0", "demo-cfg", 42, "sess-1", "write", run_id), "chat attribution (system/config/seed/session/operation/run) correct", failures)
    check(c["attribution_sources"]["operation"] == "header" and c["attribution_sources"]["system"] == "api_key_token", "chat attribution sources recorded (header + api_key_token)", failures)
    check(c["status"] == "success" and c["http_status"] == 200 and c["duration_ms"] > 0 and c["upstream_duration_ms"] > 0, "chat latency + status recorded", failures)

    em = calls[ids["embed"]]
    check(em["endpoint"] == "embeddings" and em["prompt_tokens"] == expected_embed_usage["prompt_tokens"], f"embedding tokens recorded = upstream usage {expected_embed_usage}", failures)
    check(em["embedding_inputs"] == 2 and em["embedding_vectors"] == 2 and em["embedding_dimensions"] == len(e.data[0].embedding), "embedding input count / vectors / dims recorded", failures)
    check(em["operation"] == "embed", "embedding operation tag", failures)

    st = [ev for ev in calls.values() if ev["stream"]]
    check(len(st) == 1, "exactly one streaming call logged", failures)
    if st:
        s = st[0]
        check(s["usage_source"] == "upstream" and s["total_tokens"] and s["total_tokens"] > 0, f"streaming usage captured from upstream: {s['prompt_tokens']}+{s['completion_tokens']}={s['total_tokens']}", failures)
        check(s["request_modifications"] == ["stream_options.include_usage=true"] and not saw_usage_chunk, "usage injection recorded and hidden from client", failures)
        check(s["time_to_first_byte_ms"] is not None, "streaming time_to_first_byte_ms recorded", failures)

    sc = calls[ids["scoped"]]
    check(sc["operation"] == "consolidate" and sc["session_id"] == "sess-2" and sc["scope_state"] == "single" and sc["attribution_sources"]["operation"] == "scope", "scope-based attribution applied", failures)
    check(types.count("SCOPE_ENTER") == 1 and types.count("SCOPE_EXIT") == 1, "SCOPE_ENTER/SCOPE_EXIT events present", failures)

    ok = 0
    for rid, spec, _ in conc:
        ev = calls.get(rid)
        if ev and ev["operation"] == spec["operation"] and ev["session_id"] == spec["session_id"] and ev["seed"] == spec["seed"] and ev["status"] == "success":
            ok += 1
    check(ok == 50, f"concurrent attribution isolated: {ok}/50 events match their request", failures)

    f = calls[ids["fail"]]
    check(f["status"] == "error" and f["http_status"] == rf.status_code and f["error_type"] in ("upstream_error", "upstream_unreachable"), f"failed request logged: status=error http={f['http_status']} type={f['error_type']}", failures)
    check(f["operation"] == "write" and f["session_id"] == "sess-3", "failed request keeps its attribution", failures)
    check(len(calls) == 55, f"exactly one MODEL_CALL per request: {len(calls)} == 55", failures)

    print("\n=== example trace lines (PROXY_START, chat, embed, stream, scoped, failure) ===")
    for ev in [evs[0], c, em, st[0] if st else None, sc, f]:
        if ev:
            print(json.dumps(ev, ensure_ascii=False))

    print(f"\n{len(failures)} verification failure(s)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", help="real upstream base URL; omit to use the local fake upstream")
    ap.add_argument("--upstream-api-key", default=os.environ.get("PROXY_UPSTREAM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--chat-model", default=None)
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--fail-model", default=None, help="model name that makes the upstream fail (default: fail-500 fake / a nonexistent model for real providers)")
    ap.add_argument("--proxy-port", type=int, default=8811)
    ap.add_argument("--fake-port", type=int, default=8899)
    ap.add_argument("--log", default=str(ROOT / "results/raw/proxy/demo-milestone1.jsonl"))
    args = ap.parse_args()

    real = bool(args.upstream)
    chat_model = args.chat_model or ("gpt-4o-mini" if real else "fake-model")
    embed_model = args.embed_model or ("text-embedding-3-small" if real else "fake-embed")
    fail_model = args.fail_model or ("this-model-does-not-exist" if real else "fail-500")
    upstream = args.upstream or f"http://127.0.0.1:{args.fake_port}/v1"
    log_path = Path(args.log)
    if log_path.exists():
        log_path.unlink()  # fresh demo trace (the proxy itself never truncates)

    procs: list[subprocess.Popen] = []
    env = {**os.environ, "PROXY_LOG_PATH": str(log_path), "PROXY_UPSTREAM_BASE_URL": upstream}
    if args.upstream_api_key:
        env["PROXY_UPSTREAM_API_KEY"] = args.upstream_api_key
    try:
        if not real:
            procs.append(subprocess.Popen([PY, "-m", "proxy.fake_upstream", "--port", str(args.fake_port)], cwd=ROOT))
            wait_healthy(f"http://127.0.0.1:{args.fake_port}/v1/models")
        procs.append(subprocess.Popen([PY, "-m", "proxy", "--port", str(args.proxy_port), "--log-level", "warning"], cwd=ROOT, env=env))
        proxy_url = f"http://127.0.0.1:{args.proxy_port}"
        wait_healthy(f"{proxy_url}/healthz")
        print(f"proxy up at {proxy_url}; upstream={upstream}; log={log_path}")
        return asyncio.run(run_demo(proxy_url, chat_model, embed_model, fail_model, log_path, real))
    finally:
        for p in reversed(procs):
            p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        evs = list(read_events(log_path)) if log_path.exists() else []
        print("last event:", evs[-1]["event_type"] if evs else None)


if __name__ == "__main__":
    sys.exit(main())
