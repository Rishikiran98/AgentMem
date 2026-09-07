"""Concurrent requests never corrupt each other's attribution metadata."""
from __future__ import annotations

import asyncio
import random

from proxy.client import ProxyClient
from proxy.tests.conftest import chat_body, events


async def test_concurrent_header_attribution_is_isolated(client, log_path):
    rng = random.Random(1234)
    n = 300
    specs = []
    for i in range(n):
        specs.append(
            {
                "system": rng.choice(["mem0", "zep", "letta", "hieromem"]),
                "operation": rng.choice(["write", "read", "embed", "answer", "judge", "consolidate"]),
                "session_id": f"sess-{i}",
                "seed": rng.randint(0, 10_000),
                "delay_ms": rng.randint(0, 40),
                "words": rng.randint(1, 30),
            }
        )

    async def one(i: int, spec: dict) -> tuple[str, dict]:
        headers = {"X-Bench-System": spec["system"], "X-Bench-Operation": spec["operation"], "X-Bench-Session": spec["session_id"], "X-Bench-Seed": str(spec["seed"])}
        text = " ".join(["tok"] * spec["words"])
        if spec["operation"] == "embed":
            r = await client.post("/v1/embeddings", json={"model": f"slow-{spec['delay_ms']}-embed", "input": text}, headers=headers)
        else:
            r = await client.post("/v1/chat/completions", json=chat_body(text, model=f"slow-{spec['delay_ms']}", max_tokens=3, stream=(i % 3 == 0)), headers=headers)
        assert r.status_code == 200, r.text
        return r.headers["X-Bench-Request-Id"], spec

    results = await asyncio.gather(*(one(i, s) for i, s in enumerate(specs)))
    by_id = {rid: spec for rid, spec in results}
    evs = {e["request_id"]: e for e in events(log_path)}
    assert len(evs) == n and set(evs) == set(by_id)
    for rid, spec in by_id.items():
        ev = evs[rid]
        assert ev["system"] == spec["system"], rid
        assert ev["operation"] == spec["operation"], rid
        assert ev["session_id"] == spec["session_id"], rid
        assert ev["seed"] == spec["seed"], rid
        assert ev["prompt_tokens"] == spec["words"], rid
        assert ev["status"] == "success"
        if spec["operation"] != "embed":
            assert ev["completion_tokens"] == 3 and ev["total_tokens"] == spec["words"] + 3
    # Sequence numbers are strictly increasing and gap-free across the whole file.
    seqs = [e["seq"] for e in events(log_path, None)]
    assert seqs == list(range(1, len(seqs) + 1))


async def test_concurrent_scopes_for_different_clients_are_isolated(client, log_path):
    pc = ProxyClient("http://proxy", http=client)
    clients = {cid: {"Authorization": f"Bearer {pc.api_key(system=sysname, client_id=cid)}"} for cid, sysname in [("mem0-a", "mem0"), ("zep-a", "zep"), ("letta-a", "letta")]}

    async def worker(cid: str, k: int) -> list[tuple[str, str, str]]:
        out = []
        for j in range(k):
            op = "write" if j % 2 == 0 else "read"
            sess = f"{cid}-s{j}"
            async with pc.scope(cid, operation=op, session_id=sess):
                # Several model calls happen inside a single harness operation,
                # possibly in parallel (as Mem0 does for extraction + embeddings).
                rs = await asyncio.gather(*(client.post("/v1/chat/completions", json=chat_body("a b", model=f"slow-{(j * 7 + m) % 15}"), headers=clients[cid]) for m in range(3)))
                for r in rs:
                    assert r.status_code == 200
                    out.append((r.headers["X-Bench-Request-Id"], op, sess))
        return out

    all_results = await asyncio.gather(*(worker(cid, 8) for cid in clients))
    expected = {rid: (op, sess) for group in all_results for rid, op, sess in group}
    evs = {e["request_id"]: e for e in events(log_path)}
    assert set(evs) == set(expected)
    for rid, (op, sess) in expected.items():
        assert evs[rid]["operation"] == op and evs[rid]["session_id"] == sess, rid
        assert evs[rid]["scope_state"] == "single"
        assert evs[rid]["client_id"] == sess.rsplit("-s", 1)[0]
