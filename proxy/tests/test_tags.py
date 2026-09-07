"""Attribution tags propagate through headers, API-key/path tokens and scopes."""
from __future__ import annotations

import pytest

from proxy.client import ProxyClient
from proxy.tags import ScopeRegistry, TagError, api_key_for, decode_tags, encode_tags, normalize_tags, split_path_token, token_from_api_key
from proxy.tests.conftest import chat_body, events

FULL = {"system": "mem0", "configuration": "cfg-1", "seed": 42, "session_id": "s-1", "operation": "write", "run_id": "run-1", "client_id": "mem0-a"}


def test_token_roundtrip():
    tok = encode_tags(FULL)
    assert decode_tags(tok) == FULL
    assert token_from_api_key("Bearer " + api_key_for(FULL)) == tok
    assert token_from_api_key("Bearer sk-real-key") is None
    assert split_path_token(f"/b/{tok}/v1/chat/completions") == (tok, "/v1/chat/completions")
    assert split_path_token("/v1/chat/completions") == (None, "/v1/chat/completions")


def test_normalize_rejects_bad_values():
    with pytest.raises(TagError):
        normalize_tags({"operation": "frobnicate"})
    with pytest.raises(TagError):
        normalize_tags({"seed": "not-int"})
    with pytest.raises(TagError):
        normalize_tags({"bogus": "x"})
    with pytest.raises(TagError):
        normalize_tags({"system": "other"}, allowed_systems=["mem0"])
    assert normalize_tags({"seed": "7", "session_id": None}) == {"seed": 7}


def test_scope_registry_merge():
    reg = ScopeRegistry()
    reg.enter("c", "a", {"operation": "write", "session_id": "s1"}, "t")
    assert reg.resolve("c")[0] == {"operation": "write", "session_id": "s1"}
    reg.enter("c", "b", {"operation": "write", "session_id": "s2"}, "t")
    tags, ids, state, amb = reg.resolve("c")
    assert state == "ambiguous" and amb == ["session_id"] and tags == {"operation": "write"} and set(ids) == {"a", "b"}
    reg.exit("c", "a")
    assert reg.resolve("c")[2] == "single"
    reg.exit("c", "b")
    assert reg.resolve("c") == ({}, [], "none", [])
    with pytest.raises(TagError):
        reg.exit("c", "b")


async def test_header_attribution(client, log_path):
    headers = {"X-Bench-System": "zep", "X-Bench-Configuration": "cfg-z", "X-Bench-Seed": "7", "X-Bench-Session": "sess-9", "X-Bench-Operation": "read", "X-Bench-Run": "run-z"}
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=headers)
    assert r.status_code == 200
    (ev,) = events(log_path)
    assert (ev["system"], ev["configuration"], ev["seed"], ev["session_id"], ev["operation"], ev["run_id"]) == ("zep", "cfg-z", 7, "sess-9", "read", "run-z")
    assert ev["client_id"] == "zep" and ev["attribution_sources"]["client_id"] == "derived_from_system"
    assert all(ev["attribution_sources"][k] == "header" for k in ("system", "configuration", "seed", "session_id", "operation", "run_id"))
    assert ev["unattributed"] is False and ev["scope_state"] == "none"


async def test_api_key_token_attribution(client, log_path):
    key = api_key_for(FULL)
    r = await client.post("/v1/chat/completions", json=chat_body(), headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    (ev,) = events(log_path)
    for k, v in FULL.items():
        assert ev[k] == v, k
        assert ev["attribution_sources"][k] == "api_key_token"


async def test_path_token_attribution(client, log_path):
    tok = encode_tags(FULL)
    r = await client.post(f"/b/{tok}/v1/embeddings", json={"model": "fake-embed", "input": "a b"}, headers={"Authorization": "Bearer sk-whatever"})
    assert r.status_code == 200
    (ev,) = events(log_path)
    assert ev["path"] == "/v1/embeddings" and ev["endpoint"] == "embeddings"
    assert ev["system"] == "mem0" and ev["attribution_sources"]["system"] == "path_token"


async def test_scope_attribution_and_precedence(client, log_path, proxy_app):
    pc = ProxyClient("http://proxy", http=client)
    key = pc.api_key(system="letta", configuration="cfg-l", seed=1, client_id="letta-main")
    auth = {"Authorization": f"Bearer {key}"}

    # No scope -> only static tags.
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth)
    assert r.status_code == 200
    # Scope supplies operation/session.
    async with pc.scope("letta-main", operation="write", session_id="s-77", run_id="run-l") as scope_id:
        r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth)
        assert r.status_code == 200
        # Header beats scope.
        r = await client.post("/v1/chat/completions", json=chat_body(), headers={**auth, "X-Bench-Operation": "consolidate"})
        assert r.status_code == 200
    # Scope exited -> static again.
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth)
    assert r.status_code == 200

    evs = events(log_path)
    assert len(evs) == 4
    assert evs[0]["operation"] is None and evs[0]["session_id"] is None and evs[0]["scope_state"] == "none" and evs[0]["system"] == "letta"
    assert evs[1]["operation"] == "write" and evs[1]["session_id"] == "s-77" and evs[1]["run_id"] == "run-l"
    assert evs[1]["scope_ids"] == [scope_id] and evs[1]["scope_state"] == "single"
    assert evs[1]["attribution_sources"]["operation"] == "scope" and evs[1]["attribution_sources"]["system"] == "api_key_token"
    assert evs[2]["operation"] == "consolidate" and evs[2]["attribution_sources"]["operation"] == "header" and evs[2]["session_id"] == "s-77"
    assert evs[3]["operation"] is None and evs[3]["scope_state"] == "none"

    scope_events = events(log_path, None)
    kinds = [e["event_type"] for e in scope_events]
    assert kinds.count("SCOPE_ENTER") == 1 and kinds.count("SCOPE_EXIT") == 1
    assert kinds.index("SCOPE_ENTER") < kinds.index("SCOPE_EXIT")


async def test_ambiguous_scopes_are_flagged_not_guessed(client, log_path):
    pc = ProxyClient("http://proxy", http=client)
    auth = {"Authorization": f"Bearer {pc.api_key(system='zep', client_id='zep')}"}
    a = await pc.enter_scope("zep", operation="write", session_id="s-A")
    b = await pc.enter_scope("zep", operation="write", session_id="s-B")
    r = await client.post("/v1/chat/completions", json=chat_body(), headers=auth)
    assert r.status_code == 200
    await pc.exit_scope("zep", a)
    await pc.exit_scope("zep", b)
    (ev,) = events(log_path)
    assert ev["operation"] == "write"
    assert ev["session_id"] is None and ev["ambiguous_fields"] == ["session_id"] and ev["scope_state"] == "ambiguous"
    assert set(ev["scope_ids"]) == {a, b}


async def test_invalid_operation_rejected_and_logged(client, log_path):
    r = await client.post("/v1/chat/completions", json=chat_body(), headers={"X-Bench-System": "mem0", "X-Bench-Operation": "nope"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_attribution"
    (ev,) = events(log_path)
    assert ev["status"] == "error" and ev["error_type"] == "invalid_attribution" and ev["http_status"] == 400


async def test_unknown_system_rejected(client, log_path):
    r = await client.post("/v1/chat/completions", json=chat_body(), headers={"X-Bench-System": "notasystem"})
    assert r.status_code == 400
    (ev,) = events(log_path)
    assert ev["error_type"] == "invalid_attribution"


async def test_unattributed_request_flagged(client, log_path):
    r = await client.post("/v1/chat/completions", json=chat_body(), headers={"Authorization": "Bearer sk-plain"})
    assert r.status_code == 200
    (ev,) = events(log_path)
    assert ev["unattributed"] is True and ev["system"] is None


async def test_require_attribution_mode(settings, fake_upstream, log_path):
    import httpx

    from proxy.app import create_app

    settings.require_attribution = True
    app = create_app(settings, upstream_transport=httpx.ASGITransport(app=fake_upstream))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
        r = await c.post("/v1/chat/completions", json=chat_body())
        assert r.status_code == 400 and r.json()["error"]["type"] == "unattributed_request"
        r = await c.post("/v1/chat/completions", json=chat_body(), headers={"X-Bench-System": "mem0", "X-Bench-Operation": "write"})
        assert r.status_code == 200
    evs = events(log_path)
    assert [e["status"] for e in evs] == ["error", "success"]
    assert evs[0]["error_type"] == "unattributed_request"


def test_client_headers_helper():
    h = ProxyClient.headers(system="mem0", operation="judge", seed=3, session_id="s")
    assert h == {"x-bench-system": "mem0", "x-bench-operation": "judge", "x-bench-seed": "3", "x-bench-session": "s"}
