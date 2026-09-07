"""Failures of every kind are logged as MODEL_CALL events with status=error."""
from __future__ import annotations

import httpx
import pytest

from proxy.app import create_app
from proxy.tests.conftest import chat_body, events

TAGS = {"X-Bench-System": "mem0", "X-Bench-Operation": "write", "X-Bench-Session": "s-err"}


async def test_upstream_500_is_passed_through_and_logged(client, log_path):
    r = await client.post("/v1/chat/completions", json=chat_body("a b c", model="fail-500"), headers=TAGS)
    assert r.status_code == 500
    assert r.json()["error"]["message"].startswith("injected failure")
    (ev,) = events(log_path)
    assert ev["status"] == "error" and ev["http_status"] == 500 and ev["upstream_http_status"] == 500
    assert ev["error_type"] == "upstream_error" and "injected failure" in ev["error_message"]
    assert ev["prompt_tokens"] is None and ev["usage_source"] == "none"
    assert ev["system"] == "mem0" and ev["operation"] == "write" and ev["session_id"] == "s-err"
    assert ev["request_id"] == r.headers["X-Bench-Request-Id"]


async def test_upstream_429_logged(client, log_path):
    r = await client.post("/v1/embeddings", json={"model": "fail-429", "input": "x"}, headers=TAGS)
    assert r.status_code == 429
    (ev,) = events(log_path)
    assert ev["status"] == "error" and ev["http_status"] == 429 and ev["endpoint"] == "embeddings"


async def test_streaming_upstream_error_logged(client, log_path):
    r = await client.post("/v1/chat/completions", json=chat_body("a", model="fail-400", stream=True), headers=TAGS)
    assert r.status_code == 400
    (ev,) = events(log_path)
    assert ev["status"] == "error" and ev["stream"] is True and ev["http_status"] == 400 and ev["error_type"] == "upstream_error"


@pytest.mark.parametrize(
    "exc, expected_status, expected_type",
    [
        (httpx.ConnectError("connection refused"), 502, "upstream_unreachable"),
        (httpx.ReadTimeout("read timed out"), 504, "upstream_timeout"),
        (httpx.RemoteProtocolError("server disconnected"), 502, "upstream_unreachable"),
    ],
)
async def test_transport_failures_logged(settings, log_path, exc, expected_status, expected_type):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    app = create_app(settings, upstream_transport=httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
        r = await c.post("/v1/chat/completions", json=chat_body(), headers=TAGS)
        assert r.status_code == expected_status
        assert r.json()["error"]["type"] == expected_type
        rs = await c.post("/v1/chat/completions", json=chat_body(stream=True), headers=TAGS)
        assert rs.status_code == expected_status
    evs = events(log_path)
    assert len(evs) == 2
    for ev in evs:
        assert ev["status"] == "error" and ev["http_status"] == expected_status and ev["error_type"] == expected_type
        assert ev["upstream_http_status"] is None
        assert ev["duration_ms"] is not None and ev["upstream_duration_ms"] is not None
        assert ev["system"] == "mem0"


async def test_invalid_json_body_logged(client, log_path):
    r = await client.post("/v1/chat/completions", content=b"{not json", headers={**TAGS, "content-type": "application/json"})
    assert r.status_code == 400
    (ev,) = events(log_path)
    assert ev["status"] == "error" and ev["error_type"] == "invalid_request_body" and ev["http_status"] == 400


async def test_unsupported_endpoint_logged(client, log_path):
    r = await client.post("/v1/completions", json={"model": "x", "prompt": "y"}, headers=TAGS)
    assert r.status_code == 404
    (ev,) = events(log_path)
    assert ev["status"] == "error" and ev["error_type"] == "unsupported_endpoint" and ev["endpoint"] == "unsupported" and ev["path"] == "/v1/completions"


async def test_every_request_produces_exactly_one_event(client, log_path):
    """Mixed success/failure burst: one MODEL_CALL per request, no more, no fewer."""
    import asyncio

    reqs = []
    for i in range(40):
        model = ["fake-model", "fail-500", "slow-5", "fail-429"][i % 4]
        reqs.append(client.post("/v1/chat/completions", json=chat_body("a b", model=model, stream=(i % 5 == 0)), headers=TAGS))
    rs = await asyncio.gather(*reqs)
    ids = sorted(r.headers["X-Bench-Request-Id"] for r in rs)
    evs = events(log_path)
    assert sorted(e["request_id"] for e in evs) == ids
    assert sum(e["status"] == "error" for e in evs) == 20
    assert sum(e["status"] == "success" for e in evs) == 20
