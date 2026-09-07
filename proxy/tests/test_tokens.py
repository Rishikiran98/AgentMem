"""Token counts are recorded verbatim from the upstream ``usage`` block."""
from __future__ import annotations

import json

import pytest

from proxy.tokenization import count_chat_prompt_tokens, tokenizer_status
from proxy.tests.conftest import chat_body, events


async def test_chat_usage_recorded(client, log_path):
    text = "one two three four five six seven"  # 7 words -> 7 prompt tokens under the fake rule
    r = await client.post("/v1/chat/completions", json=chat_body(text, max_tokens=5), headers={"X-Bench-System": "mem0", "X-Bench-Operation": "write"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["usage"] == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    (ev,) = events(log_path)
    assert ev["request_id"] == r.headers["X-Bench-Request-Id"]
    assert (ev["prompt_tokens"], ev["completion_tokens"], ev["total_tokens"]) == (7, 5, 12)
    assert ev["usage_source"] == "upstream"
    assert ev["endpoint"] == "chat" and ev["status"] == "success" and ev["http_status"] == 200
    assert ev["model"] == "fake-model" and ev["upstream_model"] == "fake-model"
    assert ev["finish_reason"] == "stop"
    assert ev["stream"] is False
    assert ev["duration_ms"] >= 0 and ev["upstream_duration_ms"] is not None
    assert ev["upstream_request_id"] == r.headers["X-Upstream-Request-Id"]
    assert ev["request_bytes"] > 0 and ev["response_bytes"] == len(r.content)


async def test_embedding_usage_recorded(client, log_path):
    body = {"model": "fake-embed", "input": ["alpha beta", "gamma delta epsilon", "zeta"]}  # 6 words
    r = await client.post("/v1/embeddings", json=body, headers={"X-Bench-System": "mem0", "X-Bench-Operation": "embed"})
    assert r.status_code == 200, r.text
    (ev,) = events(log_path)
    assert ev["endpoint"] == "embeddings"
    assert ev["prompt_tokens"] == 6 and ev["total_tokens"] == 6 and ev["completion_tokens"] is None
    assert ev["embedding_inputs"] == 3
    assert ev["embedding_vectors"] == 3
    from proxy.fake_upstream import EMBED_DIM
    assert ev["embedding_dimensions"] == EMBED_DIM
    assert ev["operation"] == "embed"


async def test_embedding_single_string_input(client, log_path):
    r = await client.post("/v1/embeddings", json={"model": "fake-embed", "input": "hello world"}, headers={"X-Bench-System": "mem0", "X-Bench-Operation": "read"})
    assert r.status_code == 200
    (ev,) = events(log_path)
    assert ev["embedding_inputs"] == 1 and ev["embedding_vectors"] == 1 and ev["prompt_tokens"] == 2


async def test_stream_usage_captured_and_hidden_when_not_requested(client, log_path):
    """Client did not ask for usage; proxy injects it upstream, records it, strips it from the client stream."""
    r = await client.post("/v1/chat/completions", json=chat_body("a b c", stream=True, max_tokens=4), headers={"X-Bench-System": "mem0", "X-Bench-Operation": "read"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = [json.loads(l[5:]) for l in r.text.split("\n") if l.startswith("data:") and l[5:].strip() != "[DONE]"]
    assert all(c["choices"] for c in chunks), "usage-only chunk must be stripped when the client did not request it"
    assert r.text.strip().endswith("data: [DONE]")
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == "w1 w2 w3 w4"
    (ev,) = events(log_path)
    assert ev["stream"] is True
    assert (ev["prompt_tokens"], ev["completion_tokens"], ev["total_tokens"]) == (3, 4, 7)
    assert ev["usage_source"] == "upstream"
    assert ev["request_modifications"] == ["stream_options.include_usage=true"]
    assert ev["finish_reason"] == "stop"
    assert ev["status"] == "success" and ev["http_status"] == 200


async def test_stream_usage_forwarded_when_requested(client, log_path):
    body = chat_body("a b c", stream=True, max_tokens=2, stream_options={"include_usage": True})
    r = await client.post("/v1/chat/completions", json=body, headers={"X-Bench-System": "mem0", "X-Bench-Operation": "read"})
    chunks = [json.loads(l[5:]) for l in r.text.split("\n") if l.startswith("data:") and l[5:].strip() != "[DONE]"]
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert len(usage_chunks) == 1 and usage_chunks[0]["usage"]["total_tokens"] == 5
    (ev,) = events(log_path)
    assert ev["request_modifications"] == []
    assert ev["total_tokens"] == 5


async def test_models_endpoint_logged(client, log_path):
    r = await client.get("/v1/models", headers={"X-Bench-System": "letta"})
    assert r.status_code == 200 and r.json()["object"] == "list"
    (ev,) = events(log_path)
    assert ev["endpoint"] == "models" and ev["operation"] == "meta" and ev["status"] == "success"
    assert ev["prompt_tokens"] is None and ev["usage_source"] == "none"


@pytest.mark.skipif(not tokenizer_status()["local_estimates_available"], reason="tiktoken encodings not available offline")
async def test_local_estimate_recorded_when_available(client, log_path):
    msgs = [{"role": "user", "content": "the quick brown fox"}]
    r = await client.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": msgs}, headers={"X-Bench-System": "mem0", "X-Bench-Operation": "write"})
    assert r.status_code == 200
    (ev,) = events(log_path)
    assert ev["prompt_tokens_local"] == count_chat_prompt_tokens(msgs, "gpt-4o-mini")
    assert ev["completion_tokens_local"] is not None


async def test_local_estimate_absent_is_null_not_zero(client, log_path):
    r = await client.post("/v1/chat/completions", json=chat_body("x y"), headers={"X-Bench-System": "mem0", "X-Bench-Operation": "write"})
    assert r.status_code == 200
    (ev,) = events(log_path)
    if not tokenizer_status()["local_estimates_available"]:
        assert ev["prompt_tokens_local"] is None and ev["completion_tokens_local"] is None
    # upstream numbers are never affected by local availability
    assert ev["prompt_tokens"] == 2


async def test_embedding_dimensions_from_base64(settings, log_path):
    """The openai SDK requests encoding_format=base64 by default; dims must still be recorded."""
    import base64
    import struct

    import httpx

    from proxy.app import create_app

    vec = [0.5, -0.25, 1.0, 0.0]
    b64 = base64.b64encode(struct.pack("<4f", *vec)).decode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": b64}], "model": "m", "usage": {"prompt_tokens": 3, "total_tokens": 3}})

    app = create_app(settings, upstream_transport=httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
        r = await c.post("/v1/embeddings", json={"model": "m", "input": "a b c", "encoding_format": "base64"}, headers={"X-Bench-System": "mem0", "X-Bench-Operation": "read"})
        assert r.status_code == 200
    (ev,) = events(log_path)
    assert ev["embedding_dimensions"] == 4 and ev["embedding_vectors"] == 1 and ev["prompt_tokens"] == 3


async def test_proxy_overhead_is_bounded(client, log_path):
    """duration_ms must not include tokenizer or log I/O work; overhead over upstream stays small."""
    for _ in range(20):
        r = await client.post("/v1/chat/completions", json=chat_body("a b c d e", model="never-seen-model-name"), headers={"X-Bench-System": "mem0", "X-Bench-Operation": "write"})
        assert r.status_code == 200
    evs = events(log_path)
    overhead = [e["duration_ms"] - e["upstream_duration_ms"] for e in evs]
    assert max(overhead) < 50, overhead


def test_tokenizer_never_retries_network_after_failure():
    from proxy import tokenization as tk

    status = tk.tokenizer_status()
    if status["local_estimates_available"]:
        return  # online host: nothing to assert
    assert status["disabled"] is True
    # A brand-new model name must short-circuit without touching the network.
    import time

    t0 = time.perf_counter()
    assert tk.count_text_tokens("hello", "some-model-" + str(time.time())) is None
    assert time.perf_counter() - t0 < 0.01


async def test_responses_api_usage_recorded(client, log_path):
    """Graphiti's default client uses POST /v1/responses; usage keys are normalised."""
    body = {"model": "fake-model", "input": [{"role": "system", "content": "a b"}, {"role": "user", "content": "c d e"}], "max_output_tokens": 50, "text": {"format": {"type": "json_schema", "name": "Weird", "schema": {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}}}}
    r = await client.post("/v1/responses", json=body, headers={"X-Bench-System": "zep", "X-Bench-Operation": "write"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["object"] == "response" and payload["usage"]["input_tokens"] == 5
    (ev,) = events(log_path)
    assert ev["endpoint"] == "responses" and ev["operation"] == "write" and ev["status"] == "success"
    assert ev["prompt_tokens"] == 5 and ev["completion_tokens"] == payload["usage"]["output_tokens"] and ev["total_tokens"] == ev["prompt_tokens"] + ev["completion_tokens"]
    assert ev["usage_source"] == "upstream" and ev["usage_details"]["input_tokens"] == 5
    assert ev["request_params"]["response_format"] == {"type": "json_schema", "name": "Weird"} and ev["request_params"]["max_tokens"] == 50
    assert ev["finish_reason"] == "completed" and ev["stream"] is False


async def test_responses_api_failure_logged(client, log_path):
    r = await client.post("/v1/responses", json={"model": "fail-429", "input": "x"}, headers={"X-Bench-System": "zep", "X-Bench-Operation": "write"})
    assert r.status_code == 429
    (ev,) = events(log_path)
    assert ev["endpoint"] == "responses" and ev["status"] == "error" and ev["http_status"] == 429
