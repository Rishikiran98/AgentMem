"""Deterministic mock OpenAI-compatible provider for tests and demos.

It is *not* a benchmark component; it exists so that the proxy's accounting can
be verified against numbers the test controls independently.

Token accounting rule (simple and reproducible):
    prompt_tokens     = number of whitespace-separated words across all message
                        contents (chat) or across all inputs (embeddings)
    completion_tokens = min(max_tokens or 16, 16) words of the form "w1 w2 ..."

Failure injection by model name:
    fail-500, fail-429, fail-400  -> OpenAI-style error JSON with that status
    fail-timeout                  -> sleeps 30 s before answering
    slow-<ms>[-*]                 -> sleeps <ms> milliseconds before answering
Streaming is supported (SSE, one word per chunk) with ``stream_options.include_usage``.

Mem0 emulation: when the system prompt is Mem0's additive extraction prompt
("You are a Memory Extractor"), the reply is the JSON Mem0 expects, with one
memory per user/assistant message taken verbatim from the "## New Messages"
section.  This exercises Mem0's real write pipeline without a real LLM.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

EMBED_DIM = 8


def word_count(text: Any) -> int:
    if isinstance(text, str):
        return len(text.split())
    if isinstance(text, list):
        return sum(word_count(p.get("text") if isinstance(p, dict) else p) for p in text)
    return 0


def chat_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(word_count(m.get("content")) for m in messages if isinstance(m, dict))


def embedding_prompt_tokens(inp: Any) -> int:
    if isinstance(inp, str):
        return word_count(inp)
    if isinstance(inp, list):
        if all(isinstance(x, int) for x in inp):
            return len(inp)
        return sum(embedding_prompt_tokens(x) for x in inp)
    return 0


def mem0_extraction_reply(messages: list[dict[str, Any]]) -> str | None:
    """Return Mem0-format extraction JSON if this looks like a Mem0 extraction call."""
    system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
    if not isinstance(system, str) or "Memory Extractor" not in system:
        return None
    user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    if not isinstance(user, str):
        return json.dumps({"memory": []})
    m = re.search(r"## New Messages\n(.*?)\n\n## ", user, flags=re.S)
    if not m:
        return json.dumps({"memory": []})
    section = m.group(1).strip()
    new_messages: list[dict[str, Any]] = []
    try:  # JSON form
        parsed = json.loads(section)
        if isinstance(parsed, list):
            new_messages = [x for x in parsed if isinstance(x, dict)]
    except json.JSONDecodeError:  # Mem0's parse_messages() form: "role: content" per line
        for line in section.split("\n"):
            role, sep, content = line.partition(": ")
            if sep and role in ("user", "assistant", "system"):
                new_messages.append({"role": role, "content": content})
    out = []
    for msg in new_messages:
        if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str) and msg["content"].strip():
            out.append({"id": str(len(out)), "text": msg["content"].strip(), "attributed_to": msg["role"]})
    return json.dumps({"memory": out})


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "fake_upstream_error", "param": None, "code": None}}, status_code=status)


async def _maybe_fail_or_delay(model: str | None):
    if not model:
        return None
    m = re.match(r"^slow-(\d+)", model)
    if m:
        await asyncio.sleep(int(m.group(1)) / 1000.0)
        return None
    if model == "fail-timeout":
        await asyncio.sleep(30)
        return None
    m = re.match(r"^fail-(\d{3})$", model)
    if m:
        return _error(int(m.group(1)), f"injected failure for model {model}")
    return None


def create_fake_upstream() -> FastAPI:
    app = FastAPI(title="fake OpenAI upstream")
    app.state.calls = 0

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "fake-model", "object": "model", "owned_by": "fake"}, {"id": "fake-embed", "object": "model", "owned_by": "fake"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        app.state.calls += 1
        body = json.loads(await request.body())
        model = body.get("model")
        failure = await _maybe_fail_or_delay(model)
        if failure is not None:
            return failure
        messages = body.get("messages") or []
        prompt_tokens = chat_prompt_tokens(messages)
        mem0_reply = mem0_extraction_reply(messages)
        if mem0_reply is not None:
            completion = mem0_reply
            words = [completion]
            n_words = word_count(completion)
        else:
            n_words = min(int(body.get("max_tokens") or 16), 16)
            words = [f"w{i + 1}" for i in range(n_words)]
            completion = " ".join(words)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n_words, "total_tokens": prompt_tokens + n_words}
        rid = "chatcmpl-" + uuid.uuid4().hex[:12]
        created = int(time.time())
        headers = {"x-request-id": "up-" + uuid.uuid4().hex[:8]}

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            async def gen() -> AsyncIterator[bytes]:
                def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
                    payload = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if include_usage:
                        payload["usage"] = None
                    return b"data: " + json.dumps(payload).encode() + b"\n\n"

                yield chunk({"role": "assistant", "content": ""})
                for i, w in enumerate(words):
                    await asyncio.sleep(0.001)
                    yield chunk({"content": (" " if i else "") + w})
                yield chunk({}, "stop")
                if include_usage:
                    yield b"data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage}).encode() + b"\n\n"
                yield b"data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

        return JSONResponse(
            {
                "id": rid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": completion}, "finish_reason": "stop"}],
                "usage": usage,
            },
            headers=headers,
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        app.state.calls += 1
        body = json.loads(await request.body())
        model = body.get("model")
        failure = await _maybe_fail_or_delay(model)
        if failure is not None:
            return failure
        inp = body.get("input")
        items: list[Any] = [inp] if isinstance(inp, str) or (isinstance(inp, list) and inp and all(isinstance(x, int) for x in inp)) else list(inp or [])
        data = []
        for i, item in enumerate(items):
            digest = hashlib.sha256(json.dumps(item).encode()).digest()
            vec = [((digest[j] / 255.0) * 2 - 1) for j in range(EMBED_DIM)]
            data.append({"object": "embedding", "index": i, "embedding": vec})
        pt = embedding_prompt_tokens(inp)
        return JSONResponse({"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": pt, "total_tokens": pt}}, headers={"x-request-id": "up-" + uuid.uuid4().hex[:8]})

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m proxy.fake_upstream")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args(argv)
    uvicorn.run(create_fake_upstream(), host=args.host, port=args.port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
