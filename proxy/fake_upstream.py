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

EMBED_DIM = 64
_STOP = {"the", "a", "an", "is", "my", "what", "of", "to", "in", "on", "and", "i", "me", "it", "do", "you", "your", "this", "that", "was", "are", "for", "by", "way"}


def fake_embedding(text: Any) -> list[float]:
    """Deterministic bag-of-words hashed embedding (unit norm).

    Texts sharing content words get correlated vectors, so a query like "What is
    my favorite color?" retrieves "my favorite color is teal" ahead of unrelated
    memories.  Purely a test device: no semantic model is involved.
    """
    words = [w.strip(".,!?;:\"'()[]").lower() for w in str(text).split()]
    words = [w for w in words if w and w not in _STOP]
    vec = [0.0] * EMBED_DIM
    for w in words or ["<empty>"]:
        d = hashlib.sha256(w.encode()).digest()
        for j in range(EMBED_DIM):
            vec[j] += ((d[j % len(d)] / 255.0) * 2 - 1)
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _section(text: str, start: str, end: str) -> str | None:
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(end, i + len(start))
    return text[i + len(start): j if j >= 0 else None]


def _words(text: str) -> set[str]:
    return {w.strip(".,!?;:\"'()[]").lower() for w in text.split()} - _STOP - {""}


def longmemeval_reader_reply(messages: list[dict[str, Any]]) -> str | None:
    """Emulate the reader: answer with the context line that best overlaps the question."""
    user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    if not isinstance(user, str) or "\nQuestion: " not in user:
        return None
    mem0_style = "Memories (sorted newest-first, grouped by date):" in user
    if mem0_style:
        history = (_section(user, "Memories (sorted newest-first, grouped by date):\n", "\n\nToday's Date:") or "").strip()
        question = user.rsplit("\nQuestion: ", 1)[1].split("\n", 1)[0].strip()
        lines = [l[2:] if l.startswith("- ") else l for l in history.split("\n") if l.strip() and not l.startswith("---") and l != "(No relevant memories found)"]
        qw = _words(question)
        best = max(lines, key=lambda l: (len(_words(l) & qw), -len(l))) if lines else None
        final = best if best and (_words(best) & qw) else "The information provided is not enough"
        return f"<mem_thinking>scanned {len(lines)} memories</mem_thinking>\nANSWER: {final}"
    if "History Chats:" not in user:
        return None
    history = (_section(user, "History Chats:\n\n", "\n\nCurrent Date:") or "").strip()
    question = user.rsplit("\nQuestion: ", 1)[1].split("\nAnswer", 1)[0].strip()
    lines = [l for l in history.split("\n") if l.strip()]
    if not lines:
        return "I don't have any information about that in our previous conversations."
    qw = _words(question)
    best = max(lines, key=lambda l: (len(_words(l) & qw), -len(l)))
    if not (_words(best) & qw):
        return "I don't have any information about that in our previous conversations."
    return best


def longmemeval_judge_reply(messages: list[dict[str, Any]]) -> str | None:
    """Emulate the judge: yes iff every content word of the gold answer appears in the response."""
    user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    if not isinstance(user, str) or "Model Response: " not in user:
        return None
    response = (user.rsplit("Model Response: ", 1)[1].split("\n\n", 1)[0]).lower()
    wrap = "judge_thinking" in user  # Mem0's unified judge asks for <judge_thinking> then a bare yes/no line

    def verdict(v: str) -> str:
        return f"<judge_thinking>checked</judge_thinking>\n{v}" if wrap else v

    if user.startswith("I will give you an unanswerable question"):
        return verdict("yes" if ("don't have" in response or "no information" in response or "not mentioned" in response) else "no")
    for key in ("Correct Answer: ", "Rubric: "):
        gold = _section(user, key, "\n\nModel Response: ")
        if gold is not None:
            gw = _words(gold)
            if wrap and "not enough" in response and ("never mentioned" in gold.lower() or "not enough" in gold.lower()):
                return verdict("yes")
            return verdict("yes" if gw and gw <= _words(response) else "no")
    return verdict("no")


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


def _tag(text: str, tag: str) -> str | None:
    """Return the body of <TAG>...</TAG> (Graphiti prompt sections)."""
    m = re.search(r"<" + re.escape(tag) + r">\s*(.*?)\s*</" + re.escape(tag) + r">", text, flags=re.S)
    return m.group(1) if m else None


def _entities_from_message(text: str) -> list[str]:
    names: list[str] = []
    role, sep, rest = text.partition(":")
    if sep and role.strip().lower() in ("user", "assistant", "system"):
        names.append(role.strip().lower())
    else:
        rest = text
    for tok in rest.split():
        clean = tok.strip(".,!?;:\"'()[]")
        if "memharness-canary" in clean or (clean[:1].isupper() and len(clean) >= 3 and clean.lower() not in _STOP):
            if clean not in names:
                names.append(clean)
        if len(names) >= 6:
            break
    if len(names) < 2:
        names.append("conversation")
    return names


def graphiti_structured_reply(schema_name: str, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Minimal valid instances of Graphiti's structured-output models.

    Keyed by the JSON-schema name the SDK sends (the pydantic model name), so the
    same logic serves chat-completions json_schema and the Responses API.  The
    replies make Graphiti store one fact per (first entity -> other entity)
    containing the full message text, which is what the adapter tests rely on.
    """
    user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    user = user if isinstance(user, str) else ""
    current = _tag(user, "CURRENT MESSAGE") or _tag(user, "CURRENT_MESSAGE") or _tag(user, "TEXT") or ""
    if schema_name in ("ExtractedEntities", "CombinedExtraction"):
        ents = [{"name": n, "entity_type_id": 0, "episode_indices": [0]} for n in _entities_from_message(current)]
        out: dict[str, Any] = {"extracted_entities": ents}
        if schema_name == "CombinedExtraction":
            out["edges"] = [{"source_entity_name": ents[0]["name"], "target_entity_name": e["name"], "relation_type": "MENTIONS", "fact": current.strip(), "episode_indices": [0]} for e in ents[1:3]]
        return out
    if schema_name == "ExtractedEdges":
        names: list[str] = []
        raw = _tag(user, "ENTITIES")
        try:
            names = [str(n.get("name")) for n in json.loads(raw or "[]") if isinstance(n, dict) and n.get("name")]
        except json.JSONDecodeError:
            names = []
        fact = current.strip()
        edges = [{"source_entity_name": names[0], "target_entity_name": t, "relation_type": "MENTIONS", "fact": fact, "valid_at": None, "invalid_at": None, "episode_indices": [0]} for t in names[1:4]] if len(names) >= 2 and fact else []
        return {"edges": edges}
    if schema_name == "NodeResolutions":
        raw = _tag(user, "ENTITIES")
        try:
            items = [n for n in json.loads(raw or "[]") if isinstance(n, dict)]
        except json.JSONDecodeError:
            items = []
        return {"entity_resolutions": [{"id": int(n.get("id", i)), "name": str(n.get("name", f"entity {i}")), "duplicate_candidate_id": -1} for i, n in enumerate(items)]}
    if schema_name == "EdgeDuplicate":
        return {"duplicate_facts": [], "contradicted_facts": []}
    if schema_name == "EdgeTimestamps":
        return {"valid_at": None, "invalid_at": None}
    if schema_name == "BatchEdgeTimestamps":
        return {"timestamps": []}
    if schema_name == "SummarizedEntities":
        return {"summaries": []}
    if schema_name in ("Summary", "EntitySummary", "SagaSummary"):
        return {"summary": "Summary unavailable in the fake provider."}
    if schema_name == "SummaryDescription":
        return {"description": "A summary."}
    return None


def _schema_default(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Generic filler for unknown json schemas: required fields get type defaults."""
    if "$ref" in schema:
        return _schema_default(defs.get(schema["$ref"].rsplit("/", 1)[-1], {}), defs)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "null")
    if "anyOf" in schema:
        return _schema_default(schema["anyOf"][0], defs)
    if t == "object":
        return {k: _schema_default(v, defs) for k, v in (schema.get("properties") or {}).items() if k in (schema.get("required") or [])}
    return {"array": [], "string": "", "integer": 0, "number": 0.0, "boolean": False}.get(t, None)


def structured_reply(schema: dict[str, Any] | None, messages: list[dict[str, Any]]) -> str | None:
    """JSON text for a json_schema request: Graphiti emulation first, generic filler otherwise."""
    if not schema:
        return None
    name = schema.get("name") or (schema.get("json_schema") or {}).get("name") or ""
    obj = graphiti_structured_reply(name, messages)
    if obj is None:
        js = schema.get("schema") or (schema.get("json_schema") or {}).get("schema") or {}
        obj = _schema_default(js, js.get("$defs") or {})
    return json.dumps(obj)


def reranker_reply(messages: list[dict[str, Any]]) -> tuple[str, float] | None:
    """Graphiti's OpenAI cross-encoder asks True/False with logprobs; score by word overlap."""
    user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    if not isinstance(user, str) or "<PASSAGE>" not in user or "<QUERY>" not in user:
        return None
    passage, query = _tag(user, "PASSAGE") or "", _tag(user, "QUERY") or ""
    overlap = len(_words(passage) & _words(query))
    p = min(0.95, 0.05 + 0.3 * overlap)
    return ("True" if p >= 0.5 else "False"), p


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
        rf = body.get("response_format") if isinstance(body.get("response_format"), dict) else None
        scripted = structured_reply(rf.get("json_schema"), messages) if rf and rf.get("type") == "json_schema" else None
        rerank = reranker_reply(messages) if body.get("logprobs") else None
        if scripted is None and rerank is not None:
            scripted = rerank[0]
        if scripted is None:
            scripted = mem0_extraction_reply(messages)
        if scripted is None:
            scripted = longmemeval_reader_reply(messages)
        if scripted is None:
            scripted = longmemeval_judge_reply(messages)
        if scripted is not None:
            completion = scripted
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

        choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": completion}, "finish_reason": "stop"}
        if rerank is not None:
            import math

            p = rerank[1]
            top = [{"token": "True", "logprob": math.log(p), "bytes": None}, {"token": "False", "logprob": math.log(1 - p), "bytes": None}]
            top.sort(key=lambda x: -x["logprob"])
            choice["logprobs"] = {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"], "bytes": None, "top_logprobs": top}]}
        return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": model, "choices": [choice], "usage": usage}, headers=headers)

    @app.post("/v1/responses")
    async def responses(request: Request):
        """OpenAI Responses API (non-streaming) with the same scripted replies."""
        app.state.calls += 1
        body = json.loads(await request.body())
        model = body.get("model")
        failure = await _maybe_fail_or_delay(model)
        if failure is not None:
            return failure
        inp = body.get("input")
        messages = inp if isinstance(inp, list) else [{"role": "user", "content": str(inp or "")}]
        fmt = ((body.get("text") or {}).get("format") or {}) if isinstance(body.get("text"), dict) else {}
        text = structured_reply(fmt, messages) if fmt.get("type") == "json_schema" else None
        if text is None:
            text = mem0_extraction_reply(messages) or longmemeval_reader_reply(messages) or longmemeval_judge_reply(messages) or " ".join(f"w{i + 1}" for i in range(min(int(body.get("max_output_tokens") or 16), 16)))
        prompt_tokens = chat_prompt_tokens(messages)
        out_tokens = word_count(text)
        rid = "resp_" + uuid.uuid4().hex[:12]
        return JSONResponse(
            {
                "id": rid,
                "object": "response",
                "created_at": int(time.time()),
                "model": model,
                "status": "completed",
                "output": [{"id": "msg_" + rid, "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                "usage": {"input_tokens": prompt_tokens, "output_tokens": out_tokens, "total_tokens": prompt_tokens + out_tokens, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            headers={"x-request-id": "up-" + uuid.uuid4().hex[:8]},
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
            data.append({"object": "embedding", "index": i, "embedding": fake_embedding(item)})
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
