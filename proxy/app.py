"""FastAPI application: OpenAI-compatible passthrough with per-request instrumentation.

Endpoints
---------
POST /v1/chat/completions            forwarded, instrumented (streaming supported)
POST /v1/embeddings                  forwarded, instrumented
GET  /v1/models[/{id}]               forwarded, instrumented (needed by Letta's provider probe)
*    /b/<token>/v1/...               same endpoints with a static tag token in the path

POST /_bench/scope/enter             register an active harness scope for a client
POST /_bench/scope/exit              release it
GET  /_bench/scopes                  inspect active scopes
GET  /_bench/info                    proxy settings/versions/log path
GET  /healthz                        liveness

Every forwarded request produces exactly one MODEL_CALL event in the
append-only JSONL log, whether it succeeded or failed.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging as stdlogging
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from proxy import EVENT_SCHEMA_VERSION, PROXY_VERSION
from proxy.logging import AppendOnlyJsonlLogger, utc_now_iso
from proxy.settings import ProxySettings
from proxy.tags import Attribution, ScopeRegistry, TagError, normalize_tags, resolve_attribution, split_path_token
from proxy.tokenization import (
    count_chat_prompt_tokens,
    count_embedding_input_tokens,
    count_text_tokens,
    embedding_input_count,
    preload as preload_tokenizer,
    tokenizer_status,
)

log = stdlogging.getLogger("memharness.proxy")

REQUEST_ID_HEADER = "X-Bench-Request-Id"
_PARAM_KEYS = ("temperature", "max_tokens", "max_completion_tokens", "top_p", "n", "seed", "stream", "encoding_format", "dimensions", "response_format")
_ERROR_MESSAGE_LIMIT = 2000


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:  # noqa: BLE001
        return None


class ProxyState:
    def __init__(self, settings: ProxySettings, logger: AppendOnlyJsonlLogger, upstream: httpx.AsyncClient) -> None:
        self.settings = settings
        self.logger = logger
        self.upstream = upstream
        self.registry = ScopeRegistry()
        self.bodies_logger: AppendOnlyJsonlLogger | None = None
        if settings.log_bodies:
            bodies_path = settings.log_path.with_name(settings.log_path.name.replace(".jsonl", "") + ".bodies.jsonl")
            self.bodies_logger = AppendOnlyJsonlLogger(bodies_path, fsync=settings.log_fsync, instance_id=logger.instance_id)
        self.started_at = utc_now_iso()


def create_app(
    settings: ProxySettings | None = None,
    *,
    logger: AppendOnlyJsonlLogger | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the proxy app.  ``upstream_transport`` lets tests mount a fake provider in-process."""
    settings = settings or ProxySettings.from_env()
    logger = logger or AppendOnlyJsonlLogger(settings.log_path, fsync=settings.log_fsync)
    timeout = httpx.Timeout(settings.upstream_timeout_s, connect=settings.upstream_connect_timeout_s)
    limits = httpx.Limits(max_connections=settings.upstream_max_connections, max_keepalive_connections=settings.upstream_max_connections)
    upstream = httpx.AsyncClient(base_url=settings.upstream_base_url, timeout=timeout, limits=limits, transport=upstream_transport)
    state = ProxyState(settings, logger, upstream)

    app = FastAPI(title="memharness instrumented LLM proxy", version=PROXY_VERSION, docs_url=None, redoc_url=None)
    app.state.proxy = state

    logger.append(
        {
            "event_type": "PROXY_START",
            "schema_version": EVENT_SCHEMA_VERSION,
            "timestamp": state.started_at,
            "proxy_version": PROXY_VERSION,
            "git_commit": _git_commit(),
            "settings": settings.redacted(),
            "tokenizer": preload_tokenizer(),
            "host": {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "versions": {"fastapi": _version("fastapi"), "httpx": httpx.__version__, "uvicorn": _version("uvicorn")},
        }
    )

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - exercised by the live demo
        logger.append({"event_type": "PROXY_STOP", "schema_version": EVENT_SCHEMA_VERSION, "timestamp": utc_now_iso()})
        await upstream.aclose()
        logger.close()
        if state.bodies_logger:
            state.bodies_logger.close()

    # ------------------------------------------------------------------ #
    # Control plane
    # ------------------------------------------------------------------ #
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "proxy_instance_id": logger.instance_id, "seq": logger.seq}

    @app.get("/_bench/info")
    async def info() -> dict[str, Any]:
        return {
            "proxy_version": PROXY_VERSION,
            "schema_version": EVENT_SCHEMA_VERSION,
            "proxy_instance_id": logger.instance_id,
            "started_at": state.started_at,
            "settings": settings.redacted(),
            "tokenizer": tokenizer_status(),
            "events_logged": logger.seq,
        }

    @app.get("/_bench/scopes")
    async def scopes() -> dict[str, Any]:
        return {"scopes": state.registry.snapshot()}

    @app.post("/_bench/scope/enter")
    async def scope_enter(request: Request) -> Response:
        body = await _json_body(request)
        if isinstance(body, Response):
            return body
        try:
            tags = normalize_tags(body.get("tags") or {}, allowed_systems=settings.allowed_systems)
            client_id = str(body.get("client_id") or tags.get("client_id") or "")
            scope_id = str(body.get("scope_id") or uuid.uuid4().hex)
            now = utc_now_iso()
            scope = state.registry.enter(client_id, scope_id, tags, now)
        except TagError as exc:
            return _error(400, str(exc), "invalid_scope")
        logger.append(
            {"event_type": "SCOPE_ENTER", "schema_version": EVENT_SCHEMA_VERSION, "timestamp": now, "client_id": client_id, "scope_id": scope_id, "tags": scope.tags}
        )
        return JSONResponse({"client_id": client_id, "scope_id": scope_id, "entered_at": now, "active_scopes": len(state.registry.active(client_id))})

    @app.post("/_bench/scope/exit")
    async def scope_exit(request: Request) -> Response:
        body = await _json_body(request)
        if isinstance(body, Response):
            return body
        client_id = str(body.get("client_id") or "")
        scope_id = str(body.get("scope_id") or "")
        try:
            scope = state.registry.exit(client_id, scope_id)
        except TagError as exc:
            return _error(404, str(exc), "unknown_scope")
        now = utc_now_iso()
        logger.append(
            {"event_type": "SCOPE_EXIT", "schema_version": EVENT_SCHEMA_VERSION, "timestamp": now, "client_id": client_id, "scope_id": scope_id, "tags": scope.tags, "entered_at": scope.entered_at}
        )
        return JSONResponse({"client_id": client_id, "scope_id": scope_id, "exited_at": now})

    # ------------------------------------------------------------------ #
    # Data plane
    # ------------------------------------------------------------------ #
    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def v1_plain(request: Request, path: str) -> Response:
        return await handle_model_request(state, request, path_token=None, api_path="/v1/" + path)

    @app.api_route("/b/{token}/v1/{path:path}", methods=["GET", "POST"])
    async def v1_tokenised(request: Request, token: str, path: str) -> Response:
        return await handle_model_request(state, request, path_token=token, api_path="/v1/" + path)

    return app


def _version(mod: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(mod)
    except Exception:  # noqa: BLE001
        return None


def _error(status: int, message: str, err_type: str, request_id: str | None = None) -> JSONResponse:
    headers = {REQUEST_ID_HEADER: request_id} if request_id else {}
    return JSONResponse({"error": {"message": message, "type": err_type, "code": None, "param": None}}, status_code=status, headers=headers)


async def _json_body(request: Request) -> dict[str, Any] | Response:
    try:
        obj = json.loads(await request.body() or b"{}")
    except json.JSONDecodeError as exc:
        return _error(400, f"invalid JSON body: {exc}", "invalid_request_error")
    if not isinstance(obj, dict):
        return _error(400, "body must be a JSON object", "invalid_request_error")
    return obj


# ---------------------------------------------------------------------- #
# Request handling
# ---------------------------------------------------------------------- #

class CallRecord:
    """Mutable accumulator for one model call; flushed into exactly one MODEL_CALL event."""

    def __init__(self, state: ProxyState, request: Request, api_path: str) -> None:
        self.state = state
        self.request_id = uuid.uuid4().hex
        self.t0 = time.perf_counter()
        self.timestamp_start = utc_now_iso()
        self.method = request.method
        self.path = api_path
        # The upstream base URL already carries the API version segment
        # (e.g. https://api.openai.com/v1), so forward the path without it.
        self.upstream_path = api_path[len("/v1"):] if api_path.startswith("/v1/") else api_path
        self.endpoint = _endpoint_kind(api_path, request.method)
        self.attribution = Attribution()
        self.model: str | None = None
        self.upstream_model: str | None = None
        self.stream = False
        self.usage: dict[str, Any] = {}
        self.usage_source = "none"
        self.prompt_tokens_local: int | None = None
        self.completion_tokens_local: int | None = None
        self.embedding_inputs: int | None = None
        self.embedding_dimensions: int | None = None
        self.embedding_vectors: int | None = None
        self.status = "error"
        self.http_status: int | None = None
        self.upstream_http_status: int | None = None
        self.error_type: str | None = None
        self.error_message: str | None = None
        self.finish_reason: str | None = None
        self.request_bytes = 0
        self.response_bytes = 0
        self.request_sha256: str | None = None
        self.request_params: dict[str, Any] = {}
        self.request_modifications: list[str] = []
        self.upstream_request_id: str | None = None
        self.upstream_started: float | None = None
        self.upstream_finished: float | None = None
        self.first_byte: float | None = None
        self.client_user_agent = request.headers.get("user-agent")
        self.logged = False
        self.t_end: float | None = None
        self.timestamp_end: str | None = None
        # Inputs for local token estimates, computed *after* timing is frozen.
        self._local_messages: Any = None
        self._local_embedding_input: Any = None
        self._local_completion_text: str | None = None

    # Timing helpers -----------------------------------------------------
    def mark_upstream_start(self) -> None:
        self.upstream_started = time.perf_counter()

    def mark_first_byte(self) -> None:
        if self.first_byte is None:
            self.first_byte = time.perf_counter()

    def mark_upstream_end(self) -> None:
        self.upstream_finished = time.perf_counter()

    def mark_done(self) -> None:
        """Freeze end-of-call timing.  Everything after this is off the client's critical path."""
        if self.t_end is None:
            self.t_end = time.perf_counter()
            self.timestamp_end = utc_now_iso()

    def fail(self, http_status: int, error_type: str, message: str) -> None:
        self.status = "error"
        self.http_status = http_status
        self.error_type = error_type
        self.error_message = (message or "")[:_ERROR_MESSAGE_LIMIT]

    # Serialisation --------------------------------------------------------
    def event(self) -> dict[str, Any]:
        self.mark_done()
        t_end = self.t_end
        ms = lambda a, b: round((b - a) * 1000.0, 3) if (a is not None and b is not None) else None  # noqa: E731
        tags = self.attribution.tags
        ev: dict[str, Any] = {
            "event_type": "MODEL_CALL",
            "schema_version": EVENT_SCHEMA_VERSION,
            "request_id": self.request_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "duration_ms": ms(self.t0, t_end),
            "upstream_duration_ms": ms(self.upstream_started, self.upstream_finished),
            "time_to_first_byte_ms": ms(self.upstream_started, self.first_byte),
            "endpoint": self.endpoint,
            "method": self.method,
            "path": self.path,
            # attribution
            "system": tags.get("system"),
            "configuration": tags.get("configuration"),
            "seed": tags.get("seed"),
            "session_id": tags.get("session_id"),
            "operation": tags.get("operation"),
            "run_id": tags.get("run_id"),
            "client_id": tags.get("client_id"),
            "attribution_sources": self.attribution.sources,
            "scope_ids": self.attribution.scope_ids,
            "scope_state": self.attribution.scope_state,
            "ambiguous_fields": self.attribution.ambiguous_fields,
            "unattributed": self.attribution.unattributed,
            # model + usage
            "model": self.model,
            "upstream_model": self.upstream_model,
            "stream": self.stream,
            "prompt_tokens": self.usage.get("prompt_tokens"),
            "completion_tokens": self.usage.get("completion_tokens"),
            "total_tokens": self.usage.get("total_tokens"),
            "usage_details": {k: v for k, v in self.usage.items() if k not in ("prompt_tokens", "completion_tokens", "total_tokens")} or None,
            "usage_source": self.usage_source,
            "prompt_tokens_local": self.prompt_tokens_local,
            "completion_tokens_local": self.completion_tokens_local,
            "embedding_inputs": self.embedding_inputs,
            "embedding_vectors": self.embedding_vectors,
            "embedding_dimensions": self.embedding_dimensions,
            # outcome
            "status": self.status,
            "http_status": self.http_status,
            "upstream_http_status": self.upstream_http_status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "finish_reason": self.finish_reason,
            # request/response
            "request_bytes": self.request_bytes,
            "response_bytes": self.response_bytes,
            "request_sha256": self.request_sha256,
            "request_params": self.request_params,
            "request_modifications": self.request_modifications,
            "upstream_request_id": self.upstream_request_id,
            "client_user_agent": self.client_user_agent,
        }
        return ev

    def log(self) -> None:
        """Compute deferred local estimates and append exactly one MODEL_CALL event."""
        if self.logged:
            return
        self.logged = True
        self.mark_done()
        if self._local_messages is not None:
            self.prompt_tokens_local = count_chat_prompt_tokens(self._local_messages, self.model)
        if self._local_embedding_input is not None:
            self.prompt_tokens_local = count_embedding_input_tokens(self._local_embedding_input, self.model)
        if self._local_completion_text is not None:
            self.completion_tokens_local = count_text_tokens(self._local_completion_text, self.model)
        self.state.logger.append(self.event())


def _endpoint_kind(api_path: str, method: str) -> str:
    p = api_path.rstrip("/")
    if p == "/v1/chat/completions" and method == "POST":
        return "chat"
    if p == "/v1/embeddings" and method == "POST":
        return "embeddings"
    if (p == "/v1/models" or p.startswith("/v1/models/")) and method == "GET":
        return "models"
    return "unsupported"


def _upstream_headers(state: ProxyState, request: Request) -> dict[str, str]:
    headers = {"accept": request.headers.get("accept", "application/json")}
    if state.settings.upstream_api_key:
        headers["authorization"] = f"Bearer {state.settings.upstream_api_key}"
    ct = request.headers.get("content-type")
    if ct:
        headers["content-type"] = ct
    # Pass through provider-specific routing headers if the client set them.
    for h in ("openai-organization", "openai-project", "openai-beta"):
        if h in request.headers:
            headers[h] = request.headers[h]
    return headers


async def handle_model_request(state: ProxyState, request: Request, *, path_token: str | None, api_path: str) -> Response:
    rec = CallRecord(state, request, api_path)
    settings = state.settings

    # 1. Attribution
    try:
        rec.attribution = resolve_attribution(request.headers, path_token=path_token, registry=state.registry, allowed_systems=settings.allowed_systems)
    except TagError as exc:
        rec.fail(400, "invalid_attribution", str(exc))
        rec.log()
        return _error(400, f"invalid benchmark attribution: {exc}", "invalid_attribution", rec.request_id)
    if rec.endpoint == "models" and "operation" not in rec.attribution.tags:
        rec.attribution.tags["operation"] = "meta"
        rec.attribution.sources["operation"] = "default"
    if settings.require_attribution and rec.attribution.unattributed:
        rec.fail(400, "unattributed_request", "request carries no benchmark attribution (headers, token, or scope)")
        rec.log()
        return _error(400, "request carries no benchmark attribution", "unattributed_request", rec.request_id)

    if rec.endpoint == "unsupported":
        rec.fail(404, "unsupported_endpoint", f"{request.method} {api_path} is not proxied")
        rec.log()
        return _error(404, f"{request.method} {api_path} is not supported by the benchmark proxy", "unsupported_endpoint", rec.request_id)

    # 2. Body
    body_bytes = await request.body()
    rec.request_bytes = len(body_bytes)
    rec.request_sha256 = hashlib.sha256(body_bytes).hexdigest() if body_bytes else None
    body: dict[str, Any] = {}
    if rec.endpoint in ("chat", "embeddings"):
        try:
            body = json.loads(body_bytes)
            if not isinstance(body, dict):
                raise ValueError("body is not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            rec.fail(400, "invalid_request_body", str(exc))
            rec.log()
            return _error(400, f"invalid JSON body: {exc}", "invalid_request_error", rec.request_id)
        rec.model = body.get("model") if isinstance(body.get("model"), str) else None
        rec.request_params = {k: body[k] for k in _PARAM_KEYS if k in body}
        if "tools" in body or "functions" in body:
            rec.request_params["tools"] = len(body.get("tools") or body.get("functions") or [])
        if rec.endpoint == "chat":
            rec.request_params["messages"] = len(body.get("messages") or [])
            rec._local_messages = body.get("messages")
            rec.stream = bool(body.get("stream"))
        else:
            rec.embedding_inputs = embedding_input_count(body.get("input"))
            rec._local_embedding_input = body.get("input")
        if state.bodies_logger is not None:
            state.bodies_logger.append({"event_type": "REQUEST_BODY", "request_id": rec.request_id, "body": body})

    # 3. Forward
    if rec.endpoint == "chat" and rec.stream:
        return await _forward_stream(state, request, rec, body)
    return await _forward_simple(state, request, rec, body_bytes if rec.endpoint != "models" else None)


async def _forward_simple(state: ProxyState, request: Request, rec: CallRecord, body_bytes: bytes | None) -> Response:
    headers = _upstream_headers(state, request)
    rec.mark_upstream_start()
    try:
        resp = await state.upstream.request(request.method, rec.upstream_path, content=body_bytes, headers=headers)
        rec.mark_first_byte()
        content = resp.content
        rec.mark_upstream_end()
    except httpx.TimeoutException as exc:
        rec.mark_upstream_end()
        rec.fail(504, "upstream_timeout", f"{type(exc).__name__}: {exc}")
        rec.log()
        return _error(504, f"upstream timeout: {type(exc).__name__}", "upstream_timeout", rec.request_id)
    except httpx.HTTPError as exc:
        rec.mark_upstream_end()
        rec.fail(502, "upstream_unreachable", f"{type(exc).__name__}: {exc}")
        rec.log()
        return _error(502, f"upstream unreachable: {type(exc).__name__}: {exc}", "upstream_unreachable", rec.request_id)

    rec.upstream_http_status = resp.status_code
    rec.http_status = resp.status_code
    rec.response_bytes = len(content)
    rec.upstream_request_id = resp.headers.get("x-request-id")
    payload: Any = None
    try:
        payload = json.loads(content) if content else None
    except json.JSONDecodeError:
        payload = None

    if 200 <= resp.status_code < 300:
        rec.status = "success"
        if isinstance(payload, dict):
            _absorb_usage(rec, payload)
            rec.upstream_model = payload.get("model") if isinstance(payload.get("model"), str) else None
            if rec.endpoint == "chat":
                choices = payload.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    rec.finish_reason = choices[0].get("finish_reason")
                    msg = choices[0].get("message") or {}
                    rec._local_completion_text = msg.get("content") if isinstance(msg.get("content"), str) else None
            elif rec.endpoint == "embeddings":
                data = payload.get("data") or []
                rec.embedding_vectors = len(data)
                if data and isinstance(data[0], dict):
                    rec.embedding_dimensions = _embedding_dims(data[0].get("embedding"))
        if state.bodies_logger is not None and payload is not None:
            state.bodies_logger.append({"event_type": "RESPONSE_BODY", "request_id": rec.request_id, "body": payload})
    else:
        message = None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            message = payload["error"].get("message")
        rec.fail(resp.status_code, "upstream_error", message or content[:_ERROR_MESSAGE_LIMIT].decode("utf-8", errors="replace"))

    rec.mark_done()
    out_headers = {REQUEST_ID_HEADER: rec.request_id}
    if rec.upstream_request_id:
        out_headers["X-Upstream-Request-Id"] = rec.upstream_request_id
    # The event is appended after the response bytes have been handed to the
    # client, so log I/O and local token estimates never inflate observed latency.
    return Response(content=content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"), headers=out_headers, background=BackgroundTask(rec.log))


def _embedding_dims(vec: Any) -> int | None:
    if isinstance(vec, list):
        return len(vec)
    if isinstance(vec, str):  # encoding_format=base64 -> little-endian float32 array
        try:
            return len(base64.b64decode(vec)) // 4
        except Exception:  # noqa: BLE001
            return None
    return None


def _absorb_usage(rec: CallRecord, payload: dict[str, Any]) -> None:
    usage = payload.get("usage")
    if isinstance(usage, dict) and usage:
        rec.usage = {k: v for k, v in usage.items() if v is not None}
        rec.usage_source = "upstream"
        if "total_tokens" not in rec.usage and "prompt_tokens" in rec.usage:
            rec.usage["total_tokens"] = rec.usage["prompt_tokens"] + rec.usage.get("completion_tokens", 0)


async def _forward_stream(state: ProxyState, request: Request, rec: CallRecord, body: dict[str, Any]) -> Response:
    settings = state.settings
    client_requested_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    if settings.force_stream_usage and not client_requested_usage:
        body = dict(body)
        body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        rec.request_modifications.append("stream_options.include_usage=true")
    outbound = json.dumps(body).encode()
    headers = _upstream_headers(state, request)
    headers["content-type"] = "application/json"

    rec.mark_upstream_start()
    try:
        upstream_req = state.upstream.build_request("POST", rec.upstream_path, content=outbound, headers=headers)
        resp = await state.upstream.send(upstream_req, stream=True)
    except httpx.TimeoutException as exc:
        rec.mark_upstream_end()
        rec.fail(504, "upstream_timeout", f"{type(exc).__name__}: {exc}")
        rec.log()
        return _error(504, f"upstream timeout: {type(exc).__name__}", "upstream_timeout", rec.request_id)
    except httpx.HTTPError as exc:
        rec.mark_upstream_end()
        rec.fail(502, "upstream_unreachable", f"{type(exc).__name__}: {exc}")
        rec.log()
        return _error(502, f"upstream unreachable: {type(exc).__name__}: {exc}", "upstream_unreachable", rec.request_id)

    rec.upstream_http_status = resp.status_code
    rec.upstream_request_id = resp.headers.get("x-request-id")
    if resp.status_code < 200 or resp.status_code >= 300:
        # Errors are not streamed by providers; read and pass through.
        try:
            content = await resp.aread()
        finally:
            await resp.aclose()
        rec.mark_first_byte()
        rec.mark_upstream_end()
        rec.response_bytes = len(content)
        message = None
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                message = payload["error"].get("message")
        except json.JSONDecodeError:
            pass
        rec.fail(resp.status_code, "upstream_error", message or content[:_ERROR_MESSAGE_LIMIT].decode("utf-8", errors="replace"))
        rec.log()
        return Response(content=content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"), headers={REQUEST_ID_HEADER: rec.request_id})

    rec.http_status = resp.status_code
    completion_text: list[str] = []

    def process_event(raw: bytes) -> bool:
        """Parse one SSE event; return False if it must be hidden from the client."""
        keep = True
        for line in raw.split(b"\n"):
            line = line.rstrip(b"\r")
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if isinstance(obj.get("model"), str) and rec.upstream_model is None:
                rec.upstream_model = obj["model"]
            choices = obj.get("choices") or []
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    completion_text.append(delta["content"])
                if ch.get("finish_reason"):
                    rec.finish_reason = ch["finish_reason"]
            if isinstance(obj.get("usage"), dict) and obj["usage"]:
                _absorb_usage(rec, obj)
                if not client_requested_usage and not choices:
                    keep = False  # injected usage-only chunk
        return keep

    async def stream_body() -> AsyncIterator[bytes]:
        buf = b""
        try:
            async for chunk in resp.aiter_raw():
                rec.mark_first_byte()
                rec.response_bytes += len(chunk)
                buf += chunk
                while True:
                    idx = buf.find(b"\n\n")
                    if idx < 0:
                        break
                    event_bytes, buf = buf[: idx + 2], buf[idx + 2:]
                    if process_event(event_bytes):
                        yield event_bytes
            if buf:
                if process_event(buf):
                    yield buf
            rec.mark_upstream_end()
            rec.status = "success"
        except asyncio.CancelledError:
            rec.mark_upstream_end()
            rec.fail(499, "client_disconnected", "client closed the connection before the stream completed")
            raise
        except httpx.TimeoutException as exc:
            rec.mark_upstream_end()
            rec.fail(504, "upstream_timeout", f"{type(exc).__name__}: {exc}")
            raise
        except httpx.HTTPError as exc:
            rec.mark_upstream_end()
            rec.fail(502, "upstream_stream_error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            rec.mark_done()
            await resp.aclose()
            rec._local_completion_text = "".join(completion_text)
            if state.bodies_logger is not None:
                state.bodies_logger.append({"event_type": "RESPONSE_BODY", "request_id": rec.request_id, "body": {"stream": True, "content": "".join(completion_text), "finish_reason": rec.finish_reason}})
            rec.log()

    out_headers = {REQUEST_ID_HEADER: rec.request_id}
    if rec.upstream_request_id:
        out_headers["X-Upstream-Request-Id"] = rec.upstream_request_id
    return StreamingResponse(stream_body(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "text/event-stream"), headers=out_headers)
