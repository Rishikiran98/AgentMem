"""Harness-side helpers for talking to the proxy.

Typical use inside an adapter or the benchmark runner::

    pc = ProxyClient("http://127.0.0.1:8811")

    # A system under test is configured once with a static token:
    api_key  = pc.api_key(system="mem0", configuration="cfg-abc", seed=42, client_id="mem0")
    base_url = pc.base_url()                       # http://127.0.0.1:8811/v1
    # or, for systems that validate the API key format:
    base_url = pc.base_url(system="mem0", ...)     # http://127.0.0.1:8811/b/<token>/v1

    # Around each benchmark operation the harness declares the active scope:
    async with pc.scope("mem0", operation="write", session_id="s-17") as scope_id:
        await memory.add(...)

    # Harness-owned calls (reader/judge) tag themselves per request:
    headers = pc.headers(system="mem0", operation="answer", session_id="s-17", seed=42)
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping

import httpx

from proxy.tags import FIELD_TO_HEADER, api_key_for, encode_tags, normalize_tags


class ProxyClient:
    def __init__(self, base_url: str, *, http: httpx.AsyncClient | None = None, timeout: float = 10.0) -> None:
        self.root = base_url.rstrip("/")
        self._timeout = timeout
        self._http = http or httpx.AsyncClient(timeout=timeout)

    @property
    def closed(self) -> bool:
        return self._http.is_closed

    def reopen(self) -> None:
        """Create a fresh HTTP client after ``aclose()`` (adapters restart between phases)."""
        if self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=self._timeout)

    # -- static tagging ---------------------------------------------------
    @staticmethod
    def api_key(**tags: Any) -> str:
        return api_key_for(tags)

    def base_url(self, **tags: Any) -> str:
        """OpenAI-compatible base URL; with tags, a path-token URL."""
        if tags:
            return f"{self.root}/b/{encode_tags(tags)}/v1"
        return f"{self.root}/v1"

    @staticmethod
    def headers(**tags: Any) -> dict[str, str]:
        clean = normalize_tags(tags)
        return {FIELD_TO_HEADER[k]: str(v) for k, v in clean.items()}

    # -- scopes -------------------------------------------------------------
    async def enter_scope(self, client_id: str, *, scope_id: str | None = None, **tags: Any) -> str:
        scope_id = scope_id or uuid.uuid4().hex
        r = await self._http.post(f"{self.root}/_bench/scope/enter", json={"client_id": client_id, "scope_id": scope_id, "tags": normalize_tags(tags)})
        r.raise_for_status()
        return r.json()["scope_id"]

    async def exit_scope(self, client_id: str, scope_id: str) -> None:
        r = await self._http.post(f"{self.root}/_bench/scope/exit", json={"client_id": client_id, "scope_id": scope_id})
        r.raise_for_status()

    @asynccontextmanager
    async def scope(self, client_id: str, **tags: Any) -> AsyncIterator[str]:
        scope_id = await self.enter_scope(client_id, **tags)
        try:
            yield scope_id
        finally:
            await self.exit_scope(client_id, scope_id)

    # -- introspection --------------------------------------------------------
    async def info(self) -> Mapping[str, Any]:
        r = await self._http.get(f"{self.root}/_bench/info")
        r.raise_for_status()
        return r.json()

    async def healthy(self) -> bool:
        try:
            r = await self._http.get(f"{self.root}/healthz")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._http.aclose()
