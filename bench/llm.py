"""Proxy-routed chat client for harness-owned model calls (reader, judge).

Every call carries per-request attribution headers so the proxy log can
separate reader and judge cost from the memory system's own calls.  The run
log stores only the proxy request id, text, and latency; token usage is read
from the proxy log during analysis, never from here.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from proxy.client import ProxyClient


@dataclass
class ChatOutcome:
    text: str
    proxy_request_id: str | None
    upstream_request_id: str | None
    latency_ms: float
    model: str
    prompt_sha256: str
    finish_reason: str | None
    http_status: int


class ProxiedChat:
    def __init__(self, proxy_base_url: str, *, system: str, configuration: str, seed: int, run_id: str, client_id: str, timeout_s: float = 180.0, max_retries: int = 2) -> None:
        self.pc = ProxyClient(proxy_base_url)
        self.static = dict(system=system, configuration=configuration, seed=seed, run_id=run_id, client_id=client_id)
        self._client = AsyncOpenAI(base_url=self.pc.base_url(), api_key=self.pc.api_key(**self.static), timeout=timeout_s, max_retries=max_retries)

    async def complete(self, *, model: str, messages: list[dict[str, Any]], operation: str, session_id: str, temperature: float, max_tokens: int, n: int = 1) -> ChatOutcome:
        headers = self.pc.headers(operation=operation, session_id=session_id)
        prompt_sha = hashlib.sha256("\n".join(m["content"] for m in messages).encode()).hexdigest()
        t0 = time.perf_counter()
        raw = await self._client.chat.completions.with_raw_response.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, extra_headers=headers)
        latency = round((time.perf_counter() - t0) * 1000.0, 3)
        parsed = raw.parse()
        choice = parsed.choices[0]
        return ChatOutcome(
            text=(choice.message.content or ""),
            proxy_request_id=raw.headers.get("X-Bench-Request-Id"),
            upstream_request_id=raw.headers.get("X-Upstream-Request-Id"),
            latency_ms=latency,
            model=parsed.model or model,
            prompt_sha256=prompt_sha,
            finish_reason=choice.finish_reason,
            http_status=raw.status_code,
        )

    async def aclose(self) -> None:
        await self._client.close()
        await self.pc.aclose()
