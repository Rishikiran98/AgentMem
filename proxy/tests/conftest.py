from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest

from proxy.app import create_app
from proxy.fake_upstream import create_fake_upstream
from proxy.logging import AppendOnlyJsonlLogger, read_events
from proxy.settings import ProxySettings


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "proxy.jsonl"


@pytest.fixture
def settings(log_path: Path) -> ProxySettings:
    return ProxySettings(upstream_base_url="http://upstream/v1", upstream_api_key="upstream-secret", log_path=log_path, upstream_timeout_s=5.0)


@pytest.fixture
def fake_upstream():
    return create_fake_upstream()


@pytest.fixture
def proxy_app(settings: ProxySettings, fake_upstream):
    """Proxy wired to the fake upstream through an in-process ASGI transport."""
    transport = httpx.ASGITransport(app=fake_upstream)
    return create_app(settings, upstream_transport=transport)


@pytest.fixture
async def client(proxy_app) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_app), base_url="http://proxy", timeout=10.0) as c:
        yield c


def events(log_path: Path, event_type: str | None = "MODEL_CALL") -> list[dict]:
    evs = list(read_events(log_path))
    return [e for e in evs if event_type is None or e.get("event_type") == event_type]


def chat_body(text: str = "the quick brown fox", *, model: str = "fake-model", **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}
