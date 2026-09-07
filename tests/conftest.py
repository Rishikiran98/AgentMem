"""Live-server fixtures: the fake upstream and the proxy run as real uvicorn servers
in background threads, because the systems under test use their own HTTP clients."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest
import uvicorn

from proxy.app import create_app
from proxy.fake_upstream import create_fake_upstream
from proxy.settings import ProxySettings


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    def __init__(self, app, port: int) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False, lifespan="on")
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, health_path: str) -> "LiveServer":
        self.thread.start()
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if httpx.get(self.url + health_path, timeout=0.5).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        raise RuntimeError(f"server on {self.url} did not start")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture(scope="session")
def fake_upstream_server() -> Iterator[LiveServer]:
    srv = LiveServer(create_fake_upstream(), _free_port()).start("/v1/models")
    yield srv
    srv.stop()


@pytest.fixture
def proxy_log(tmp_path: Path) -> Path:
    return tmp_path / "proxy.jsonl"


@pytest.fixture
def proxy_server(fake_upstream_server: LiveServer, proxy_log: Path) -> Iterator[LiveServer]:
    settings = ProxySettings(upstream_base_url=fake_upstream_server.url + "/v1", upstream_api_key="fake", log_path=proxy_log, upstream_timeout_s=30.0)
    srv = LiveServer(create_app(settings), _free_port()).start("/healthz")
    yield srv
    srv.stop()


@pytest.fixture(scope="session")
def letta_server(fake_upstream_server: LiveServer, tmp_path_factory) -> Iterator["LettaServerProcess"]:
    """Retired Letta V1 server on embedded PostgreSQL; skipped when ./.venv-letta is absent."""
    from tests.letta_server import LettaServerProcess, letta_env_available

    if not letta_env_available():
        pytest.skip("Letta server environment not built (run scripts/setup_letta_env.sh)")
    srv = LettaServerProcess(tmp_path_factory.mktemp("letta"), openai_base_url=fake_upstream_server.url + "/v1").start()
    yield srv
    srv.stop()
