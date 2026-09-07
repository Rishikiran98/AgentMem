"""Run the retired Letta V1 server (0.16.8) locally for tests and demos.

Requires the isolated environment from scripts/setup_letta_env.sh (./.venv-letta with
the server, its lock-pinned dependencies, and the Alembic migration tree) and the
``pgserver`` wheel for an embedded PostgreSQL with pgvector.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import httpx

from tests.pg_embedded import EmbeddedPostgres

ROOT = Path(__file__).resolve().parents[1]
LETTA_ENV = Path(os.environ.get("LETTA_ENV_DIR", ROOT / ".venv-letta"))

# Deployment environment of the server under test.  Mirrors compose/letta and the
# ``server_env`` block of configs/letta.yaml (a test asserts they agree); the adapter
# copies that block into every fingerprint.  Server defaults are kept except the two
# logging switches (LLM API logging and debug logging only add log volume).
SERVER_ENV = {
    "LETTA_LLM_API_LOGGING": "false",
    "LETTA_DEBUG": "false",
}

# The official image installs Letta with ``uv sync --all-extras``; that includes the
# ``experimental`` extra (uvloop 0.21.0), and uvicorn's ``loop="auto"`` therefore runs
# the server on uvloop.  On CPython's default selector loop the server has a 60 s
# per-step stall (README, Milestone 6: the teardown of its per-request OpenAI client
# deregisters a reused file descriptor and kills a NullPool asyncpg connect), so a
# local environment without uvloop is not the system under test.  ``start()`` refuses
# to run such an environment rather than measuring it.
REQUIRED_RUNTIME = {"event_loop": "uvloop"}


def letta_env_available() -> bool:
    return (LETTA_ENV / "bin" / "letta").exists() and (LETTA_ENV / "letta-src" / "alembic.ini").exists()


def letta_env_runtime() -> dict[str, str | None]:
    """What the server process will actually run on (checked, not assumed)."""
    r = subprocess.run([str(LETTA_ENV / "bin" / "python"), "-c", "import uvloop, uvicorn; print(uvloop.__version__); print(uvicorn.__version__)"], capture_output=True, text=True)
    if r.returncode != 0:
        return {"event_loop": "asyncio-selector (uvloop missing)", "uvloop": None, "uvicorn": None}
    uvl, uvi = r.stdout.split()
    return {"event_loop": "uvloop", "uvloop": uvl, "uvicorn": uvi}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LettaServerProcess:
    def __init__(self, workdir: str | Path, *, openai_base_url: str, openai_api_key: str = "sk-bench-letta-server", port: int | None = None, server_env: dict[str, str] | None = None) -> None:
        self.workdir = Path(workdir)
        self.server_env = dict(server_env or {})
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.port = port or _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.pg = EmbeddedPostgres(self.workdir / "pgdata")
        self.proc: subprocess.Popen | None = None
        self.log = self.workdir / "letta-server.log"
        self.runtime: dict[str, str] = {}

    def _env(self) -> dict[str, str]:
        """Process environment: inherited env, then the frozen deployment env, then the per-instance wiring.

        ``server_env`` overrides (constructor) are for experiments; they are never silent
        because the caller passes them explicitly.
        """
        return {**os.environ, **SERVER_ENV, **self.server_env, "LETTA_DIR": str(self.workdir / "letta-dir"), "LETTA_PG_URI": self.pg.uri, "OPENAI_API_KEY": self.openai_api_key, "OPENAI_BASE_URL": self.openai_base_url}

    def start(self, timeout: float = 300.0) -> "LettaServerProcess":
        self.runtime = letta_env_runtime()
        if self.runtime["event_loop"] != REQUIRED_RUNTIME["event_loop"]:
            raise RuntimeError(f"Letta environment runs on {self.runtime['event_loop']!r}; the official image runs on uvloop (all extras). Rebuild with scripts/setup_letta_env.sh.")
        self.pg.start()
        env = self._env()
        r = subprocess.run([str(LETTA_ENV / "bin" / "alembic"), "upgrade", "head"], cwd=str(LETTA_ENV / "letta-src"), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"alembic upgrade failed: {r.stderr[-2000:]}")
        self.proc = subprocess.Popen([str(LETTA_ENV / "bin" / "letta"), "server", "--port", str(self.port)], env=env, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(self.url + "/v1/health/", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"letta server did not become healthy; log tail:\n{self.log.read_text()[-3000:]}")

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        self.pg.stop()

    def kill(self) -> None:
        """SIGKILL the server process only (Fault C); PostgreSQL keeps running."""
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait(20)
            self.proc = None

    def restart_server(self, timeout: float = 300.0) -> None:
        env = self._env()
        self.proc = subprocess.Popen([str(LETTA_ENV / "bin" / "letta"), "server", "--port", str(self.port)], env=env, stdout=open(self.log, "a"), stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(self.url + "/v1/health/", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError("letta server did not restart")
