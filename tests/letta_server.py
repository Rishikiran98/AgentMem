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


def letta_env_available() -> bool:
    return (LETTA_ENV / "bin" / "letta").exists() and (LETTA_ENV / "letta-src" / "alembic.ini").exists()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LettaServerProcess:
    def __init__(self, workdir: str | Path, *, openai_base_url: str, openai_api_key: str = "sk-bench-letta-server", port: int | None = None) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.port = port or _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.pg = EmbeddedPostgres(self.workdir / "pgdata")
        self.proc: subprocess.Popen | None = None
        self.log = self.workdir / "letta-server.log"

    def start(self, timeout: float = 300.0) -> "LettaServerProcess":
        self.pg.start()
        env = {**os.environ, "LETTA_DIR": str(self.workdir / "letta-dir"), "LETTA_PG_URI": self.pg.uri, "OPENAI_API_KEY": self.openai_api_key, "OPENAI_BASE_URL": self.openai_base_url, "LETTA_LLM_API_LOGGING": "false", "LETTA_DEBUG": "false"}
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
        env = {**os.environ, "LETTA_DIR": str(self.workdir / "letta-dir"), "LETTA_PG_URI": self.pg.uri, "OPENAI_API_KEY": self.openai_api_key, "OPENAI_BASE_URL": self.openai_base_url, "LETTA_LLM_API_LOGGING": "false"}
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
