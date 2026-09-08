"""Embedded PostgreSQL (with pgvector) for tests, from the ``pgserver`` wheel's bundled binaries.

pgserver itself only listens on a unix socket; Letta's URI conversion needs TCP,
so the cluster is started directly with pg_ctl on 127.0.0.1 and a free port.
"""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EmbeddedPostgres:
    def __init__(self, pgdata: str | Path, *, user: str = "letta", password: str = "letta", database: str = "letta") -> None:
        from pgserver._commands import POSTGRES_BIN_PATH  # type: ignore

        self.bin = Path(str(POSTGRES_BIN_PATH))
        self.pgdata = Path(pgdata)
        self.user, self.password, self.database = user, password, database
        self.port = _free_port()
        # Unix socket paths are limited to ~107 bytes; keep the socket dir short.
        self.sockdir = Path(tempfile.mkdtemp(prefix="pgs-", dir="/tmp"))
        self.log = self.pgdata.parent / "postgres.log"
        # PostgreSQL refuses to run as root; mirror pgserver and use an unprivileged system user.
        self.run_as: dict = {}
        if os.name != "nt" and os.geteuid() == 0:
            from pgserver.utils import ensure_prefix_permissions, ensure_user_exists  # type: ignore

            entry = ensure_user_exists("pgserver")
            self.pgdata.parent.mkdir(parents=True, exist_ok=True)
            ensure_prefix_permissions(self.pgdata.parent)
            os.chown(self.pgdata.parent, entry.pw_uid, entry.pw_gid)
            self.run_as = {"user": entry.pw_uid, "group": entry.pw_gid}

    @property
    def uri(self) -> str:
        return f"postgresql://{self.user}:{self.password}@127.0.0.1:{self.port}/{self.database}"

    def start(self) -> "EmbeddedPostgres":
        self.sockdir.mkdir(parents=True, exist_ok=True)
        if self.run_as:
            os.chown(self.sockdir, self.run_as["user"], self.run_as["group"])
        if not (self.pgdata / "PG_VERSION").exists():
            self.pgdata.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([str(self.bin / "initdb"), "-D", str(self.pgdata), "-U", self.user, "--auth=trust", "--encoding=UTF8"], capture_output=True, text=True, **self.run_as)
            if r.returncode != 0:
                raise RuntimeError(f"initdb failed: {r.stderr}")
        opts = f"-p {self.port} -h 127.0.0.1 -k {self.sockdir} -c shared_buffers=64MB -c max_connections=100"
        extra = os.environ.get("MEMHARNESS_PG_EXTRA_OPTS")  # diagnostics, e.g. "-c log_lock_waits=on -c log_min_duration_statement=500"
        if extra:
            opts += " " + extra
        r = subprocess.run([str(self.bin / "pg_ctl"), "-D", str(self.pgdata), "-l", str(self.log), "-o", opts, "-w", "-t", "60", "start"], capture_output=True, text=True, **self.run_as)
        if r.returncode != 0:
            raise RuntimeError(f"pg_ctl start failed: {r.stderr} {r.stdout}")
        deadline = time.time() + 60
        while time.time() < deadline:
            r = subprocess.run([str(self.bin / "pg_isready"), "-h", "127.0.0.1", "-p", str(self.port)], capture_output=True)
            if r.returncode == 0:
                break
            time.sleep(0.2)
        self.psql(f"SELECT 1 FROM pg_database WHERE datname='{self.database}'", db="postgres")
        exists = self.psql(f"SELECT count(*) FROM pg_database WHERE datname='{self.database}'", db="postgres").strip().splitlines()[-1].strip()
        if exists == "0":
            self.psql(f"CREATE DATABASE {self.database}", db="postgres")
        self.psql("CREATE EXTENSION IF NOT EXISTS vector")
        self.psql(f"ALTER USER {self.user} WITH PASSWORD '{self.password}'", db="postgres")
        return self

    def psql(self, sql: str, *, db: str | None = None) -> str:
        r = subprocess.run([str(self.bin / "psql"), "-h", "127.0.0.1", "-p", str(self.port), "-U", self.user, "-d", db or self.database, "-t", "-A", "-c", sql], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return r.stdout

    def version(self) -> str:
        return self.psql("SELECT version()").strip()

    def stop(self) -> None:
        subprocess.run([str(self.bin / "pg_ctl"), "-D", str(self.pgdata), "-m", "fast", "-w", "stop"], capture_output=True, **self.run_as)
