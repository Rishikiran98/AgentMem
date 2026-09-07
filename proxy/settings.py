"""Environment-driven configuration for the proxy.

Every setting is captured (with secrets redacted) in the PROXY_START event so
that a trace file is self-describing.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


DEFAULT_ALLOWED_SYSTEMS = ("mem0", "zep", "letta", "hieromem", "harness", "test")


@dataclass
class ProxySettings:
    host: str = "127.0.0.1"
    port: int = 8811

    # Upstream provider (OpenAI-compatible).  Everything after the host is kept,
    # e.g. "https://api.openai.com/v1".
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str | None = None
    upstream_connect_timeout_s: float = 10.0
    upstream_timeout_s: float = 180.0
    upstream_max_connections: int = 512

    # Event log
    log_path: Path = field(default_factory=lambda: Path("results/raw/proxy") / f"proxy-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}.jsonl")
    log_fsync: bool = False
    # When enabled, full request/response bodies are written to a sibling
    # "<log>.bodies.jsonl" file keyed by request_id (useful for prompt audits).
    log_bodies: bool = False

    # Inject stream_options.include_usage=true into streaming chat requests so
    # the upstream reports exact usage.  The extra usage chunk is stripped from
    # the client-facing stream unless the client asked for it itself.
    force_stream_usage: bool = True

    # Reject (HTTP 400) any model request that carries no attribution at all.
    # Off by default for exploratory use; turn on for paper campaigns.
    require_attribution: bool = False

    allowed_systems: tuple[str, ...] = DEFAULT_ALLOWED_SYSTEMS

    @classmethod
    def from_env(cls) -> "ProxySettings":
        s = cls()
        s.host = os.environ.get("PROXY_HOST", s.host)
        s.port = int(os.environ.get("PROXY_PORT", s.port))
        s.upstream_base_url = os.environ.get("PROXY_UPSTREAM_BASE_URL", s.upstream_base_url).rstrip("/")
        s.upstream_api_key = os.environ.get("PROXY_UPSTREAM_API_KEY") or os.environ.get("OPENAI_API_KEY") or None
        s.upstream_connect_timeout_s = _env_float("PROXY_UPSTREAM_CONNECT_TIMEOUT_S", s.upstream_connect_timeout_s)
        s.upstream_timeout_s = _env_float("PROXY_UPSTREAM_TIMEOUT_S", s.upstream_timeout_s)
        s.upstream_max_connections = int(os.environ.get("PROXY_UPSTREAM_MAX_CONNECTIONS", s.upstream_max_connections))
        if os.environ.get("PROXY_LOG_PATH"):
            s.log_path = Path(os.environ["PROXY_LOG_PATH"])
        s.log_fsync = _env_bool("PROXY_LOG_FSYNC", s.log_fsync)
        s.log_bodies = _env_bool("PROXY_LOG_BODIES", s.log_bodies)
        s.force_stream_usage = _env_bool("PROXY_FORCE_STREAM_USAGE", s.force_stream_usage)
        s.require_attribution = _env_bool("PROXY_REQUIRE_ATTRIBUTION", s.require_attribution)
        if os.environ.get("PROXY_ALLOWED_SYSTEMS"):
            s.allowed_systems = tuple(x.strip() for x in os.environ["PROXY_ALLOWED_SYSTEMS"].split(",") if x.strip())
        return s

    def redacted(self) -> dict:
        d = asdict(self)
        d["log_path"] = str(self.log_path)
        d["upstream_api_key"] = "<set>" if self.upstream_api_key else None
        d["allowed_systems"] = list(self.allowed_systems)
        return d
