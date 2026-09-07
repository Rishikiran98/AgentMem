"""Run the proxy:  python -m proxy [--host H] [--port P] [--upstream URL] [--log PATH]

Environment variables (see proxy/settings.py) provide defaults; CLI flags override.
The proxy must run as a single worker because the scope registry is in-process.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import uvicorn

from proxy.app import create_app
from proxy.settings import ProxySettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m proxy", description="memharness instrumented LLM proxy")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--upstream", help="upstream OpenAI-compatible base URL, e.g. https://api.openai.com/v1")
    parser.add_argument("--upstream-api-key", help="upstream API key (default: $PROXY_UPSTREAM_API_KEY or $OPENAI_API_KEY)")
    parser.add_argument("--log", help="append-only JSONL event log path")
    parser.add_argument("--fsync", action="store_true", help="fsync after every event")
    parser.add_argument("--log-bodies", action="store_true", help="also record request/response bodies")
    parser.add_argument("--require-attribution", action="store_true", help="reject unattributed model requests")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    settings = ProxySettings.from_env()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.upstream:
        settings.upstream_base_url = args.upstream.rstrip("/")
    if args.upstream_api_key:
        settings.upstream_api_key = args.upstream_api_key
    if args.log:
        from pathlib import Path

        settings.log_path = Path(args.log)
    if args.fsync:
        settings.log_fsync = True
    if args.log_bodies:
        settings.log_bodies = True
    if args.require_attribution:
        settings.require_attribution = True

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not settings.upstream_api_key:
        logging.getLogger("memharness.proxy").warning("no upstream API key configured (PROXY_UPSTREAM_API_KEY / OPENAI_API_KEY); requests are forwarded without Authorization")
    app = create_app(settings)
    logging.getLogger("memharness.proxy").info("event log: %s", settings.log_path)
    uvicorn.run(app, host=settings.host, port=settings.port, workers=1, log_level=args.log_level, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
