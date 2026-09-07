"""Host and software metadata captured in every RUN_START event."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def git_info(root: str | Path | None = None) -> dict[str, Any]:
    cwd = str(root) if root else None
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL, timeout=5).decode().strip())
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        return {"commit": sha, "dirty": dirty, "branch": branch}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None, "branch": None}


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or None


def _mem_total_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                kb = int(re.findall(r"\d+", line)[0])
                return round(kb / 1024 / 1024, 2)
    except Exception:  # noqa: BLE001
        pass
    return None


def _gpu() -> list[str] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    return out.splitlines() if out else None


def _versions(names: tuple[str, ...]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def host_metadata(root: str | Path | None = None) -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": {"model": _cpu_model(), "logical_cores": os.cpu_count()},
        "ram_gb": _mem_total_gb(),
        "gpu": _gpu(),
        "docker_version": _run(["docker", "--version"]),
        "python": sys.version.split()[0],
        "git": git_info(root),
        "packages": _versions(("memharness", "mem0ai", "qdrant-client", "openai", "httpx", "fastapi", "uvicorn", "tiktoken", "pyyaml")),
    }
