"""Load a system config file and build its adapter."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

import yaml

from adapters.base import EventSink, MemoryAdapter, SettlementConfig


def load_config(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    cfg = yaml.safe_load(raw) or {}
    cfg["_config_path"] = str(path)
    # The configuration id is the hash of the committed file content, so every
    # proxy event and run record points at exactly one config text.
    cfg["_configuration_id"] = hashlib.sha256(raw).hexdigest()[:16]
    return cfg


def build_adapter(cfg: dict[str, Any], *, proxy_base_url: str, seed: int, run_id: str, event_sink: EventSink | None = None, overrides: dict[str, Any] | None = None) -> MemoryAdapter:
    module_name, _, class_name = cfg["adapter"].partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    s = cfg.get("settlement", {})
    settlement = SettlementConfig(timeout_s=float(s.get("timeout_s", 30.0)), poll_interval_s=float(s.get("poll_interval_s", 0.25)), canary_cleanup=bool(s.get("canary_cleanup", True)))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    settings = cls.settings_from_config(cfg) if hasattr(cls, "settings_from_config") else _default_settings(cls, cfg)
    return cls(settings, proxy_base_url=proxy_base_url, configuration_id=cfg["_configuration_id"], seed=seed, run_id=run_id, settlement=settlement, event_sink=event_sink)


def _default_settings(cls, cfg):
    mod = importlib.import_module(cls.__module__)
    settings_cls = getattr(mod, f"{cls.__name__.replace('Adapter', '')}Settings")
    return settings_cls.from_config(cfg)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
