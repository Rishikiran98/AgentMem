"""Load raw JSONL traces into DataFrames.

Two sources, joined only by identifiers that both carry:
  run log   (results/raw/runs/<run_id>/run.jsonl)   events keyed by run_id, question_id
  proxy log (results/raw/proxy/*.jsonl)              MODEL_CALL events keyed by run_id, session_id (= question_id), operation
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_jsonl(path: str | Path) -> list[dict]:
    """All events of one append-only JSONL file; a corrupt line is an error, never skipped."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: corrupt JSONL") from exc
    return out


@dataclass
class RunTrace:
    run_dir: Path
    manifest: dict
    events: pd.DataFrame
    model_calls: pd.DataFrame  # proxy MODEL_CALL rows for this run (may be empty)

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    def of(self, event_type: str) -> pd.DataFrame:
        return self.events[self.events["event_type"] == event_type].copy()


def load_run(run_dir: str | Path, proxy_logs: Iterable[str | Path] | None = None) -> RunTrace:
    run_dir = Path(run_dir)
    events = pd.DataFrame(read_jsonl(run_dir / "run.jsonl"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run_id = manifest["run_id"]
    starts = events[events["event_type"] == "RUN_START"]
    if len(starts) != 1 or starts.iloc[0]["run_id"] != run_id:
        raise ValueError(f"{run_dir}: run.jsonl does not contain exactly one RUN_START for {run_id}")

    candidates: list[Path] = [Path(p) for p in (proxy_logs or [])]
    if not candidates:
        recorded = manifest.get("proxy", {}).get("log_path")
        if recorded and Path(recorded).exists():
            candidates = [Path(recorded)]
    rows: list[dict] = []
    for p in candidates:
        for ev in read_jsonl(p):
            if ev.get("event_type") == "MODEL_CALL" and ev.get("run_id") == run_id:
                ev["proxy_log"] = str(p)
                rows.append(ev)
    calls = pd.DataFrame(rows)
    if calls.empty:
        calls = pd.DataFrame(columns=["request_id", "run_id", "session_id", "operation", "endpoint", "status", "prompt_tokens", "completion_tokens", "total_tokens", "duration_ms", "upstream_duration_ms", "model"])
    return RunTrace(run_dir=run_dir, manifest=manifest, events=events, model_calls=calls)


def load_runs(root: str | Path, proxy_logs: Iterable[str | Path] | None = None) -> list[RunTrace]:
    root = Path(root)
    return [load_run(d, proxy_logs) for d in sorted(root.iterdir()) if (d / "run.jsonl").exists() and (d / "manifest.json").exists()]
