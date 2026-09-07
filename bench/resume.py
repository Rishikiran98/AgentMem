"""Exact-run resume indexing with fingerprint mismatch rejection."""
from __future__ import annotations
from pathlib import Path
from proxy.logging import read_events

def completed_cells(path:str|Path, *, run_id:str, configuration_id:str)->set[tuple[str,str]]:
    p=Path(path)
    if not p.exists(): return set()
    events=list(read_events(p)); starts=[e for e in events if e.get("event_type")=="RUN_START"]
    if not starts: raise ValueError("resume log has no RUN_START")
    start=starts[0]
    if start.get("run_id")!=run_id: raise ValueError("resume run_id mismatch")
    if start.get("configuration_id")!=configuration_id: raise ValueError("resume configuration fingerprint mismatch")
    return {(e["question_id"],"instance") for e in events if e.get("event_type")=="INSTANCE_END" and e.get("error") is None}
