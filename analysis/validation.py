"""Fail-closed validation for canonical paper runs."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from analysis.ingest import read_jsonl

@dataclass
class ValidationReport:
    valid:bool; errors:list[str]; warnings:list[str]; run_id:str|None
    def as_dict(self): return asdict(self)

def validate_run(path:str|Path, *, expected_benchmark_revision:str|None=None)->ValidationReport:
    ev=read_jsonl(path); errors=[]; warnings=[]
    starts=[x for x in ev if x.get("event_type")=="RUN_START"]; ends=[x for x in ev if x.get("event_type")=="RUN_END"]
    if len(starts)!=1: errors.append("exactly one RUN_START required")
    if len(ends)!=1: errors.append("exactly one RUN_END required")
    start=starts[0] if starts else {}; run_id=start.get("run_id")
    required=("run_id","seed","configuration_id","dataset","adapter_fingerprint","host")
    for k in required:
        if start.get(k) is None: errors.append(f"missing {k}")
    if start.get("dataset",{}).get("synthetic"): errors.append("synthetic/fake data forbidden in canonical run")
    if not start.get("dataset",{}).get("sha256"): errors.append("missing dataset hash")
    if start.get("host",{}).get("git",{}).get("dirty"): errors.append("dirty git tree")
    if start.get("overrides_present"): errors.append("runtime config overrides are noncanonical")
    rev=start.get("benchmark_version")
    if expected_benchmark_revision and rev!=expected_benchmark_revision: errors.append("unexpected benchmark revision")
    for i,x in enumerate(ev):
        if run_id and x.get("run_id") not in (None,run_id): errors.append(f"mixed run_id at event {i}")
        if x.get("event_type")=="MODEL_CALL" and x.get("unattributed"): errors.append(f"unattributed model call at event {i}")
        a=x.get("scheduled_send_time"); d=x.get("actual_dispatch_time"); c=x.get("completion_time")
        if a is not None and not a<=d<=c: errors.append(f"invalid timestamp order at event {i}")
    return ValidationReport(not errors,sorted(set(errors)),warnings,run_id)
