"""Canonical-run safety checks."""
from __future__ import annotations
import os,shutil,subprocess
from pathlib import Path
from bench.metadata import git_info,host_metadata

def preflight(*,root:Path,proxy_info:dict,target_system:str,config_id:str,dataset_sha256:str,allow_paid:bool)->dict:
    errors=[]; git=git_info(root)
    if git["dirty"]: errors.append("working tree is dirty")
    if not proxy_info.get("settings",{}).get("require_attribution"): errors.append("PROXY_REQUIRE_ATTRIBUTION is not enabled")
    if not config_id or not dataset_sha256: errors.append("missing configuration or dataset fingerprint")
    if not allow_paid or os.environ.get("BENCH_ALLOW_PAID_RUN")!="1": errors.append("BENCH_ALLOW_PAID_RUN=1 is required")
    free=shutil.disk_usage(root).free
    if free < 5*1024**3: errors.append("less than 5 GiB free disk")
    return {"ok":not errors,"errors":errors,"target_system":target_system,"free_disk_bytes":free,"host":host_metadata(root)}

def require_budget(estimated_usd:float)->None:
    ceiling=os.environ.get("BENCH_MAX_USD")
    if ceiling is None: raise RuntimeError("BENCH_MAX_USD must be set for paid campaigns")
    if estimated_usd>float(ceiling): raise RuntimeError(f"estimated ${estimated_usd:.2f} exceeds BENCH_MAX_USD=${float(ceiling):.2f}")
