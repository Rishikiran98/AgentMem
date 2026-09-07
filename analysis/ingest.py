from __future__ import annotations
import json
from pathlib import Path

def read_jsonl(path:str|Path)->list[dict]:
    out=[]
    with Path(path).open() as fh:
        for n,line in enumerate(fh,1):
            try: out.append(json.loads(line))
            except json.JSONDecodeError as exc: raise ValueError(f"{path}:{n}: corrupt JSONL") from exc
    return out
