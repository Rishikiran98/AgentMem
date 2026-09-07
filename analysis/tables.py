from __future__ import annotations
import csv,json
from pathlib import Path
def write_table(rows:list[dict],stem:str|Path):
    stem=Path(stem); stem.parent.mkdir(parents=True,exist_ok=True); (stem.with_suffix('.json')).write_text(json.dumps(rows,indent=2)+"\n")
    if rows:
      with stem.with_suffix('.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
