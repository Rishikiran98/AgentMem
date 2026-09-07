"""Figure generation is data-driven and refuses an empty canonical input."""
from __future__ import annotations
from pathlib import Path
def generate(rows:list[dict],out:Path):
    if not rows: raise ValueError("no validated rows; refusing placeholder figures")
    try: import matplotlib.pyplot as plt
    except ImportError as exc: raise RuntimeError("install analysis dependencies") from exc
    out.mkdir(parents=True,exist_ok=True)
    specs=[("figure1_write_latency","write_p99_ms"),("figure2_read_latency","read_p99_ms"),("figure3_retrievability_lag","visibility_upper_bound_ms"),("figure4_cost","tokens"),("figure5_scale","accuracy")]
    for name,key in specs:
      fig,ax=plt.subplots(); valid=[r for r in rows if r.get(key) is not None]; ax.plot([r.get("scale",r.get("load",i)) for i,r in enumerate(valid)],[r[key] for r in valid],marker='o'); ax.set_ylabel(key); fig.savefig(out/f"{name}.pdf",bbox_inches='tight'); fig.savefig(out/f"{name}.png",dpi=200,bbox_inches='tight'); plt.close(fig)
