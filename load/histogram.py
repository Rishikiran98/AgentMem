"""Exact quantiles over raw samples; raw traces remain authoritative."""
from __future__ import annotations
import math

def quantile(values: list[float], q: float) -> float | None:
    if not values: return None
    xs=sorted(values); pos=(len(xs)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi]-xs[lo])*(pos-lo)

def summary(values: list[float]) -> dict:
    out={"n":len(values)}
    for label,q in (("p50",.5),("p95",.95),("p99",.99),("p99_9",.999)):
        out[label]=quantile(values,q) if q < .99 or len(values)>=100 else None
    out["quantile_warning"] = "p99 requires N>=100" if len(values)<100 else None
    return out
