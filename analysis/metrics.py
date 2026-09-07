from __future__ import annotations
import math
from load.histogram import quantile

def latency(values:list[float])->dict:
    xs=sorted(values); q1=quantile(xs,.25); q3=quantile(xs,.75)
    return {"n":len(xs),"median_ms":quantile(xs,.5),"iqr_ms":None if q1 is None else q3-q1,"p95_ms":quantile(xs,.95),"p99_ms":quantile(xs,.99) if len(xs)>=100 else None}

def accuracy(correct:list[bool])->dict:
    n=len(correct)
    if not n:return {"n":0,"accuracy":None,"ci95":[None,None]}
    p=sum(correct)/n; z=1.959963984540054; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {"n":n,"accuracy":p,"ci95":[max(0,center-half),min(1,center+half)]}

def visibility_intervals(events:list[dict])->list[tuple[float,float]]:
    return [(e["visibility_lower_bound_ms"],e["visibility_upper_bound_ms"]) for e in events if e.get("settled") and e.get("visibility_upper_bound_ms") is not None]
