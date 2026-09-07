from __future__ import annotations

def pareto(rows, x="p99_ms", y="accuracy"):
    return [r for r in rows if not any((o[y]>=r[y] and o[x]<=r[x]) and (o[y]>r[y] or o[x]<r[x]) for o in rows)]
def constrained_rank(rows, *, max_p99_ms=None, max_tokens=None):
    ok=[r for r in rows if (max_p99_ms is None or r["p99_ms"]<=max_p99_ms) and (max_tokens is None or r["tokens"]<=max_tokens)]
    return sorted(ok,key=lambda r:(-r["accuracy"],r["p99_ms"]))
def inversion(rows,**constraints):
    base=constrained_rank(rows); constrained=constrained_rank(rows,**constraints)
    return {"observed":bool(base and constrained and base[0]["system"]!=constrained[0]["system"]),"accuracy_winner":base[0]["system"] if base else None,"constrained_winner":constrained[0]["system"] if constrained else None}
