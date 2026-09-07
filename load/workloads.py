from __future__ import annotations
import random

def operation_schedule(n:int, read_ratio:float, seed:int)->list[str]:
    if not 0 <= read_ratio <= 1: raise ValueError("read_ratio must be in [0,1]")
    rng=random.Random(seed); return ["read" if rng.random()<read_ratio else "write" for _ in range(n)]
