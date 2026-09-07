"""Deterministic open-loop arrival schedules."""
from __future__ import annotations
import random


def poisson_arrivals(rate_rps: float, duration_s: float, seed: int) -> list[float]:
    if rate_rps <= 0 or duration_s < 0:
        raise ValueError("rate_rps must be positive and duration_s non-negative")
    rng = random.Random(seed); t = 0.0; out = []
    while True:
        t += rng.expovariate(rate_rps)
        if t >= duration_s: return out
        out.append(t)
