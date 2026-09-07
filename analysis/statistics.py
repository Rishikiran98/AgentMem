"""Statistical helpers.  Pure functions over counts and arrays; no I/O."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    n: int
    method: str

    def as_dict(self) -> dict:
        return {"point": self.point, "low": self.low, "high": self.high, "n": self.n, "method": self.method}


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "wilson")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), n, "wilson")


def bootstrap_mean_interval(values: Sequence[float], *, iters: int = 2000, seed: int = 0, alpha: float = 0.05) -> Interval:
    vals = list(values)
    if not vals:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "bootstrap")
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(iters))
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return Interval(sum(vals) / n, lo, hi, n, "bootstrap_percentile")


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> dict:
    """Two-sided z-test for equal proportions (pooled)."""
    if n1 == 0 or n2 == 0:
        return {"z": float("nan"), "p_value": float("nan")}
    p1, p2 = s1 / n1, s2 / n2
    pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0}
    z = (p1 - p2) / se
    p = 2 * (1 - _norm_cdf(abs(z)))
    return {"z": z, "p_value": p}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 100]); None for empty input."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, math.ceil(q / 100.0 * len(vals)) - 1))
    return vals[k]


def cluster_bootstrap(clusters: dict[str, list[float]], statistic, *, seed: int, samples: int = 2000) -> tuple[float, float]:
    """95% interval of ``statistic`` under resampling of whole clusters (e.g. questions), seeded."""
    if not clusters:
        raise ValueError("clusters required")
    rng = random.Random(seed)
    keys = list(clusters)
    vals = []
    for _ in range(samples):
        drawn = [v for _ in keys for v in clusters[rng.choice(keys)]]
        vals.append(statistic(drawn))
    vals.sort()
    return vals[int(0.025 * samples)], vals[min(samples - 1, int(0.975 * samples))]


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired per-item outcomes of two systems."""
    if len(a) != len(b):
        raise ValueError("paired outcomes required")
    b01 = sum((not x) and y for x, y in zip(a, b))
    b10 = sum(x and (not y) for x, y in zip(a, b))
    n = b01 + b10
    tail = sum(math.comb(n, k) for k in range(0, min(b01, b10) + 1)) / (2 ** n) if n else 1
    return {"discordant_01": b01, "discordant_10": b10, "p_value": min(1, 2 * tail), "accuracy_delta": sum(b) / len(b) - sum(a) / len(a) if a else None}
