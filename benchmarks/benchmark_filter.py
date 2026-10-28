"""Repeatable wall-clock microbenchmark of the finite-K variance recursion.

No claims are made about the original research implementations.
Run: python benchmarks/benchmark_filter.py
"""
from __future__ import annotations

import time
import numpy as np
from darm_vol import DARM


def benchmark(n: int, k: int, repeats: int = 3) -> float:
    rng = np.random.default_rng(20261028)
    returns = rng.normal(0, .01, n)
    p = {"omega": .00001, "rho": .8, "alpha": 2., "delta": 5.}
    model = DARM(kernel="power", distribution="normal", lags=k)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.conditional_variance(returns, p)
        times.append(time.perf_counter()-start)
    return min(times)


if __name__ == "__main__":
    for k in (30, 252, 1000):
        for n in (1000, 5000, 10000):
            if n > k:
                print(f"T={n:5d} K={k:4d}: {benchmark(n,k):.6f} sec (best of three)")
