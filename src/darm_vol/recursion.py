"""Exact finite-K ARCH recursion conditional on the first K observed returns."""
from __future__ import annotations

import numpy as np


def variance_recursion(transformed: np.ndarray, omega: float, rho: float,
                       weights: np.ndarray, *, use_numba: bool = False) -> np.ndarray:
    """Return h_t for t>=K; h_0..h_(K-1) are undefined (NaN).

    No artificial presample shocks are used when filtering a real data series.
    Parameters and input validation are responsibilities of calling code.
    """
    n, k = transformed.size, weights.size
    if use_numba:
        try:
            return _numba_recursion(transformed, omega, rho, weights)
        except ImportError as exc:
            raise ImportError("Numba is optional: pip install 'darm-vol[performance]'") from exc
    h = np.full(n, np.nan)
    for t in range(k, n):
        h[t] = omega + rho * np.dot(weights, transformed[t-k:t][::-1])
    return h


_NUMBA_FILTER = None


def _numba_recursion(u: np.ndarray, omega: float, rho: float,
                     weights: np.ndarray) -> np.ndarray:
    """Create the optional JIT-compiled function once and reuse it during MLE."""
    global _NUMBA_FILTER
    if _NUMBA_FILTER is None:
        from numba import njit

        @njit
        def _filter(values: np.ndarray, baseline: float, persistence: float,
                    w: np.ndarray) -> np.ndarray:
            n, k = values.size, w.size
            h = np.full(n, np.nan)
            for t in range(k, n):
                s = 0.0
                for j in range(k):
                    s += w[j] * values[t-j-1]
                h[t] = baseline + persistence * s
            return h
        _NUMBA_FILTER = _filter

    return _NUMBA_FILTER(u, omega, rho, weights)
