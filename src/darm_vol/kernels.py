"""Finite, exactly normalised DARM memory kernels (Connell, 2026, eqs. 10-12)."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import logsumexp

PARAMETERS = {"exponential": ("beta",), "power": ("alpha", "delta"),
              "hyperbolic": ("s", "p")}
ALIASES = {"exp": "exponential", "power-law": "power", "hyperbolic": "hyperbolic",
           "exponential": "exponential", "power": "power"}


def canonical_kernel(name: str) -> str:
    """Translate documented kernel aliases into canonical identifiers."""
    if not isinstance(name, str) or name.lower() not in ALIASES:
        raise ValueError(f"Unknown kernel {name!r}; choose exponential, power, hyperbolic")
    return ALIASES[name.lower()]


def kernel_weights(kernel: str, lags: int, params: Mapping[str, float]) -> np.ndarray:
    """Calculate K strictly positive weights summing to one.

    These are finite-K *renormalised* kernels. No assertion of equivalence
    to an infinite-lag process is made. Log-sum-exp prevents underflow of
    the normalisation constant at large decay parameters.
    """
    kernel = canonical_kernel(kernel)
    if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or lags < 1:
        raise ValueError("lags must be a positive integer")
    names = PARAMETERS[kernel]
    values = []
    for name in names:
        if name not in params:
            raise ValueError(f"Missing kernel parameter {name!r}")
        value = float(params[name])
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        values.append(value)
    k = np.arange(1, lags + 1, dtype=float)
    if kernel == "exponential":
        (beta,) = values
        if not 0 < beta < 1:
            raise ValueError("exponential beta must lie strictly between 0 and 1")
        logw = (k - 1) * np.log(beta)
    elif kernel == "power":
        alpha, delta = values
        if alpha <= 0 or delta <= 0:
            raise ValueError("power alpha and delta must be positive")
        logw = -alpha * np.log(k - 1 + delta)
    else:
        s, p = values
        if s <= 0 or p <= 0:
            raise ValueError("hyperbolic s and p must be positive")
        logw = -p * np.log1p(s * k)
    w = np.exp(logw - logsumexp(logw))
    # A representable positive weight is not guaranteed at arbitrarily large shape parameters.
    if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() == 0:
        raise ValueError("Kernel weights cannot be represented numerically")
    w /= w.sum()
    return w
