"""Diagnostics for standardized model residuals."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, kurtosis


def ljung_box(series: np.ndarray, lags: int = 10) -> dict[str, float]:
    """Ljung-Box test at a single lag; asymptotic reference, no fit correction."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if lags < 1 or lags >= n:
        raise ValueError("lags must lie between 1 and n-1")
    x = x - x.mean()
    norm = float(np.dot(x, x))
    if norm <= 0:
        raise ValueError("cannot test a constant residual sequence")
    acf = np.asarray([np.dot(x[k:], x[:-k]) / norm for k in range(1, lags + 1)])
    q = float(n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags+1))))
    return {"statistic": q, "pvalue": float(chi2.sf(q, lags)), "lags": lags}


def residual_diagnostics(standardized: np.ndarray, lags: int = 10) -> dict:
    """Describe standardized residuals and their squares (informal checks)."""
    z = np.asarray(standardized, dtype=float)
    z = z[np.isfinite(z)]
    return {"nobs": int(z.size), "mean": float(np.mean(z)),
            "variance": float(np.var(z)), "excess_kurtosis": float(kurtosis(z)),
            "ljung_box": ljung_box(z, lags),
            "squared_ljung_box": ljung_box(z**2, lags)}
