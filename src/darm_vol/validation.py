"""Data and parameter validation; inputs are never silently rescaled."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def returns_array(data: object, *, minimum: int = 1) -> np.ndarray:
    """Validate a one-dimensional numeric NumPy array or pandas Series."""
    try:
        x = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("returns must be numeric and one-dimensional") from exc
    if x.ndim != 1 or x.size < minimum:
        raise ValueError(f"returns must be 1D with at least {minimum} observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain NaN or infinite observations; clean data explicitly")
    return np.ascontiguousarray(x)


def check_variance_params(params: Mapping[str, float], asymmetric: bool,
                          distribution: str) -> None:
    """Validate model coefficients without coercing out-of-domain parameters."""
    try:
        omega = float(params["omega"])
        rho = float(params["rho"])
    except (KeyError, TypeError) as exc:
        raise ValueError("omega and rho are required") from exc
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega must be positive and finite")
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must lie in [0,1)")
    if asymmetric:
        if "kappa" not in params:
            raise ValueError("kappa required for asymmetric DARM")
        kappa = float(params["kappa"])
        if not np.isfinite(kappa) or not 0 <= kappa < 1:
            raise ValueError("kappa must lie in [0,1)")
    if distribution == "student-t":
        if "nu" not in params:
            raise ValueError("nu required for Student-t DARM")
        nu = float(params["nu"])
        if not np.isfinite(nu) or nu <= 2:
            raise ValueError("nu must exceed 2")


def transformed_shocks(eps: np.ndarray, kappa: float = 0.0) -> np.ndarray:
    """u_t=eps_t^2*(1-kappa+2*kappa*1[eps_t<0])."""
    eps = np.asarray(eps, dtype=float)
    return np.square(eps) * (1 - kappa + 2 * kappa * (eps < 0))
