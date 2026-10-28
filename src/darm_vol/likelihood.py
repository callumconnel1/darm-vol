"""Gaussian and standardised Student-t finite-K conditional likelihoods."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .distributions import gaussian_logpdf, standardized_t_logpdf
from .kernels import kernel_weights
from .recursion import variance_recursion
from .validation import check_variance_params, transformed_shocks


def conditional_variance(eps: np.ndarray, params: Mapping[str, float],
                         kernel: str, asymmetric: bool, distribution: str, lags: int,
                         *, use_numba: bool = False) -> np.ndarray:
    """Filter conditional variance, leaving first K values as NaN."""
    check_variance_params(params, asymmetric, distribution)
    weights = kernel_weights(kernel, lags, params)
    u = transformed_shocks(eps, float(params["kappa"]) if asymmetric else 0.0)
    return variance_recursion(u, float(params["omega"]), float(params["rho"]),
                              weights, use_numba=use_numba)


def log_likelihood(eps: np.ndarray, params: Mapping[str, float], kernel: str,
                   asymmetric: bool, distribution: str, lags: int,
                   *, use_numba: bool = False) -> float:
    """Return sum of conditional log-likelihood contributions over t=K,...,T-1."""
    if len(eps) <= lags:
        raise ValueError("Sample must contain more observations than memory length K")
    h = conditional_variance(eps, params, kernel, asymmetric, distribution, lags,
                             use_numba=use_numba)[lags:]
    if np.any(~np.isfinite(h)) or np.any(h <= 0):
        raise FloatingPointError("Invalid conditional variances")
    z = eps[lags:] / np.sqrt(h)
    if distribution == "normal":
        logp = gaussian_logpdf(z)
    elif distribution == "student-t":
        logp = standardized_t_logpdf(z, float(params["nu"]))
    else:
        raise ValueError("Unknown innovation distribution")
    ll = np.sum(logp - 0.5 * np.log(h))
    if not np.isfinite(ll):
        raise FloatingPointError("Non-finite log-likelihood")
    return float(ll)
