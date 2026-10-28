"""Finite-K simulations with a documented deterministic presample convention."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .distributions import draw_shocks
from .kernels import kernel_weights
from .validation import check_variance_params, transformed_shocks


@dataclass(frozen=True)
class Simulation:
    """Simulated conditional variances, returns/shocks and standardised innovations."""
    innovations: np.ndarray
    conditional_variance: np.ndarray
    standardized_shocks: np.ndarray

    @property
    def conditional_volatility(self) -> np.ndarray:
        """Square root of simulated conditional variance."""
        return np.sqrt(self.conditional_variance)

    @property
    def returns(self) -> np.ndarray:
        """Alias for zero-mean simulated returns epsilon_t."""
        return self.innovations


def simulate_model(params: Mapping[str, float], kernel: str, distribution: str,
                   lags: int, asymmetric: bool, nobs: int, burn: int = 1000,
                   random_state: int | np.random.Generator | None = None) -> Simulation:
    """Generate nobs observations after burn-in from finite-K DARM.

    The K presample transformed shocks are set to E[h] = omega/(1-rho),
    valid for symmetric standardized innovation distributions. The first K
    shock values are NOT passed through the fitting likelihood.
    """
    for value, label, min_value in ((nobs, "nobs", 1), (burn, "burn", 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < min_value:
            raise ValueError(f"{label} must be an integer >= {min_value}")
    check_variance_params(params, asymmetric, distribution)
    w = kernel_weights(kernel, lags, params)
    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))
    n = nobs + burn
    shocks = draw_shocks(rng, n, distribution, params.get("nu"))
    h = np.empty(n)
    eps = np.empty(n)
    u = np.full(lags + n, float(params["omega"]) / (1 - float(params["rho"])))
    omega, rho = float(params["omega"]), float(params["rho"])
    kappa = float(params["kappa"]) if asymmetric else 0.0
    for t in range(n):
        h[t] = omega + rho * np.dot(w, u[t:t + lags][::-1])
        eps[t] = np.sqrt(h[t]) * shocks[t]
        u[lags + t] = transformed_shocks(np.asarray([eps[t]]), kappa)[0]
    return Simulation(eps[burn:].copy(), h[burn:].copy(), shocks[burn:].copy())
