"""Multi-horizon finite-state conditional mean forecasts of h_t."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from .kernels import kernel_weights
from .validation import transformed_shocks


@dataclass(frozen=True)
class Forecast:
    """Daily forecast E_T[h_(T+j)] and cumulative expected integrated variance."""

    daily_variance: np.ndarray

    @property
    def daily_volatility(self) -> np.ndarray:
        """Square root of *expected* conditional variance, not E[sqrt(h)]."""
        return np.sqrt(self.daily_variance)

    @property
    def integrated_variance(self) -> float:
        """Sum of E_T[h] over the requested horizon (not a confidence interval)."""
        return float(self.daily_variance.sum())


def variance_forecast(eps: np.ndarray, params: Mapping[str, float],
                      kernel: str, lags: int, asymmetric: bool, horizon: int) -> Forecast:
    """Forecast replacing future transformed shocks by E[u|F_{t-1}]=h.

    Supported innovation laws are symmetric, hence E[z^2(1-k+2k1[z<0])]=1.
    The latest K realised transformed shocks provide the initial state.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if eps.size < lags:
        raise ValueError("At least K observed shocks required for forecasts")
    w = kernel_weights(kernel, lags, params)
    history = list(transformed_shocks(eps[-lags:],
                   float(params["kappa"]) if asymmetric else 0.0))
    omega, rho = float(params["omega"]), float(params["rho"])
    daily = np.empty(horizon)
    for t in range(horizon):
        expected_h = omega + rho * np.dot(w, np.asarray(history[-lags:][::-1]))
        daily[t] = expected_h
        history.append(expected_h)
    return Forecast(daily)
