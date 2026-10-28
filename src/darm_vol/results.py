"""Immutable-in-convention fitted results, forecasting and diagnostics."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
import numpy as np

from .diagnostics import residual_diagnostics
from .estimation import OptimizerReport
from .forecasting import Forecast, variance_forecast
from .simulation import Simulation

if TYPE_CHECKING:
    from .model import DARM


class DARMResults:
    """Conditional ML fit with residuals, forecasts and optimizer diagnostics.

    Standard errors are intentionally absent: parameter-specific profile
    inference is recommended because kernel geometry can be weakly identified.
    """

    def __init__(self, model: DARM, params: Mapping[str, float], eps: np.ndarray,
                 optimization: OptimizerReport, optimizer_runs: list[OptimizerReport]) -> None:
        self.model = model
        self.params = dict(params)
        self.residuals = eps.copy()
        self.optimization = optimization
        self.optimizer_runs = list(optimizer_runs)
        self.converged = bool(optimization.success)
        self.loglikelihood = float(optimization.loglikelihood)
        self.nobs = eps.size - model.lags
        self.nparams = len(params)
        self.aic = 2 * self.nparams - 2 * self.loglikelihood
        self.bic = self.nparams * np.log(self.nobs) - 2 * self.loglikelihood
        self.conditional_variance = model.conditional_variance(eps, params)
        self.conditional_volatility = np.sqrt(self.conditional_variance)
        self.standardized_residuals = np.full(eps.size, np.nan)
        self.standardized_residuals[model.lags:] = (
            eps[model.lags:] / self.conditional_volatility[model.lags:])

    def forecast(self, horizon: int = 1) -> Forecast:
        """Forecast individual expected h and their finite-horizon sum."""
        return variance_forecast(self.residuals, self.params, self.model.kernel,
                                 self.model.lags, self.model.asymmetric, horizon)

    def simulate(self, nobs: int = 5000, *, burn: int = 1000,
                 random_state: int | np.random.Generator | None = None) -> Simulation:
        """Simulate from the fitted finite-K process."""
        return self.model.simulate(self.params, nobs, burn=burn, random_state=random_state)

    def diagnostics(self, lags: int = 10) -> dict:
        """Report standardized-residual moments and Ljung-Box diagnostics."""
        return residual_diagnostics(self.standardized_residuals, lags=lags)

    def summary(self) -> str:
        """Readable overview of estimated parameters and conditional likelihood."""
        lines = ["Decaying AutoRegressive Memory (DARM)", "=" * 44,
                 f"Kernel: {self.model.kernel}  Distribution: {self.model.distribution}",
                 f"Lags K: {self.model.lags}  Asymmetric: {self.model.asymmetric}",
                 f"Conditional observations (T-K): {self.nobs}",
                 f"Converged: {self.converged}",
                 f"Log likelihood: {self.loglikelihood:.6f}",
                 f"AIC: {self.aic:.6f}  BIC: {self.bic:.6f}",
                 "Parameters:"]
        lines += [f"  {name:>8}: {value:.8g}" for name, value in self.params.items()]
        lines.append(f"Optimizer: {self.optimization.message}")
        lines.append("Warning: finite-K conditioned likelihood; no automatic standard errors.")
        return "\n".join(lines)

    def plot_volatility(self):
        """Plot fitted conditional volatility; return (fig, ax)."""
        from .plotting import volatility_plot
        return volatility_plot(self)

    def plot_residuals(self):
        """Plot standardized residuals; return (fig, ax)."""
        from .plotting import residual_plot
        return residual_plot(self)

    def plot_kernel(self):
        """Plot finite kernel weights; return (fig, ax)."""
        from .plotting import kernel_plot
        return kernel_plot(self)

    def plot_forecast(self, horizon: int = 10):
        """Plot individual expected daily variances; return (fig, ax)."""
        from .plotting import forecast_plot
        return forecast_plot(self.forecast(horizon))
