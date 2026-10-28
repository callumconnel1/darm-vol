"""Optional plotting submodule, imported lazily by results methods."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .forecasting import Forecast
    from .results import DARMResults


def _axes():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires: pip install 'darm-vol[plot]'") from exc
    return plt.subplots()


def volatility_plot(fit: DARMResults):
    """Plot h_t**0.5 and return matplotlib Figure, Axes."""
    fig, ax = _axes()
    ax.plot(fit.conditional_volatility)
    ax.set(xlabel="Observation index", ylabel="Conditional volatility", title="DARM volatility")
    return fig, ax


def residual_plot(fit: DARMResults):
    """Plot filtered, standardized innovations."""
    fig, ax = _axes()
    ax.plot(fit.standardized_residuals)
    ax.set(xlabel="Observation index", ylabel="Standardized residual", title="DARM residuals")
    return fig, ax


def kernel_plot(fit: DARMResults):
    """Plot the finite K normalized kernel weights."""
    fig, ax = _axes()
    ax.plot(np.arange(1, fit.model.lags + 1), fit.model.kernel_weights(fit.params))
    ax.set(xlabel="Lag k", ylabel="Weight", title="DARM memory kernel")
    return fig, ax


def forecast_plot(prediction: Forecast):
    """Plot conditional expected future daily variances."""
    fig, ax = _axes()
    ax.plot(np.arange(1, len(prediction.daily_variance) + 1), prediction.daily_variance)
    ax.set(xlabel="Forecast horizon", ylabel="Expected conditional variance", title="DARM forecast")
    return fig, ax


def simulation_plot(simulation):
    """Plot a finite-K generated sample; return (fig, ax)."""
    fig, ax = _axes()
    ax.plot(simulation.innovations)
    ax.set(xlabel="Observation index", ylabel="Simulated innovations", title="DARM simulation")
    return fig, ax
