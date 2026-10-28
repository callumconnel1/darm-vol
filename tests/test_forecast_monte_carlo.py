"""Verify future transformed shocks are replaced by conditional expectations."""

import numpy as np
from darm_vol.forecasting import variance_forecast
from darm_vol.kernels import kernel_weights


def test_multistep_against_conditional_monte_carlo():
    rng = np.random.default_rng(42)
    p = {"omega": 0.1, "rho": .7, "alpha": 2.0, "delta": 1.5, "kappa": .65}
    history = np.array([.2, -.8, .4, -.3])
    w = kernel_weights("power", 4, p)
    forecast = variance_forecast(history, p, "power", 4, True, 3).daily_variance
    n = 30000
    # Simulate from exactly the same last-four observed shocks; no stochastic
    # initialization or further presample observations enter these horizons.
    initial_u = history**2*(1-p["kappa"]+2*p["kappa"]*(history<0))
    future = np.empty((n,3))
    for i in range(n):
        hist = list(initial_u)
        for j in range(3):
            h = p["omega"] + p["rho"]*sum(w[k-1]*hist[-k] for k in range(1,5))
            future[i,j] = h
            z = rng.standard_normal()
            hist.append(h*z*z*(1-p["kappa"]+2*p["kappa"]*(z<0)))
    np.testing.assert_allclose(future.mean(axis=0), forecast, rtol=.03)
