"""Fit all three kernel families to common simulated observations."""
from darm_vol import DARM

source = DARM(lags=20, distribution="normal")
params = {"omega": .00008, "rho": .7, "alpha": 2., "delta": 3.}
x = source.simulate(params, nobs=900, random_state=3).returns
for name in ("exponential", "power", "hyperbolic"):
    fit = DARM(kernel=name, distribution="normal", lags=20).fit(x, n_starts=1)
    print(name, fit.aic, fit.params)
