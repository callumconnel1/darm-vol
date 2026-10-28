"""Exact future h recursion using expected transformed shocks."""
from darm_vol import DARM

m = DARM(kernel="power", distribution="normal", lags=15)
p = {"omega": .0001, "rho": .8, "alpha": 2., "delta": 1.}
x = m.simulate(p, nobs=600, random_state=90)
fit = m.fit(x.returns, n_starts=1)
pred = fit.forecast(horizon=20)
print("Daily conditional variance forecasts:", pred.daily_variance)
print("20-day expected integrated variance:", pred.integrated_variance)
