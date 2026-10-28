"""Self-contained basic usage (no internet or finance feed needed)."""
from darm_vol import DARM

model = DARM(kernel="power", distribution="student-t", lags=30, asymmetric=True)
true = {"omega": 0.00002, "rho": 0.70, "alpha": 1.8,
        "delta": 5., "kappa": .4, "nu": 8.}
returns = model.simulate(true, nobs=1100, burn=500, random_state=42).returns
result = model.fit(returns, n_starts=2)
print(result.summary())
print("5-day integrated conditional variance:", result.forecast(5).integrated_variance)
print("Simulated returns:", result.simulate(nobs=5, random_state=7).returns)
