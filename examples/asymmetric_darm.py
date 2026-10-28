"""Effect of asymmetric transformed-shock kernel weights."""
from darm_vol import DARM

m = DARM(lags=10, asymmetric=True, distribution="normal")
p = {"omega": .0001, "rho": .6, "alpha": 2., "delta": 2., "kappa": .6}
x = m.simulate(p, nobs=500, random_state=42)
print("Sample variance:", x.returns.var())
print("Model's long-run mean h:", p["omega"]/(1-p["rho"]))
