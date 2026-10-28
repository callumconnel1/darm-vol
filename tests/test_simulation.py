import numpy as np

from darm_vol import DARM


def test_simulation_reproducible_and_positive():
    m = DARM(lags=2, distribution="student-t", asymmetric=True)
    p = {"omega": .0001, "rho": .6, "alpha": 2., "delta": 2., "kappa": .4, "nu": 8.}
    a = m.simulate(p, nobs=4500, burn=800, random_state=2026)
    b = m.simulate(p, nobs=4500, burn=800, random_state=2026)
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.conditional_variance, b.conditional_variance)
    assert np.all(a.conditional_variance > 0)
    np.testing.assert_allclose(a.returns / np.sqrt(a.conditional_variance), a.standardized_shocks)
    assert abs(np.mean(a.standardized_shocks**2) - 1.) < .1
    assert abs(np.mean(a.conditional_variance) / (p["omega"]/(1-p["rho"])) - 1.) < .15


def test_simulated_recursion_after_burn():
    m = DARM(lags=3, distribution="normal")
    p = {"omega": 1e-4, "rho": .7, "alpha": 2., "delta": 1.}
    s = m.simulate(p, nobs=100, burn=300, random_state=7)
    filtered = m.conditional_variance(s.returns, p)
    np.testing.assert_allclose(filtered[3:], s.conditional_variance[3:], rtol=1e-12)
