"""End-to-end multi-parameter fitting path for asymmetric Student-t DARM."""

import numpy as np
from darm_vol import DARM


def test_student_t_asymmetric_fit_end_to_end():
    model = DARM(kernel="power", distribution="student-t", lags=3, asymmetric=True)
    params = {"omega": .0001, "rho": .5, "alpha": 2., "delta": 2.,
              "kappa": .4, "nu": 9.}
    sample = model.simulate(params, nobs=250, burn=300, random_state=100).returns
    result = model.fit(sample, n_starts=1, maxiter=300)
    assert result.converged
    assert result.params["nu"] > 2
    assert 0 <= result.params["kappa"] < 1
    assert result.nparams == 6
    assert result.forecast(10).daily_variance.shape == (10,)
    assert np.all(np.isfinite(result.standardized_residuals[3:]))
    diagnostics = result.diagnostics(lags=5)
    assert diagnostics["nobs"] == 247
