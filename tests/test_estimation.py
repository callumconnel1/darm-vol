import numpy as np
import pytest

from darm_vol import DARM


def test_fitting_improves_likelihood_and_information_criteria():
    m = DARM(lags=4, distribution="normal")
    p = {"omega": .00006, "rho": .65, "alpha": 1.8, "delta": 2.}
    x = m.simulate(p, nobs=420, burn=300, random_state=501).returns
    fit = m.fit(x, n_starts=1, maxiter=90)
    assert fit.converged
    assert set(fit.params) == set(p)
    assert fit.nobs == len(x)-m.lags
    assert fit.aic == pytest.approx(2*fit.nparams-2*fit.loglikelihood)
    assert fit.bic == pytest.approx(fit.nparams*np.log(fit.nobs)-2*fit.loglikelihood)
    assert np.isnan(fit.conditional_variance[:4]).all()
    assert np.all(np.isfinite(fit.conditional_variance[4:]))
    assert fit.forecast(5).integrated_variance > 0
    assert "Converged: True" in fit.summary()
    assert len(fit.optimizer_runs) == 1


def test_failed_optimizer_requires_optin():
    m = DARM(lags=3, distribution="normal")
    p = {"omega": .00006, "rho": .65, "alpha": 1.8, "delta": 2.}
    x = m.simulate(p, nobs=60, burn=200, random_state=19).returns
    with pytest.raises(RuntimeError, match="did not converge"):
        m.fit(x, maxiter=1, n_starts=1)
    result = m.fit(x, maxiter=1, n_starts=1, allow_failed=True)
    assert not result.converged
