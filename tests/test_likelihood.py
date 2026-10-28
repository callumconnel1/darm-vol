import numpy as np
import pytest
from scipy import stats

from darm_vol import DARM
from darm_vol.distributions import gaussian_logpdf, standardized_t_logpdf


def test_standardized_t_against_scipy():
    z = np.array([-5., -1.5, 0., 2., 5.])
    nu = 7.4
    s = np.sqrt((nu - 2) / nu)
    np.testing.assert_allclose(standardized_t_logpdf(z, nu),
                               stats.t.logpdf(z / s, nu) - np.log(s), rtol=1e-13)
    with pytest.raises(ValueError):
        standardized_t_logpdf(z, 2.)


def test_gaussian_against_scipy():
    z = np.array([-3., 0., .4, 3.])
    np.testing.assert_allclose(gaussian_logpdf(z), stats.norm.logpdf(z))


def test_conditional_likelihood_reference():
    model = DARM(lags=2, distribution="student-t")
    x = np.array([-.05, .01, -.03, .02, .01])
    p = {"omega": .0003, "rho": .75, "alpha": 2.1, "delta": .9, "nu": 8.}
    h = model.conditional_variance(x, p)
    s = np.sqrt((p["nu"]-2)/p["nu"])
    reference = sum(stats.t.logpdf(x[t]/np.sqrt(h[t])/s, p["nu"])
                    - np.log(s) - .5*np.log(h[t]) for t in range(2, len(x)))
    assert model.loglikelihood(x, p) == pytest.approx(reference)
