import numpy as np
import pytest
from darm_vol import DARM


PARAMS = {"omega": .00002, "rho": .7, "alpha": 1.8, "delta": 3.}


def test_validation():
    with pytest.raises(ValueError):
        DARM(lags=0)
    with pytest.raises(ValueError):
        DARM(distribution="cauchy")
    with pytest.raises(ValueError):
        DARM().fit([.1, .2])
    model = DARM(lags=3, distribution="normal")
    with pytest.raises(ValueError):
        model.conditional_variance([.01, np.nan, .01, .02], PARAMS)
    with pytest.raises(ValueError):
        model.loglikelihood(np.ones(10), {**PARAMS, "rho": 1.})


def test_complete_lag_reference():
    data = np.array([.01, -.02, .03, -.04, .05, -.06])
    model = DARM(lags=3, distribution="normal")
    h = model.conditional_variance(data, PARAMS)
    w = model.kernel_weights(PARAMS)
    reference = np.full(len(data), np.nan)
    for t in range(3, len(data)):
        reference[t] = PARAMS["omega"] + PARAMS["rho"] * sum(
            w[k-1] * data[t-k]**2 for k in range(1, 4))
    np.testing.assert_allclose(h[3:], reference[3:])
    assert np.all(np.isnan(h[:3]))


def test_asymmetric_reduces_to_symmetric():
    data = np.array([.02, -.04, .03, .01, -.02, .01])
    sym = DARM(lags=3, distribution="normal")
    asym = DARM(lags=3, distribution="normal", asymmetric=True)
    np.testing.assert_array_equal(sym.conditional_variance(data, PARAMS)[3:],
                                  asym.conditional_variance(data, {**PARAMS, "kappa": 0})[3:])


def test_asymmetric_negative_has_greater_effect():
    model = DARM(lags=1, asymmetric=True, distribution="normal")
    p = {"omega": .02, "rho": .7, "alpha": 1, "delta": 2, "kappa": .5}
    neg = model.conditional_variance([-.2, 0.], p)[1]
    pos = model.conditional_variance([.2, 0.], p)[1]
    assert (neg - p["omega"]) / (pos - p["omega"]) == pytest.approx(3.0)
