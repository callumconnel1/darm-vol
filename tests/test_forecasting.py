import numpy as np
import pytest

from darm_vol import DARM
from darm_vol.forecasting import variance_forecast


def test_forecast_k1_recursion_exact():
    params = {"omega": .2, "rho": .6, "beta": .7}
    x = np.array([.2, -.1, .4, .5])
    prediction = variance_forecast(x, params, "exponential", 1, False, 4)
    expected = [params["omega"] + params["rho"] * x[-1]**2]
    for _ in range(3):
        expected.append(params["omega"] + params["rho"] * expected[-1])
    np.testing.assert_allclose(prediction.daily_variance, expected)
    assert prediction.integrated_variance == pytest.approx(sum(expected))


def test_multistep_independent_reference():
    params = {"omega": .0002, "rho": .78, "alpha": 2., "delta": 1.}
    model = DARM(lags=3, distribution="normal")
    x = np.array([.1, -.3, .2, -.2, .4])
    forecast = variance_forecast(x, params, "power", 3, False, 5)
    hist = list(x**2)
    w = model.kernel_weights(params)
    expected = []
    for _ in range(5):
        h = params["omega"] + params["rho"] * sum(
            w[i-1] * hist[-i] for i in range(1, 4))
        expected.append(h)
        hist.append(h)
    np.testing.assert_allclose(forecast.daily_variance, expected)


def test_forecast_only_uses_past():
    p = {"omega": .001, "rho": .7, "alpha": 2., "delta": 2.}
    def f(x):
        return variance_forecast(
            np.asarray(x), p, "power", 2, False, 2
        ).daily_variance
    np.testing.assert_array_equal(f([.1, .2, .3]), f([.8, .2, .3]))
    with pytest.raises(ValueError):
        variance_forecast(np.array([.1, .2]), p, "power", 2, False, 0)
