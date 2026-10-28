import numpy as np
import pytest

from darm_vol import DARM


def test_numba_matches_numpy_if_available():
    pytest.importorskip("numba")
    x = np.random.default_rng(1).normal(0, .01, 100)
    p = {"omega": .00002, "rho": .8, "alpha": 2., "delta": 2.}
    np_model = DARM(lags=30, distribution="normal")
    jit_model = DARM(lags=30, distribution="normal", use_numba=True)
    np.testing.assert_allclose(np_model.conditional_variance(x, p)[30:],
                               jit_model.conditional_variance(x, p)[30:], rtol=1e-13)


def test_plotting_does_not_force_show():
    plt = pytest.importorskip("matplotlib.pyplot")
    from darm_vol.plotting import simulation_plot
    p = {"omega": .00002, "rho": .8, "alpha": 2., "delta": 2.}
    model = DARM(lags=3, distribution="normal")
    s = model.simulate(p, nobs=15, random_state=19)
    fig, ax = simulation_plot(s)
    assert ax is not None
    plt.close(fig)
