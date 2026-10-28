import numpy as np
import pytest
from darm_vol import kernel_weights


@pytest.mark.parametrize("name,params", [
    ("exponential", {"beta": .9}),
    ("power", {"alpha": 1.8, "delta": 5}),
    ("hyperbolic", {"s": .3, "p": 2.2}),
])
def test_normalized(name, params):
    w = kernel_weights(name, 250, params)
    assert w.shape == (250,)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1., abs=1e-14)
    assert np.all(np.diff(w) < 0)


def test_hyperbolic_is_shifted_power():
    s, p = .4, 2.3
    a = kernel_weights("hyperbolic", 88, {"s": s, "p": p})
    b = kernel_weights("power", 88, {"alpha": p, "delta": 1 + 1 / s})
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-16)


def test_exponential_geometric():
    w = kernel_weights("exponential", 4, {"beta": .5})
    np.testing.assert_allclose(w, np.array([8, 4, 2, 1])/15)


def test_boundary_and_small_k():
    assert kernel_weights("power", 1, {"alpha": 1.8, "delta": .01})[0] == 1.0
    assert np.all(kernel_weights("power", 250, {"alpha": 9, "delta": .001}) >= 0)
    with pytest.raises(ValueError):
        kernel_weights("power", 4, {"alpha": 0, "delta": 2})
    with pytest.raises(ValueError):
        kernel_weights("exponential", 4, {"beta": 1.})
    with pytest.raises(ValueError):
        kernel_weights("hyperbolic", 4, {"s": -1., "p": 1.})
