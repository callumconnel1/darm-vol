"""Independent hand-computable deterministic reference cases (NOT archived research fits)."""
import numpy as np
import pytest
from darm_vol import DARM


def test_arch1_geometric_kernel_direct_formula():
    m = DARM(lags=1, kernel="exponential", distribution="normal")
    p = {"omega": .01, "rho": .4, "beta": .75}
    x = np.array([.2, -.5, .3])
    np.testing.assert_allclose(m.conditional_variance(x, p)[1:], [.026, .11])
    expected = sum(-.5 * (np.log(2*np.pi*h)+r*r/h)
                   for r, h in zip(x[1:], [.026, .11]))
    assert m.loglikelihood(x, p) == pytest.approx(expected)
