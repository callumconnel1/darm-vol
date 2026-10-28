import numpy as np
from darm_vol.diagnostics import ljung_box, residual_diagnostics


def test_diagnostics_valid():
    z = np.random.default_rng(3).standard_normal(1000)
    d = residual_diagnostics(z, 10)
    assert d["nobs"] == 1000
    assert 0 <= d["ljung_box"]["pvalue"] <= 1
    assert ljung_box(z**2, 5)["statistic"] >= 0
