"""User-facing DARM specification and estimation API."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .results import DARMResults

from .estimation import maximum_likelihood
from .kernels import canonical_kernel, kernel_weights
from .likelihood import conditional_variance, log_likelihood
from .simulation import Simulation, simulate_model
from .validation import check_variance_params, returns_array


class DARM:
    """Finite-memory Decaying AutoRegressive Memory ARCH(K) model.

    Parameters
    ----------
    kernel : {'exponential', 'power', 'hyperbolic'}
        Kernel parameterizations from Connell (2026), equations 10-12.
    distribution : {'normal', 'student-t'}
        Gaussian or *unit-variance* Student-t innovations.
    lags : int, default 252
        Finite, explicitly re-normalized memory length K.
    asymmetric : bool, default False
        Apply transformed-shock feedback with kappa in [0, 1).
    use_numba : bool, default False
        Enable optional Numba acceleration for variance filtering.

    Notes
    -----
    Supply RETURNS IN THE ORIGINAL UNITS: e.g. 0.01 for 1%, not 1.
    The package never rescales or demeans observations. It assumes zero
    conditional mean, so demean externally if your application requires it.
    Conditional MLE uses observations t=K,...,T-1; t<K are presample data.
    """

    def __init__(self, kernel: str = "power", distribution: str = "student-t",
                 lags: int = 252, asymmetric: bool = False, *,
                 use_numba: bool = False) -> None:
        self.kernel = canonical_kernel(kernel)
        if distribution not in {"normal", "student-t"}:
            raise ValueError("distribution must be 'normal' or 'student-t'")
        if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or lags < 1:
            raise ValueError("lags must be a positive integer")
        if not isinstance(asymmetric, bool):
            raise ValueError("asymmetric must be a bool")
        if not isinstance(use_numba, bool):
            raise ValueError("use_numba must be a bool")
        self.distribution = distribution
        self.lags = int(lags)
        self.asymmetric = asymmetric
        self.use_numba = use_numba

    def kernel_weights(self, params: Mapping[str, float]) -> np.ndarray:
        """Return exactly normalized finite-memory weights."""
        return kernel_weights(self.kernel, self.lags, params)

    def conditional_variance(self, returns: object,
                             params: Mapping[str, float]) -> np.ndarray:
        """Return conditional h_t (first K entries are NaN by design)."""
        data = returns_array(returns, minimum=self.lags + 1)
        return conditional_variance(data, params, self.kernel, self.asymmetric,
                                    self.distribution, self.lags, use_numba=self.use_numba)

    def loglikelihood(self, returns: object, params: Mapping[str, float]) -> float:
        """Evaluate finite-K conditional log-likelihood for specified parameters."""
        data = returns_array(returns, minimum=self.lags + 1)
        return log_likelihood(data, params, self.kernel, self.asymmetric,
                              self.distribution, self.lags, use_numba=self.use_numba)

    def fit(self, returns: object, *,
            start_params: Mapping[str, float] | None = None,
            n_starts: int = 2, random_state: int | None = 42,
            maxiter: int = 400, tol: float = 1e-7,
            disp: bool = False, allow_failed: bool = False) -> "DARMResults":
        """Fit parameters using bounded multi-start L-BFGS-B conditional MLE.

        Failed optimisation raises RuntimeError unless allow_failed=True.
        Returns a non-mutating results object with all optimizer reports.
        """
        from .results import DARMResults

        eps = returns_array(returns, minimum=self.lags + 1)
        params, opt, runs = maximum_likelihood(
            eps, kernel=self.kernel, distribution=self.distribution,
            lags=self.lags, asymmetric=self.asymmetric, start_params=start_params,
            n_starts=n_starts, random_state=random_state,
            maxiter=maxiter, tol=tol, disp=disp, allow_failed=allow_failed,
            use_numba=self.use_numba)
        return DARMResults(self, params, eps.copy(), opt, runs)

    def simulate(self, params: Mapping[str, float], nobs: int = 5000,
                 *, burn: int = 1000,
                 random_state: int | np.random.Generator | None = None) -> Simulation:
        """Simulate this explicitly parameterized finite-K DARM process."""
        check_variance_params(params, self.asymmetric, self.distribution)
        return simulate_model(params, self.kernel, self.distribution, self.lags,
                              self.asymmetric, nobs, burn, random_state)

    def __repr__(self) -> str:
        return (f"DARM(kernel={self.kernel!r}, distribution={self.distribution!r}, "
                f"lags={self.lags}, asymmetric={self.asymmetric})")
