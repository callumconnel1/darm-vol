"""Bounded conditional maximum likelihood with explicit convergence diagnostics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from .kernels import PARAMETERS
from .likelihood import log_likelihood


# Numerical boxes used for v0.1 fitting, not theoretical supports.
SHAPE_BOUNDS = {"beta": (1e-4, .9999), "alpha": (1e-4, 10.),
                "delta": (1e-4, 100.), "s": (1e-4, 100.),
                "p": (1e-4, 10.), "kappa": (0., .999), "nu": (2.05, 150.)}
SHAPE_DEFAULTS = {"beta": .90, "alpha": 1.8, "delta": 5.,
                  "s": .5, "p": 1.8, "kappa": .3, "nu": 8.}


@dataclass(frozen=True)
class OptimizerReport:
    """Detailed optimizer outcome for one starting point."""
    success: bool
    message: str
    loglikelihood: float
    iterations: int
    evaluations: int
    parameters: dict[str, float]


def maximum_likelihood(eps: np.ndarray, *, kernel: str, lags: int,
                       distribution: str, asymmetric: bool,
                       start_params: Mapping[str, float] | None = None,
                       n_starts: int = 2, random_state: int | None = 42,
                       maxiter: int = 400, tol: float = 1e-7,
                       disp: bool = False, allow_failed: bool = False,
                       use_numba: bool = False) -> tuple[dict[str, float], OptimizerReport,
                                                        list[OptimizerReport]]:
    """Find best *converged* L-BFGS-B fit across starts.

    omega is internally optimized as log(omega / sample_variance) to avoid
    terrible scaling on decimal returns. Returned parameters are unscaled.
    First K returns only initialize the observed ARCH state, not likelihood.
    """
    if not isinstance(n_starts, int) or isinstance(n_starts, bool) or n_starts < 1:
        raise ValueError("n_starts must be a positive integer")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be positive")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    scale = float(np.mean(eps[lags:]**2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sample variance proxy must be positive and finite")
    names = ["omega", "rho", *PARAMETERS[kernel]]
    if asymmetric:
        names.append("kappa")
    if distribution == "student-t":
        names.append("nu")
    bounds = [(-16., 5.), (1e-5, .999)] + [SHAPE_BOUNDS[name] for name in names[2:]]

    def encode(d: Mapping[str, float]) -> np.ndarray:
        return np.array([np.log(d["omega"] / scale), *[d[n] for n in names[1:]]], dtype=float)

    def decode(theta: Sequence[float]) -> dict[str, float]:
        return {"omega": float(scale * np.exp(theta[0])),
                **{n: float(theta[i]) for i, n in enumerate(names[1:], 1)}}

    base = {"omega": scale * .25, "rho": .75,
            **{n: SHAPE_DEFAULTS[n] for n in names[2:]}}
    if start_params is not None:
        extra = set(start_params) - set(names)
        if extra:
            raise ValueError(f"Unknown start_params: {sorted(extra)}")
        base.update(start_params)
    first = encode(base)
    if any(not np.isfinite(v) or v < lo or v > hi
           for v, (lo, hi) in zip(first, bounds)):
        raise ValueError("start_params lie outside the supported numerical bounds")
    rng = np.random.default_rng(random_state)
    starts = [first]
    for i in range(1, n_starts):
        theta = first.copy()
        theta[0] = np.clip(theta[0] + rng.normal(0, .75), *bounds[0])
        theta[1] = rng.uniform(.3, .94)
        for j, name in enumerate(names[2:], start=2):
            if name in {"alpha", "p"}:
                theta[j] = rng.uniform(.7, 3.5)
            elif name == "delta":
                theta[j] = rng.uniform(.3, 15.)
            elif name == "s":
                theta[j] = rng.uniform(.1, 2.)
            elif name == "beta":
                theta[j] = rng.uniform(.45, .98)
            elif name == "nu":
                theta[j] = rng.uniform(5., 16.)
            else:
                theta[j] = rng.uniform(.05, .8)
        starts.append(theta)

    def objective(theta: np.ndarray) -> float:
        try:
            ll = log_likelihood(eps, decode(theta), kernel, asymmetric, distribution,
                                lags, use_numba=use_numba)
            return -ll if np.isfinite(ll) else 1e100
        except (ValueError, FloatingPointError, OverflowError):
            return 1e100

    def callback(theta: np.ndarray) -> None:
        if disp:
            print(f"DARM iteration: conditional log likelihood {-objective(theta):.8f}")

    reports = []
    for x0 in starts:
        opt = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       callback=callback if disp else None,
                       options={"maxiter": maxiter, "ftol": tol})
        p = decode(opt.x)
        val = objective(opt.x)
        reports.append(OptimizerReport(bool(opt.success) and np.isfinite(val) and val < 1e99,
                         str(opt.message), -float(val), int(opt.nit), int(opt.nfev), p))
    converged = [r for r in reports if r.success]
    if not converged and not allow_failed:
        raise RuntimeError("DARM estimation did not converge: "
                           + "; ".join(r.message for r in reports))
    pool = converged if converged else reports
    best = max(pool, key=lambda r: r.loglikelihood)
    return best.parameters, best, reports
