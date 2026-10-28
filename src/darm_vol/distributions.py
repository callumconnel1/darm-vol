"""Conditional Gaussian and unit-variance Student-t innovations."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def standardized_t_logpdf(z: np.ndarray, nu: float) -> np.ndarray:
    """Log density of t_nu scaled by sqrt((nu-2)/nu), nu > 2."""
    if not np.isfinite(nu) or nu <= 2:
        raise ValueError("Student-t degrees of freedom nu must be > 2")
    z = np.asarray(z, dtype=float)
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log((nu - 2) * np.pi)
            - 0.5 * (nu + 1) * np.log1p(z**2 / (nu - 2)))


def gaussian_logpdf(z: np.ndarray) -> np.ndarray:
    """Standard Gaussian log-density."""
    return -0.5 * (np.log(2 * np.pi) + np.asarray(z, dtype=float)**2)


def draw_shocks(rng: np.random.Generator, count: int, distribution: str,
                nu: float | None = None) -> np.ndarray:
    """Draw conditionally standardised, independent shocks."""
    if distribution == "normal":
        return rng.standard_normal(count)
    if distribution == "student-t":
        if nu is None or nu <= 2:
            raise ValueError("Student-t simulation requires nu > 2")
        return rng.standard_t(nu, size=count) * np.sqrt((nu - 2) / nu)
    raise ValueError(f"Unknown innovation distribution: {distribution}")
