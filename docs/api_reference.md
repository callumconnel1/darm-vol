# Public API

## `DARM(kernel='power', distribution='student-t', lags=252, asymmetric=False, *, use_numba=False)`

- `kernel`: `'exponential'` (`'exp'`), `'power'` (`'power-law'`), `'hyperbolic'`.
- `distribution`: `'normal'` or `'student-t'` (unit-variance innovations).
- `lags`: positive finite integer K.
- `asymmetric`: enable nonnegative impact asymmetry parameter `kappa`.
- `use_numba`: optionally accelerate filtering; not needed to import or fit core.

## `DARM.fit(returns, *, start_params=None, n_starts=2, random_state=42, maxiter=400, tol=1e-7, disp=False, allow_failed=False)`

Uses SciPy L-BFGS-B and returns `DARMResults`. `returns` is a numeric 1D
NumPy array or pandas Series; no automatic rescaling, demeaning, gap filling,
coercing invalid data, or inference for missing values. Failed convergence
raises unless explicitly opted out. Each numeric optimization bound is
listed in `src/darm_vol/estimation.py`; shape parameters, kappa, nu use
computational boxes, not general parameter support statements.

## `DARM.kernel_weights(params)`

Return array of K nonnegative weights summing to one.

## `DARM.conditional_variance(returns, params)`

Return length-T array, with first K elements NaN because no observed full
lag state exists there; remainder is the exact finite-K formula.

## `DARM.loglikelihood(returns, params)`

Scalar sum of conditional Gaussian or Student-t log densities, t=K..T-1.

## `DARM.simulate(params, nobs=5000, *, burn=1000, random_state=None)`

Return `Simulation(innovations, conditional_variance, standardized_shocks)`;
properties: `returns`, `conditional_volatility`.

## `DARMResults`

Properties: `params` (dict); `loglikelihood`, `aic`, `bic`, `nobs`,
`nparams`; `conditional_variance`, `conditional_volatility` (arrays,
first K NaN); `residuals`, `standardized_residuals` (first K NaN);
`converged`, `optimization`, `optimizer_runs`.

Methods: `summary()`, `forecast(horizon=1)`, `simulate(...)`,
`diagnostics(lags=10)`, `plot_volatility()`, `plot_residuals()`,
`plot_kernel()`, `plot_forecast(horizon=10)`.

`Forecast`: `daily_variance`, `daily_volatility`, `integrated_variance`.

`Simulation`: `innovations`, `returns`, `conditional_variance`,
`conditional_volatility`, `standardized_shocks`.

Parameter dictionaries by kernel:

- Base: `omega`, `rho`.
- Exponential: `beta`; power: `alpha`, `delta`; hyperbolic: `s`, `p`.
- Asymmetric: `kappa` (if enabled).
- Student-t: `nu` (if enabled).

## Optimizer diagnostics

`OptimizerReport` fields: `success`, `message`, `loglikelihood`,
`iterations`, `evaluations`, `parameters`. `optimizer_runs` preserves the
outcome for *every* starting point. `summary()` does **not** include
asymptotic standard errors.
