# darm-vol

**Decaying AutoRegressive Memory (DARM) — conditional-volatility models for Python**

**Author:** Callum Connell  
**Release:** 0.1.1

`darm-vol` implements parsimonious, normalized-kernel ARCH models for estimating, forecasting, and simulating **time-varying conditional variance**.

The implementation follows *Decaying Autoregressive Memory Models for Conditional Volatility: Mathematical Foundations, Finite-Sample Identification and Out-of-Sample Evidence* (Connell, preprint v1.0, 19 October 2026).

## Installation

Install directly from PyPI:

```bash
python -m pip install darm-vol
```

Import using:

```python
from darm_vol import DARM
```

## Features

- **Symmetric and asymmetric DARM** with finite-lag conditional-variance feedback.
- **Three normalized memory kernels:** exponential, power-law, and hyperbolic.
- **Gaussian and standardized Student-t** conditional innovation distributions.
- **Maximum-likelihood fitting** with bounded multi-start optimization and convergence diagnostics.
- **One-step and multi-step conditional-variance forecasts**, including expected integrated variance.
- **Simulation, residual diagnostics, and optional plotting** using Matplotlib.
- Optional Numba acceleration for parts of the numerical implementation.

## Mathematical Specification

For standardized independent innovations with zero mean and unit variance, DARM models return innovations as:

```text
εₜ = √hₜ · zₜ
```

### Symmetric DARM

The symmetric finite-memory conditional variance is:

```text
hₜ = ω + ρ Σₖ₌₁ᴷ wₖ(θ) ε²ₜ₋ₖ
```

Subject to:

- ω > 0
- 0 ≤ ρ < 1
- wₖ(θ) ≥ 0
- Σₖ₌₁ᴷ wₖ(θ) = 1

Here:

- **ω** is the variance intercept.
- **ρ** controls aggregate volatility feedback.
- **wₖ(θ)** determines the distribution of memory across historical observations.
- **K** is the finite memory truncation length.
- **εₜ** represents the return innovation.

### Asymmetric DARM

For asymmetric DARM, define the transformed squared innovation:

```text
uₜ = ε²ₜ [1 − κ + 2κ · 1{εₜ < 0}]
```

The corresponding conditional variance is:

```text
hₜ = ω + ρ Σₖ₌₁ᴷ wₖ(θ) uₜ₋ₖ
```

Subject to:

```text
0 ≤ κ < 1
```

A negative shock contributes a factor of `1 + κ`, compared with `1 − κ` for an equally large positive shock.

The transformed shock `uₜ` is distinct from the Student-t degrees-of-freedom parameter `ν`.

Both Gaussian and Student-t innovation laws are standardized to unit variance.

## Memory Kernels

DARM supports three normalized kernel families.

All weights are defined for lags `k = 1, …, K` and normalized to sum to one.

| Kernel | Unnormalized weights | Parameters |
|---|---|---|
| Exponential | β^(k−1) | 0 < β < 1 |
| Power law | (k−1+δ)^(−α) | α > 0, δ > 0 |
| Hyperbolic | (1+sk)^(−p) | s > 0, p > 0 |

Kernel normalization:

```text
wₖ(θ) = qₖ(θ) / Σⱼ₌₁ᴷ qⱼ(θ)
```

DARM is a restricted, low-dimensional ARCH(K) specification.

Finite-lag exponential DARM is not algebraically identical to GARCH(1,1). The exact correspondence applies to the appropriate infinite geometric kernel.

## Quick Start

Simulate a finite-memory DARM process, estimate its parameters, and forecast variance.

```python
from darm_vol import DARM

model = DARM(
    kernel="power",
    distribution="student-t",
    lags=30,
    asymmetric=True,
)

true_params = {
    "omega": 0.00002,
    "rho": 0.70,
    "alpha": 1.8,
    "delta": 5.0,
    "kappa": 0.4,
    "nu": 8.0,
}

# Simulate observations
sample = model.simulate(
    true_params,
    nobs=1200,
    burn=1000,
    random_state=42
)

# Estimate parameters
results = model.fit(
    sample.returns,
    n_starts=2,
    random_state=42
)

print(results.summary())

# Forecast future conditional variance
forecast = results.forecast(horizon=5)

print("Daily conditional variance:")
print(forecast.daily_variance)

print("Expected integrated variance:")
print(forecast.integrated_variance)

# Simulate from fitted parameters
simulated = results.simulate(
    nobs=1000,
    burn=1000,
    random_state=99
)

print(simulated.returns.shape)
```

For a real-data example, see:

[Fit Equity Returns](https://github.com/callumconnel1/darm-vol/blob/main/examples/fit_equity_returns.py)

The example downloads historical market prices using yfinance, calculates log returns, and fits a DARM volatility model.

**Input scale:** Supply returns in their original units. For example, `0.01` represents a 1% decimal return.

DARM does not silently rescale or demean observations.

## Maximum-Likelihood Estimation

Parameters are estimated using conditional maximum likelihood.

For memory length K, estimation treats the first K returns as presample observations.

The effective number of likelihood contributions is:

```text
n = T − K
```

The results object exposes:

```python
results.params
results.loglikelihood
results.aic
results.bic
results.conditional_variance
results.conditional_volatility
results.residuals
results.standardized_residuals
results.converged
results.summary()
```

The first K fitted variance entries are NaN under the conditioning convention.

## Volatility Forecasting

DARM supports one-step and multi-step conditional variance forecasting.

```python
forecast = results.forecast(horizon=10)
```

The forecast object exposes:

```python
forecast.daily_variance
forecast.daily_volatility
forecast.integrated_variance
```

The expected integrated variance over horizon H is:

```text
IVₜ,ₕ = Σⱼ₌₁ᴴ Eₜ[hₜ₊ⱼ]
```

The daily volatility forecast is the square root of expected conditional variance, rather than the expectation of the square root of variance.

Future unobserved shocks are integrated out using their conditional expectations.

The forecast does not access future realized returns.

## Simulation

Simulate observations using Gaussian or standardized Student-t innovations.

```python
simulation = results.simulate(
    nobs=5000,
    burn=1000,
    random_state=42
)
```

Simulated observations follow the specified DARM conditional-variance recursion.

## Optional Dependencies

Install plotting support:

```bash
python -m pip install 'darm-vol[plot]'
```

Install optional performance acceleration:

```bash
python -m pip install 'darm-vol[performance]'
```

## Development Installation

Clone the GitHub repository:

```bash
git clone https://github.com/callumconnel1/darm-vol.git

cd darm-vol

python -m pip install -e '.[dev]'
```

Run the automated tests:

```bash
python -m pytest -q
```

Check code quality:

```bash
python -m ruff check src tests examples
```

## Scientific Limitations

- Kernel-shape parameters can be weakly identified in finite samples even when variance forecasts are accurate.
- The package does not automatically claim valid asymptotic parameter standard errors.
- Estimation uses a conditional finite-lag likelihood.
- The stationary distribution of presample observations is not integrated out.
- Finite power-law weights do not automatically imply non-summable econometric long memory.
- For Student-t innovations with 2 < ν ≤ 4, fourth moments are infinite, affecting squared-return diagnostics.
- A point forecast of integrated conditional variance does not guarantee calibrated predictive intervals for latent integrated variance.
- The research paper's primary 10-day comparison with GJR-GARCH was statistically inconclusive (HAC p = 0.146291).
- No universal model superiority or trading profitability is claimed.
- Package-level automated tests do not constitute full end-to-end replication of the original empirical results.

## Documentation

- [GitHub Repository](https://github.com/callumconnel1/darm-vol)
- [API Reference](https://github.com/callumconnel1/darm-vol/blob/main/docs/api_reference.md)
- [Mathematical Background](https://github.com/callumconnel1/darm-vol/blob/main/docs/mathematical_background.md)
- [Installation Guide](https://github.com/callumconnel1/darm-vol/blob/main/docs/installation.md)
- [Research Papers](https://github.com/callumconnel1/darm-vol/tree/main/research)

## Citation

Connell, Callum (2026).

*Decaying Autoregressive Memory Models for Conditional Volatility: Mathematical Foundations, Finite-Sample Identification and Out-of-Sample Evidence.*

Preprint v1.0, 19 October 2026.

## Project Status

DARM is early-stage scientific research software.

The current release provides finite-memory conditional-volatility estimation, simulation, forecasting, and diagnostics.

The package is intended for quantitative researchers, financial econometricians, and developers working with financial time series.

For theoretical assumptions, empirical validation, and model limitations, consult the accompanying research papers.