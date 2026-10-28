# darm-vol

**Decaying AutoRegressive Memory (DARM) conditional-volatility models**  
Author: **Callum Connell** · Version: **0.1.0 pre-release candidate**

A Python library for the finite-lag, normalized-kernel ARCH specifications developed in
*Decaying Autoregressive Memory Models for Conditional Volatility: Mathematical Foundations,
Finite-Sample Identification and Out-of-Sample Evidence* (preprint v1.0, 19 October 2026).

> **Not published.** `pip install darm-vol` will only work after a PyPI release.
> Source installation works now. The package has no selected software licence;
> author approval is required **before** any public distribution.

## Model

For zero conditional-mean innovations `epsilon_t = sqrt(h_t) z_t`, where independent,
standardized, symmetric `z_t` have mean zero and variance one, symmetric DARM is

```text
h_t = omega + rho * sum_{k=1}^K w_k(theta) * epsilon_(t-k)^2
```

with `omega > 0`, `0 <= rho < 1`, `w_k >= 0`, `sum w_k = 1`.
For asymmetric DARM, replace each squared innovation with

```text
u_t = epsilon_t^2 * [1 - kappa + 2*kappa*1{epsilon_t<0}]
```

where `0 <= kappa < 1`. Here `u_t` denotes the transformed shock; **`nu` in
parameter dictionaries denotes Student-t degrees of freedom, not `u_t`**.

Finite-lag kernels use `k=1..K`:

| Kernel | Unnormalized weight | Parameters |
|---|---|---|
| Exponential | `beta^(k-1)` | `0 < beta < 1` |
| Power | `(k-1+delta)^(-alpha)` | `alpha, delta > 0` |
| Hyperbolic | `(1+s*k)^(-p)` | `s, p > 0` |

Every family is normalized over exactly **K** lags. Hyperbolic coincides with
a subset of shifted power weights (`alpha=p`, `delta=1+1/s`) rather than an
independent stochastic-process class. The infinite-lag counterpart is a
**different specification**; finite exponential DARM is not exactly GARCH(1,1).
The infinite power kernel has a finite normalizing constant only when `alpha>1`.

DARM is a low-dimensional parametrization of ARCH(K), not a new stochastic
process outside ARCH theory. A finite-K DARM can show persistent squared returns
but is **not** strictly long-memory in the covariance non-summability sense under
the manuscript's finite-moment conditions.

## Installation

In a checked-out source directory:

```bash
python -m pip install .
python -m pip install -e '.[dev]'
```

Python >=3.10; runtime: NumPy and SciPy. pandas Series are accepted as inputs
without requiring pandas at runtime. Plotting and optional acceleration:

```bash
python -m pip install '.[plot]'
python -m pip install '.[performance]'
```

## Quick start — fit, forecast, simulate

```python
import numpy as np
from darm_vol import DARM

model = DARM(kernel="power", distribution="student-t", lags=30,
             asymmetric=True)
true = {"omega": 0.00002, "rho": 0.70, "alpha": 1.8,
        "delta": 5.0, "kappa": 0.4, "nu": 8.0}

sample = model.simulate(true, nobs=1200, burn=1000, random_state=42)
fit = model.fit(sample.returns, n_starts=2, random_state=42)
print(fit.summary())
print(fit.params, fit.loglikelihood, fit.aic, fit.bic)

fc = fit.forecast(horizon=5)
print(fc.daily_variance, fc.daily_volatility)
print(fc.integrated_variance)

new_data = fit.simulate(nobs=1000, burn=1000, random_state=99)
print(new_data.returns.shape, new_data.conditional_volatility.shape)

# Optional: pip install '.[plot]'
# fig, ax = fit.plot_volatility()
```

For live equity data see [`examples/fit_equity_returns.py`](examples/fit_equity_returns.py).
**Inputs are always in their original units**: e.g., a 1% decimal log return
is `0.01`, not `1.0`. No automatic scaling or demeaning occurs.

The **first K observations initialize the observed lag state** and are excluded
from the fitted conditional likelihood. Conditional variance arrays have `NaN`
in the first K positions. Therefore AIC/BIC use `T-K` observations and the
estimated number of free parameters. `omega` is internally reparameterized
by a sample-variance scale during optimization, then returned in the original
input units.

## Forecasting

For future times, unknown transformed shocks are replaced by their conditional
expectations. Both Gaussian and standardized Student-t innovations are
symmetric, giving `E[A_t]=1` with the stated asymmetric transformation.
`forecast(horizon=H).daily_variance` computes `E_T[h_(T+j)]` for `j=1,...,H`.
`integrated_variance` sums them. `daily_volatility` is `sqrt(E_T[h])`, **not**
`E_T[sqrt(h)]`.

With zero-mean martingale-difference returns, the sum of future conditional
variances also equals the conditional variance of their cumulative sum;
that equivalence should not be applied if separate serially dependent mean
dynamics are introduced. No interval coverage claim is implied.

## Simulation and diagnostics

Simulation uses independent standardized Gaussian or Student-t shocks and
exact finite-K transformed-shock recursion with K presample transformed shocks
fixed at `omega/(1-rho)` (valid due to the symmetric innovation laws). A burn-in
period reduces the effect of initialization. For infinite fourth-moment
Student-t laws with `2<nu<=4`, squared-return variance diagnostics do not
have the usual finite-moment theoretical justification.

`fit.diagnostics(lags=10)` returns residual mean, variance, excess kurtosis,
and Ljung-Box tests on residuals and residual squares (raw asymptotic p-values;
no automatic degrees-of-freedom correction for fitted parameters).
`fit.plot_volatility()`, `fit.plot_residuals()`, `fit.plot_kernel()`,
`fit.plot_forecast(10)` return Matplotlib `(Figure, Axes)` objects.

## Scientific limitations

- **Identification:** high kernel-shape uncertainty, flat likelihood ridges,
  and boundary estimates can arise even when point forecasts are accurate.
  No default Wald standard errors are provided; use careful profile inference.
- **Conditional MLE:** the first K returns are treated as conditioning
  observations, rather than integrating out a stationary presample.
- **Finite K:** neither finite geometric kernel GARCH equivalence nor genuine
  covariance non-summable long memory is claimed.
- **Forecast calibration:** prediction intervals for future squared returns
  are not interchangeable with intervals for latent integrated variance.
- **Performance:** source supplied for this build includes the papers, but
  **not original research implementation or raw experiment inputs**. Our
  independent numerical unit tests do *not* establish full paper replication.
- **Inference:** fitted Student-t nu can be weakly identified; numerical boxes
  are implementation limits, not the unrestricted theoretical parameter domain.
- **Empirical superiority:** the paper's primary 10-day comparison against
  GJR-GARCH is inconclusive (HAC p=0.146291); no universal model superiority
  or economic trading benefit is claimed.

## Research, attribution and citation

Callum Connell (2026), *Decaying Autoregressive Memory Models for Conditional
Volatility: Mathematical Foundations, Finite-Sample Identification and
Out-of-Sample Evidence*, preprint v1.0, 19 October 2026. See
[`research/`](research/) for the documents used to implement this package,
including the technical supplement and mathematical correction report.
See [`CITATION.cff`](CITATION.cff) for software citation metadata.
No DOI or public external paper URL has been supplied, so none is fabricated.

## Development status

This repository is a **0.1.0 pre-release source candidate**, not a published
PyPI release or independent replication of Experiments 13–18. See
[`docs/limitations.md`](docs/limitations.md) and
[`docs/publishing.md`](docs/publishing.md) before releasing.

Contributions: see [`CONTRIBUTING.md`](CONTRIBUTING.md).
