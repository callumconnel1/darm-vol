# darm-vol

**Decaying AutoRegressive Memory (DARM) conditional-volatility models**  
Author: **Callum Connell** · Version: **0.1.0**

`darm-vol` is a Python library for fitting, forecasting and simulating finite-lag, normalized-kernel ARCH volatility models, as developed in *Decaying Autoregressive Memory Models for Conditional Volatility: Mathematical Foundations, Finite-Sample Identification and Out-of-Sample Evidence* (preprint v1.0, 19 October 2026).

**Install from [PyPI](https://pypi.org/project/darm-vol/):**

```bash
python -m pip install darm-vol
```

Import the package with `from darm_vol import DARM`.

## Model

Let $\varepsilon_t=\sqrt{h_t}\,z_t$, where the innovations $z_t$ are independent, symmetric, standardized shocks with mean zero and variance one. **Symmetric DARM** has conditional variance

$$h_t=\omega+\rho\sum_{k=1}^{K}w_k(\theta)\,\varepsilon_{t-k}^{2}$$

with $\omega>0$, $0\leq\rho<1$, $w_k(\theta)\geq0$, and $\sum_{k=1}^{K}w_k(\theta)=1$.

For **asymmetric DARM**, define the transformed squared shock

$$u_t=\varepsilon_t^2\left(1-\kappa+2\kappa\,\mathbf{1}_{\{\varepsilon_t<0\}}\right)$$

and use the conditional-variance recursion

$$h_t=\omega+\rho\sum_{k=1}^{K}w_k(\theta)\,u_{t-k},\qquad 0\leq\kappa<1$$

A negative shock contributes a factor of $1+\kappa$, versus $1-\kappa$ for a positive shock of the same magnitude. The asymmetry parameter is $\kappa$; the separate parameter $\nu$ denotes the Student-t degrees of freedom.

### Memory kernels

Each kernel is normalized over exactly $K$ lags:

| Kernel | Unnormalized weight at lag $k$ | Shape parameters |
|---|---|---|
| Exponential | $\beta^{k-1}$ | $0<\beta<1$ |
| Power law | $(k-1+\delta)^{-\alpha}$ | $\alpha>0,\ \delta>0$ |
| Hyperbolic | $(1+sk)^{-p}$ | $s>0,\ p>0$ |

For an unnormalized kernel $q_k(\theta)$, the implemented finite-lag weights are

$$w_k(\theta)=\frac{q_k(\theta)}{\sum_{j=1}^{K}q_j(\theta)},\qquad k=1,\ldots,K$$

The hyperbolic family coincides with a subset of shifted power weights ($\alpha=p$ and $\delta=1+1/s$), rather than defining an independent stochastic-process class. The infinite-lag counterpart is a **different specification**: finite exponential DARM is not exactly GARCH(1,1), and an infinite power-law kernel is normalizable only when $\alpha>1$.

DARM is a parsimoniously parameterized subclass of ARCH($K$), not a stochastic-process class outside established ARCH theory. Under the finite-moment conditions in the paper, persistent squared returns do not imply non-summable, econometric long memory.

## Installation

Install the published package:

```bash
python -m pip install darm-vol
```

Requires Python 3.10 or newer, NumPy and SciPy. NumPy arrays and pandas Series can be supplied as data; pandas is not a mandatory runtime dependency.

Optional plotting and acceleration dependencies are available as extras:

```bash
python -m pip install 'darm-vol[plot]'
python -m pip install 'darm-vol[performance]'
```

For local source development, clone the repository and install in editable mode:

```bash
git clone https://github.com/callumconnel1/darm-vol.git
cd darm-vol
python -m pip install -e '.[dev]'
```

## Quick start — fit, forecast, simulate

```python
import numpy as np
from darm_vol import DARM

model = DARM(
    kernel="power",
    distribution="student-t",
    lags=30,
    asymmetric=True,
)

true = {
    "omega": 0.00002,
    "rho": 0.70,
    "alpha": 1.8,
    "delta": 5.0,
    "kappa": 0.4,
    "nu": 8.0,
}

sample = model.simulate(true, nobs=1200, burn=1000, random_state=42)
fit = model.fit(sample.returns, n_starts=2, random_state=42)

print(fit.summary())
print(fit.params, fit.loglikelihood, fit.aic, fit.bic)

forecast = fit.forecast(horizon=5)
print(forecast.daily_variance, forecast.daily_volatility)
print(forecast.integrated_variance)

simulated = fit.simulate(nobs=1000, burn=1000, random_state=99)
print(simulated.returns.shape, simulated.conditional_volatility.shape)

# Requires: python -m pip install 'darm-vol[plot]'
# fig, ax = fit.plot_volatility()
```

For a real-equity demonstration using `yfinance`, see [`examples/fit_equity_returns.py`](examples/fit_equity_returns.py). `yfinance` is an optional example dependency, not a requirement for DARM.

**Return scaling:** input returns remain in their original units. For example, a 1% decimal log return is `0.01`, not `1.0`. The library does not automatically rescale or demean observations.

**Conditional fitting:** the first $K$ observations initialize the observed lag state and are omitted from the conditional log-likelihood. Fitted conditional-variance arrays contain `NaN` for these first $K$ positions; AIC and BIC use $T-K$ likelihood observations and the number of fitted parameters. The optimizer scales $\omega$ internally using the sample variance, then reports it in the units of the original data.

## Forecasting

Unknown future transformed shocks are replaced by their conditional expectations. For the supported symmetric, standardized Gaussian and Student-t innovation laws, $\mathbb{E}[A_t]=1$ even when the asymmetric shock transformation is enabled.

For forecast horizon $H$, `forecast(horizon=H).daily_variance` returns $\mathbb{E}_T[h_{T+j}]$ for $j=1,\ldots,H$. The expected integrated variance is

$$\operatorname{IV}_{T,H}=\sum_{j=1}^{H}\mathbb{E}_T[h_{T+j}]$$

The `daily_volatility` output is $\sqrt{\mathbb{E}_T[h_{T+j}]}$, **not** $\mathbb{E}_T[\sqrt{h_{T+j}}]$.

With zero-mean martingale-difference returns, this expected integrated variance also equals the conditional variance of the sum of the next $H$ returns. Do not assume the identity remains valid after adding serially dependent mean dynamics. A variance point forecast does not itself establish predictive-interval coverage.

## Simulation and diagnostics

Simulation uses independent standardized Gaussian or Student-t shocks and the finite-$K$ transformed-shock recursion. The $K$ presample transformed shocks are initialized at $\omega/(1-\rho)$, and a configurable burn-in reduces sensitivity to this initialization.

For Student-t innovations with $2<\nu\leq4$, fourth moments are infinite; ordinary squared-return variance diagnostics then lack their usual finite-moment justification.

`fit.diagnostics(lags=10)` provides standardized-residual moments and Ljung–Box tests on residuals and residual squares. Reported asymptotic p-values do not automatically adjust degrees of freedom for estimated parameters. The plotting methods `fit.plot_volatility()`, `fit.plot_residuals()`, `fit.plot_kernel()` and `fit.plot_forecast(10)` return Matplotlib `(Figure, Axes)` objects.

## Scientific limitations

- **Identification:** kernel-shape parameters may exhibit flat likelihood ridges, weak finite-sample identification and numerical boundary estimates even when forecasts are accurate. The package does not report default Wald standard errors.
- **Conditional MLE:** the first $K$ observations condition the likelihood; a stationary presample distribution is not integrated out.
- **Finite memory:** finite exponential DARM is not algebraically identical to GARCH(1,1), and finite-lag power weights do not imply genuine non-summable long memory.
- **Forecast calibration:** predictive intervals for future squared returns are not interchangeable with intervals for latent integrated conditional variance.
- **Reproducibility:** package unit tests validate implemented equations and behaviour; they are **not** an end-to-end replication of the paper's original empirical experiments. The archived source build does not contain the full original market panel or simulation paths.
- **Inference:** the Student-t degrees-of-freedom parameter $\nu$ can be weakly identified. Implementation bounds are numerical restrictions, not unrestricted theoretical parameter domains.
- **Empirical comparisons:** the paper's primary 10-day comparison against GJR-GARCH was statistically inconclusive (HAC $p=0.146291$). No universal superiority or demonstrated trading profitability is claimed.

## Research and citation

**Connell, Callum (2026).** *Decaying Autoregressive Memory Models for Conditional Volatility: Mathematical Foundations, Finite-Sample Identification and Out-of-Sample Evidence*. Preprint v1.0, 19 October 2026.

The [`research/`](research/) directory contains the preprint, technical supplement and mathematical correction report used as references for this implementation. See [`CITATION.cff`](CITATION.cff) for software citation metadata. No DOI is claimed.

## Development and contributing

**Release:** `0.1.0` — initial public package version. The API, mathematical assumptions and scientific limitations are documented in [`docs/`](docs/). The library remains early-stage research software; a public package release does not certify independent experimental replication.

Development instructions: [`CONTRIBUTING.md`](CONTRIBUTING.md). Release procedure: [`docs/publishing.md`](docs/publishing.md).
