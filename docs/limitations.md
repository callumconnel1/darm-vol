# Limitations and scientific provenance

**Uploaded source base (28 October 2026):** DARM Preprint v1.0,
Technical Supplement v1.0, and Mathematical Correction Report v1.0.
The original executable experiment notebooks, source implementations,
optimized parameter checkpoints and original market data were not supplied.

This implementation reproduces kernel definitions and model equations from
the papers; the tests use independent analytic/deterministic references and
synthetic tests. **It has not been regression-tested against the original
2010–2025 model-fitting code or archived Experiments 13–18 outputs.**

- Finite K: renormalization and conditioning on the first K returns are part
  of the estimand. This is not an exact infinite-memory likelihood.
- Conditional likelihood assumes returns have zero mean and independent,
  symmetric, standardized innovations. It does not estimate a mean process.
- Shape identification can be weak. Inference needs profile or bootstrap
  methods with appropriate identifiability/boundary qualifications.
- Student-t innovations with `2<nu<=4` have infinite fourth moment.
- Forecasts report expected integrated variance, not latent variance
  prediction intervals. Squared-return predictive intervals require separate
  predictive distributions and may miscover latent IV.
- Hyperbolic kernels are a subset of shifted power weights.
- Historical paper results do not establish universal outperformance: the
  10-day primary GJR comparison is inconclusive (HAC p=0.146291).
- Parameter estimation is bounded numerical optimization and may get stuck in
  flat/ridged regions. Save and inspect the optimizer reports.
- Plotting is optional; performance without Numba is correct but can be slow
  for very long series and large K.

**Remaining verification**: acquire the original fitting and forecasting
scripts and numerical fixtures; run cross-implementation parity tests and
large-K/horizon timing benchmarks; validate supported Python versions in CI.
