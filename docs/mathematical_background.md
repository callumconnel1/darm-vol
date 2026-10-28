# Model mathematics

Reference: Connell, DARM Preprint v1.0 (19 October 2026), §§2, 4, 5;
Mathematical Correction Report, §1.

For `K` finite the model is the ARCH(K) restriction

```text
h_t = omega + rho Σ_{k=1}^K w_k(theta) u_(t-k)
```

where `u=epsilon^2` for symmetric DARM and
`u=epsilon^2 (1-kappa+2*kappa*I[epsilon<0])` otherwise.
Assume iid standardized, symmetric innovation `z_t`,
`epsilon_t = sqrt(h_t)z_t`, and `0<=rho<1`. Then
`E[h] = omega/(1-rho)` and future transformed shocks have
conditional expectation `E[u_t|F_(t-1)]=h_t`.

The kernels in this version have exactly normalized finite K weights.
The exponential weight is proportional to `beta^(k-1)`, the power weight to
`(k-1+delta)^(-alpha)`, and the hyperbolic weight to `(1+s*k)^(-p)`.
All parameters have strictly positive shape bounds except beta in (0,1).

The infinite geometric DARM representation corresponds to a restricted
GARCH(1,1), **not** its finite renormalized counterpart. Asymmetry with an
infinite geometric kernel corresponds to a restricted positive-news-impact
GJR-GARCH; this does not confer equality for finite K.

Multi-step recursion uses known `u` for past lags and its conditional
expectation `h` for not-yet-observed future lags. The expected cumulative
integrated variance is `Σ_{j=1}^H E_T[h_(T+j)]`. No forecast of an unobserved
future squared return is substituted for this expectation.

The model inherits finite-sample identification issues in power kernel
parameters. Do not interpret a high fitted shape or finite-lag response
half-life as proven structural long memory.
