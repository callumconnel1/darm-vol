# Examples

Runnable, independent scripts in `examples/`:

- `basic_usage.py`: simulate, fit, forecast and simulate again.
- `fit_equity_returns.py`: optional yfinance AAPL acquisition; requires internet.
- `compare_kernels.py`: compare model AIC for all three kernels.
- `asymmetric_darm.py`: asymmetric innovations and long-run variance.
- `volatility_forecasting.py`: daily and integrated conditional variance.

To run against editable source install:

```bash
python -m pip install -e .
python examples/basic_usage.py
```

For valid financial applications, specify return units explicitly. The
zero-mean model takes returns as epsilons and does not estimate a mean model.
