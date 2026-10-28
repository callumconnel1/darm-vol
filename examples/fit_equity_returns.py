"""Optional yfinance demonstration; requires internet and pip install yfinance pandas."""
import numpy as np
import yfinance as yf

from darm_vol import DARM

prices = yf.download("AAPL", start="2015-01-01", end="2025-01-01",
                     auto_adjust=True)["Close"].squeeze()
returns = np.log(prices / prices.shift(1)).dropna()  # decimal log returns
result = DARM(kernel="power", distribution="student-t", lags=252).fit(returns)
print(result.summary())
print("5-day expected integrated variance:", result.forecast(5).integrated_variance)
