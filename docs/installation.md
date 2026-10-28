# Installation

Requires Python >=3.10 and NumPy/SciPy. In the root of this repository:

```bash
python -m pip install .
python -m pip install -e '.[dev]'
python -m pytest -q
```

Optional plotting (`matplotlib`): `python -m pip install '.[plot]'`.
Optional Numba acceleration: `python -m pip install '.[performance]'`.
After the first authorized PyPI release, use `pip install darm-vol`.

No pandas, yfinance, Matplotlib or Numba dependency is required by the core.
