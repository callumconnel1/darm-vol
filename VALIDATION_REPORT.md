# DARM v0.1.0 candidate — build & validation report

Date: 28 October 2026. Package has **not** been uploaded to PyPI or
published on GitHub. Software licence remains undecided and is a **release
blocker** requiring explicit author approval.

## Source inspection and mathematical design

- Supplied: preprint v1.0, technical supplement v1.0, mathematical
  correction report v1.0, package implementation requirements.
- Not supplied: original research Python sources, Monte Carlo notebook
  implementation archive, numerical model-fit regression fixtures,
  original equity market data, exact run environment.
- Implemented kernel equations 10–12 from preprint, finite K normalization.
- Implemented finite-K symmetric/asymmetric transformed-shock recursion
  and Gaussian / unit-variance Student-t conditional log-likelihood.
- Adopted the supplement's conditional likelihood convention: first K
  observed returns initialize the state; effective n=T-K.
- Multi-step forecasts use future conditional expected transformed shocks,
  not unavailable realized future returns, under symmetric innovation laws.
- Source code is newly implemented from papers; no claim that it recreates
  original research Python code line-for-line.

## Checks executed (current candidate)

| Check | Command | Result |
| --- | --- | --- |
| Unit and Monte Carlo tests | `PYTHONPATH=src python -m pytest -q` | 26 passed |
| Bytecode syntax check | `python -m compileall -q src tests examples` | Passed |
| Wheel build | `python -m pip wheel --no-build-isolation --no-deps . -w dist` | Passed |
| Source distribution | `python -c 'import setuptools; setuptools.setup()' sdist --dist-dir dist` | Passed |
| Built-wheel smoke test | Install wheel with `pip --no-index --no-deps` into new venv; inject existing environment's NumPy/SciPy paths; import + ML fit + 5-day forecast from `/tmp` | Passed (dependencies NOT isolated) |
| Time benchmark | `PYTHONPATH=src python benchmarks/benchmark_filter.py` | Success; approximately 0.014 seconds for T=10,000 K=252 (best of 3) |
| `python -m build` | Attempted tool availability | Not run: `build` not installed and network unavailable |
| `twine check dist/*` | Attempted tool availability | Not run: `twine` not installed and network unavailable |
| `ruff check` | Attempted tool availability | Not run: `ruff` not installed and network unavailable |
| Base `pip check` | `python -m pip check` | Failed due unrelated existing moviepy/pillow version mismatch |
| Research code numerical parity | Compare original likelihood/forecasts | Not possible: original implementation absent |
| Multi-Python matrix | GitHub Actions configured for Python 3.10–3.13 | Not executed locally; Python 3.13 tested |
| PyPI TestPyPI publishing | Manual upload | NOT attempted; approval and licence required |

### Representative filter benchmarks

Environment: current Python 3.13 session, NumPy/SciPy, single-threaded
Python loop around NumPy dot operations. The figures are local wall times;
not a comparison against original DARM research software or GARCH models.

| T | K | best of 3 |
| ---: | ---: | ---: |
| 1,000 | 30 | 0.0013 s |
| 5,000 | 30 | 0.0062 s |
| 10,000 | 30 | 0.0129 s |
| 1,000 | 252 | 0.0011 s |
| 5,000 | 252 | 0.0070 s |
| 10,000 | 252 | 0.0138 s |
| 5,000 | 1,000 | 0.0075 s |
| 10,000 | 1,000 | 0.0162 s |

## Release blockers

1. Ask Callum Connell to choose and approve a software licence.
2. Have the author review public project metadata / legal data rights.
3. Run Ruff, `python -m build`, `twine check`, clean dependency-isolated
   wheel install, and CI matrix in an online development environment.
4. Acquire old research implementation and numeric fixtures for
   independent regression comparisons and original research parity checks.
5. Register GitHub Trusted Publishers with matching protected environments.
6. Authorize a deliberate TestPyPI dispatch and subsequent GitHub Release.
7. Confirm at publication time that the `darm-vol` distribution is available.

See `docs/publishing.md` for step-by-step instructions.
