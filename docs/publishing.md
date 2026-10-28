# PyPI release playbook (requires explicit author approval)

**STOP: Do not publish without choosing a software licence and obtaining
Callum Connell's explicit authorization.** A PyPI HTTP 404 for `darm-vol`
was observed while preparing this candidate (28 October 2026), which is
*not* a reservation or guarantee of future availability.

1. Decide on MIT, BSD-3-Clause, Apache-2.0 or another licence; install its
   exact text as `LICENSE` and remove `LICENSE_PENDING.md`. Check whether
   any code/images are licensed by third parties. Update metadata if needed.
2. Create a GitHub repository for the package and push the reviewed source.
   Configure its default branch protections. No URL is included in metadata
   until its real repository address is known.
3. Create accounts on [PyPI](https://pypi.org/) and
   [TestPyPI](https://test.pypi.org/) and turn on 2FA.
4. Run `python -m pip install -e '.[dev]'`, `pytest -q`,
   `ruff check src tests examples`, `python -m build`, and
   `python -m twine check dist/*` locally. Validate a clean wheel install.
5. Create GitHub Environments named `testpypi` and `pypi`, protected by
   approval rules if desired. Register **Trusted Publisher** separately for
   TestPyPI and PyPI, selecting GitHub owner/repo, workflow filename
   `publish.yml`, and corresponding GitHub environment `testpypi` or `pypi`.
6. With explicit approval, manually dispatch `publish.yml` selecting
   `testpypi`. Confirm install from the TestPyPI index with the dependencies
   resolved from regular PyPI, and check the public metadata.
7. Only after the license, tests, TestPyPI verification, and explicit
   authorization, create and publish a GitHub **Release** with tag `v0.1.0`.
   `publish.yml` deploys to real PyPI *only* for this release event.
8. Verify the resulting PyPI project and a fresh virtual environment:
   `python -m pip install darm-vol`, then `python -c 'from darm_vol import DARM; print(DARM())'`.
9. To release future versions, bump `__version__`, `pyproject.toml`, and
   `CITATION.cff`; update changelog and tests, then publish an authorized
   release with a matching tag.

If `darm-vol` becomes unavailable, select a different distribution name
*deliberately*, updating `pyproject.toml` and workflow/docs; the import
namespace may remain `darm_vol` if appropriate.
