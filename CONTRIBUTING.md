# Contributing

Please open a GitHub issue to discuss substantial numerical or public API changes.
Submit tests verifying mathematical identities independently from the implementation.
Do not alter kernel definitions or sample conditioning silently. Document version,
seed, data construction and optimization tolerances of all numeric comparisons.

Locally:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests examples
python -m build
python -m twine check dist/*
```

**Licence selection is pending.** Obtain author approval before accepting
third-party contributions or publicly redistributing the source.
