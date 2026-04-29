# Contributing

Issues and PRs welcome.

## Setup

```bash
git clone https://github.com/Neures-1158/resp_metrics.git
cd resp_metrics
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # optional but recommended
```

Requires Python >=3.10.

## Workflow

```bash
pytest
pytest tests/test_ventilatory.py::TestVentilatoryCycleValidation -v
ruff check src/ tests/
black src/ tests/
isort --profile black src/ tests/
```

CI runs tests across Linux/macOS/Windows x Python 3.10-3.12, plus blocking
Black, isort, Ruff, and package build checks.

## Pull requests

- Branch from `main`, keep PRs small.
- Add tests for behavior changes in cycle detection, unit conversion, metric
  formulas, output schemas, or CSV export naming.
- Update [CHANGELOG.md](CHANGELOG.md) under `## [Unreleased]` for
  user-visible changes.
- Keep scientific assumptions explicit. Do not add automatic breath detection
  unless the project direction changes.
- Pre-commit should pass; CI must be green.

By contributing you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
