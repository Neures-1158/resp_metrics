# CLAUDE.md — Guidance for Claude Assistants

This file provides comprehensive guidance for Claude to understand the codebase structure, development workflows, and conventions in the `resp_metrics` project.

## Project Overview

**resp_metrics** is a specialized research package for computing cycle-by-cycle respiratory/ventilatory metrics from LabChart text exports. It is a small, focused Python library (not a framework) built with a clear, auditable computation pipeline.

- **Type**: Science/research tool (physiology)
- **Size**: ~1,500 LOC across 6 core modules
- **Maturity**: Stable with active maintenance (recent commits through April 2026)
- **Language**: Python 3.10+
- **License**: MIT

### Key Principle

This is a **research tool where behavior must remain explicit and auditable**. Cycle detection is comment-based only (users manually mark `INSPI` and `EXPI` in LabChart after visual inspection). Do not add automatic breath detection or heuristics unless explicitly requested.

---

## Repository Structure

```
resp_metrics/
├── src/resp_metrics/              # Main package (src layout)
│   ├── __init__.py                # Public API exports
│   ├── api.py                     # High-level pipeline (compute_from_labchart)
│   ├── cycles.py                  # Cycle detection from comments
│   ├── ventilatory.py             # Spontaneous breathing metrics
│   ├── mechanical_vent.py         # Mechanical ventilation metrics (experimental)
│   └── utils.py                   # Shared utilities (unit conversion, filtering)
│
├── tests/                          # Unit and integration tests
│   ├── conftest.py                # Synthetic test fixtures and signals
│   ├── test_api.py                # API and pipeline tests
│   ├── test_cycles.py             # Cycle detection tests
│   ├── test_ventilatory.py        # Ventilatory metric tests
│   ├── test_mechanical_vent.py    # Mechanical ventilation tests
│   └── test_utils.py              # Utility function tests
│
├── examples/                       # Example usage
│   ├── example_usage.py           # Standalone Python example
│   ├── example_notebook.ipynb     # Jupyter notebook example
│   └── data/                      # Small LabChart export examples
│       ├── labchart_file.example.txt
│       ├── labchart_file_vs.example.txt
│       └── labchart_file_negTime.txt
│
├── .github/workflows/
│   └── ci.yml                      # CI matrix (Linux/macOS/Windows × 3.10/3.11/3.12)
│
├── .pre-commit-config.yaml         # Pre-commit hooks for formatting, linting
├── pyproject.toml                  # Package config, tool settings, dependencies
├── AGENT.md                        # Notes for AI agents (will deprecate in favor of this file)
├── CONTRIBUTING.md                # Contributor setup and workflow
├── README.md                       # User-facing documentation
├── CHANGELOG.md                    # Version history
└── LICENSE                         # MIT license
```

---

## Architecture & Module Relationships

### Data Flow

```
LabChart .txt file
        ↓
LabChartFile.from_file()  [labchart_parser]
        ↓
comments DataFrame + block DataFrames
        ↓
cycles_from_comments()  → Cycle DataFrame (t_inspi, t_expi, t_next_inspi)
        ↓
        ├─→ ventilatory_from_cycles()  → Standard metrics (BF, VT, VE, Ti, Te, ...)
        │
        └─→ mechanical_from_cycles()   → Mechanical metrics (PEEP, Ppeak, Pplat, dP, Cstat, R, MAP)
        ↓
[Optional: CSV export via _save_outputs()]
```

### Module Responsibilities

#### `api.py` — High-Level Pipeline

- **`compute_from_labchart(path, ...)`**: Single entry point for end users. Handles:
  - Loading LabChart files
  - Detecting cycles from comments
  - Computing ventilatory (and optionally mechanical) metrics
  - Optional CSV export
  - Multi-block processing

- **`_process_single_block()`**: Orchestrates cycle detection → metrics computation for one block

- **`_with_block_columns()`**: Ensures leading `block_name` and `block` columns (required for output format)

- **`_save_outputs()`**: Optional CSV export (never add without user request)

#### `cycles.py` — Cycle Detection

- **`cycles_from_comments(comments_df, block, insp_label, expi_label)`**: Builds cycles by pairing comment timestamps
  - Input: Comments from LabChart (time_block, block, Comment columns)
  - Output: DataFrame with `t_inspi`, `t_expi`, `t_next_inspi` for each complete cycle
  - **Invariant**: Only keeps cycles where EXPI is strictly between two consecutive INSPI markers
  - **No auto-detection**: Relies entirely on user-placed comments

#### `ventilatory.py` — Spontaneous Breathing Metrics

Computes per-cycle metrics for **spontaneous breathing** (inspiration = negative flow):
- Frequency: `BF` (breaths/min)
- Volume: `VT` (tidal volume, L)
- Ventilation: `VE` (minute ventilation, L/min)
- Timing: `Ti` (inspiration duration), `Te` (expiration), `Ttot` (total cycle), `IE` (ratio)
- Flow: `PIF` (peak inspiratory flow), `PEF` (peak expiratory flow)
- Pressure: `PTP` (pressure-time product, if pressure available)
- Work: `WOB` (work of breathing, requires esophageal pressure)

**Key convention**: Spontaneous = negative flow during inspiration.

#### `mechanical_vent.py` — Mechanical Ventilation Metrics (Experimental)

Extends `ventilatory_from_cycles()` with mechanical-specific metrics:
- Pressure: `PEEP`, `Ppeak`, `Pplat` (plateau), `dP` (driving pressure)
- Compliance: `Cstat` (static compliance)
- Resistance: `R` (airway resistance)
- Mean: `MAP` (mean airway pressure)

**Key convention**: Mechanical = positive flow during inspiration.

**Limitations**:
- `Pplat`, `Cstat`, `R` require a low-flow inspiratory plateau; absent → `NaN`
- If `Pplat` unavailable, `dP = Ppeak - PEEP` is a fallback (overestimates true driving pressure)
- Still experimental; use caution in publications

#### `utils.py` — Shared Utilities

- **`convert_flow_unit(flow, from_unit, to_unit)`**: Handles L/min ↔ L/s conversions
  - Accepted input units: `L/min`, `lpm`, `L/s`, `l/sec`, `ls` (case-insensitive)
  - Internal standard: **L/s**
  
- **`mean_with_nan()`**: Computes mean ignoring NaN values (for pressure/volume metrics that may be absent)

---

## Data Conventions

### Column Names & Units

**Mandatory columns (must be exact):**
- Cycle: `t_inspi`, `t_expi`, `t_next_inspi` (seconds, absolute time from block start)
- Output: `block_name`, `block`, `n_cycle` (always leading columns)

**Standard channel units:**
- Flow: **L/s** (internally; input units converted via `convert_flow_unit()`)
- Volume: **L**
- Pressure: **cmH2O**
- Time: **seconds** (from `time_block` in LabChart)

**LabChart convention (from labchart_parser dependency):**
- Time column: `time_block` (relative to block start)
- Comments: `time_block`, `block`, `Comment`
- Block data: includes `Time`, `time_block`, `Comment`, and channel columns

### DataFrame Key Conventions

Keep `block_name` and `block` as **leading columns** when present. This ensures user-facing output has semantic structure.

```python
# Correct output structure:
df = df[["block_name", "block"] + [c for c in df.columns if c not in ("block_name", "block")]]
```

---

## Scientific Invariants

These are non-negotiable rules reflecting respiratory physiology and are embedded in the code:

1. **Spontaneous breathing**: Inspiratory flow is **negative** (airflow in)
2. **Mechanical ventilation**: Inspiratory flow is **positive** (ventilator pushes air in)
3. **VT (tidal volume)**: Uses `volume_col` when available and valid; otherwise integrated from flow
4. **WOB (work of breathing)**: Requires esophageal pressure (`pes_col`). Do **not** substitute airway pressure
5. **PTP (pressure-time product)**: Computed relative to baseline (PEEP or start-of-inspiration pressure)
6. **Pplat, Cstat, R**: Require a valid low-flow inspiratory plateau; stay `NaN` if unavailable
7. **dP fallback**: When `Pplat` is missing, `dP = Ppeak - PEEP` overestimates true driving pressure
8. **Prefer NaN**: Missing or physiologically unavailable metrics should be `NaN`, never invented defaults

---

## Development Workflow

### Setup

```bash
git clone https://github.com/Neures-1158/resp_metrics.git
cd resp_metrics
python -m venv venv && source venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install -e ".[dev]"                          # Installs with test/lint dependencies
pre-commit install                               # Optional but recommended
```

### Key Commands

```bash
# Testing
pytest                                           # All tests
pytest tests/test_ventilatory.py -v             # Specific test file with verbose output
pytest tests/test_ventilatory.py::TestVentilatoryCycleValidation -v  # Specific class/function

# Formatting & Linting
black src/ tests/                               # Format with Black (line-length=88)
isort --profile black src/ tests/               # Sort imports
ruff check src/ tests/                          # Lint (E, F, I, N, W, UP, B rules; ignore E501)

# All at once (what CI does)
ruff check src/ tests/
black --target-version py310 src/ tests/
isort --profile black src/ tests/
pytest --cov=resp_metrics --cov-report=term-missing

# Package build
python -m build                                 # Create dist/ wheel and sdist
twine check dist/*                              # Validate package metadata
```

### Pre-commit Hooks

Pre-commit is configured in `.pre-commit-config.yaml` and runs:
- Basic checks: trailing whitespace, large files (>1MB), merge conflicts, private keys
- **Black** (24.3.0): Code formatting (line-length=88)
- **Ruff** (0.3.4): Linting with auto-fix (exits non-zero if fixes applied)
- **isort** (5.13.2): Import sorting with Black profile
- **nbstripout** (0.7.1): Strips cell outputs from `.ipynb` files (keeps repo clean)

If pre-commit fails on a commit, fix the issues (usually auto-fixable by the tools) and re-stage before committing.

### CI/CD

GitHub Actions runs on:
- **OS matrix**: Ubuntu Linux, macOS, Windows
- **Python versions**: 3.10, 3.11, 3.12
- **Checks**:
  1. Install dev dependencies
  2. Run tests with coverage (`pytest --cov`)
  3. Upload coverage to Codecov (only on `ubuntu-latest` + `3.11`)
  4. (CI also implicitly checks Black, isort, Ruff via successful import of formatted code)

**All tests must pass**; CI is blocking for PRs to `main`.

---

## Git Workflow

### Branching Strategy

- **Main branch**: `main` (protected, requires passing CI)
- **Feature branches**: Branch from `main`, use descriptive names (e.g., `fix-cycle-pairing-validation`)
- **Keep PRs small**: One logical change per PR

### Commit Messages

- Clear, descriptive first line (imperative mood)
- Reference related issues if applicable (e.g., `Fixes #123`)
- Example: `Fix cycle pairing and validate timings`

### Pull Requests

- Write a clear description of what changed and why
- Link to related issues/discussions
- Ensure all CI checks pass before merging
- Reviewer must approve (if needed per org settings)
- **Update [CHANGELOG.md](CHANGELOG.md)** under `## [Unreleased]` for user-visible changes

---

## Testing Philosophy & Patterns

### Test Organization

- **Unit tests**: Synthetic signals in `tests/conftest.py` (deterministic, fast)
- **Integration tests**: Small real LabChart exports in `examples/data/` (realistic but minimal)
- **Test coverage floor**: 90% (see `pyproject.toml`: `fail_under = 90`)

### Fixtures (conftest.py)

Synthetic signals used throughout tests:
- `synthetic_flow_spon()`: Spontaneous breathing (negative inspiration)
- `synthetic_flow_mech()`: Mechanical ventilation (positive inspiration)
- `synthetic_volume()`: Integrated volume (increases during inspiration)
- `synthetic_pressure()`: Airway pressure with plateau

These are small, deterministic, and repeatable. Use them for unit tests.

### Example LabChart Files

Do **not** remove:
- `labchart_file.example.txt`: Standard LabChart export
- `labchart_file_vs.example.txt`: With volume signal
- `labchart_file_negTime.txt`: With negative times (edge case for block start)

These are integration test fixtures and real-world examples.

### Testing Strategy for Changes

When making changes:

1. **Behavior changes** (bug fixes, algorithm tweaks): Add synthetic unit tests in the relevant `test_*.py` file
2. **Output format changes**: Update schema tests and verify example notebooks still run
3. **New metrics**: Implement with reference publication formulas; add formula-specific tests
4. **Parser dependency updates**: Check `labchart_parser` invariants; test with example files

Run `pytest --cov` locally; CI will catch issues on all OS/Python combinations.

---

## Common Tasks & How to Approach Them

### Adding a New Metric

1. **Identify the module**: Spontaneous → `ventilatory.py`, Mechanical → `mechanical_vent.py`
2. **Write formula tests first** (TDD). Use synthetic signals where the expected result is known.
3. **Implement in the appropriate function** (e.g., `ventilatory_from_cycles()`)
4. **Add column to output schema** and ensure `block_name`/`block` stay first
5. **Update CHANGELOG.md** under `## [Unreleased]`
6. **Update README.md metrics list** if user-visible
7. **Run full test suite**: `pytest --cov` must maintain >90% coverage

### Fixing a Bug in Cycle Detection

1. **Isolate with a synthetic test** in `test_cycles.py` that reproduces the issue
2. **Fix in `cycles.py`** (ensure the invariant "EXPI is strictly between two consecutive INSPI" holds)
3. **Verify with example files** if the bug manifests in real data
4. **Check all downstream metrics** aren't affected (run full suite)
5. **Document the fix** in the commit message (what went wrong and why it's fixed)

### Updating Dependencies

- **Never break Python >=3.10 floor**: Keep `pyproject.toml` `requires-python = ">=3.10"`
- **labchart_parser** is pinned to git; when updating, test against example files
- **numpy/pandas**: These are stable; version bumps are generally safe, but test with synthetic signals
- **Dev tools** (pytest, ruff, black, isort): Safe to bump; pre-commit ensures compatibility

### Modifying CSV Export

- **Why**: Output format is part of the public API; changes break user workflows
- **Approach**: Avoid unless there's a strong reason (bug fix, schema inconsistency)
- **If necessary**: Add tests for the new format, update example notebooks, document in CHANGELOG.md
- **Backward compat**: Generally not required for research tools, but think about user impact

---

## Code Style & Conventions

### Naming

- **Functions**: `snake_case` (e.g., `cycles_from_comments`, `ventilatory_from_cycles`)
- **Classes**: `PascalCase` (rarely used; mostly just utility functions)
- **Constants**: `UPPER_SNAKE_CASE` (unit conversion factors, magic thresholds)
- **Internal helpers**: Prefix with `_` (e.g., `_with_block_columns`, `_save_outputs`)

### Docstrings

- **Public functions**: NumPy-style docstrings (Parameters, Returns, Notes)
- **Internal functions**: Brief docstring or none if self-evident
- **Scientific assumptions**: Document explicitly (e.g., "Spontaneous: inspiration = negative flow")

### Imports

- Organized by `isort` (Black profile)
- Order: Standard library → Third-party (numpy, pandas, labchart_parser) → Local (from . import ...)
- Type hints: Use `from __future__ import annotations` for forward compatibility

### Formatting

- **Line length**: 88 (Black default)
- **Tool chain**: Black + isort + Ruff
- **Target Python**: 3.10+

---

## Important Constraints & Considerations

### No Automatic Breath Detection

This is a **research tool**, not a clinical device. Users mark cycles manually in LabChart after visual inspection. This is deliberate:
- Respects the scientific process (transparency)
- Avoids masking signal artifacts
- Allows non-standard breathing patterns

Unless explicitly asked to add heuristics, do not add automatic detection.

### Mechanical Ventilation is Experimental

`mechanical_vent.py` is still experimental. Use in publications with caution. Future changes are possible.

### Dependency on labchart_parser

`resp_metrics` is not self-contained; it depends on [labchart_txt_parser](https://github.com/Neures-1158/labchart_txt_parser). Key invariants from the parser:
- `LabChartFile.comments`: Always has `time_block`, `block`, `Comment` columns (sorted by time_abs)
- `LabChartFile.get_block_df(block)`: Returns one block's data with `time_block` and signal columns
- `LabChartFile.blocks`: Returns `list[int]` of available block numbers

If the parser changes, this package may break; tests will catch it.

### Versioning

- Version lives in `pyproject.toml` only (`[project] version = "X.Y.Z"`)
- Accessed at runtime via `importlib.metadata.version("resp_metrics")`
- Follows semantic versioning (major.minor.patch)

---

## Troubleshooting

### Tests Fail on Import

**Likely cause**: Missing dependency or broken `labchart_parser` git reference.

```bash
pip install -e ".[dev]"  # Reinstall with dev extras
python -c "from resp_metrics import *"  # Test import
```

### Pre-commit Fails with "ruff: exit non-zero on fix"

**This is expected**: Ruff auto-fixed something. Re-stage and commit:

```bash
git add <file>
git commit
```

### Coverage Drops Below 90%

Run `pytest --cov` locally to identify untested lines. Add tests before pushing.

### LabChart Example Files Missing

These are integration test fixtures. If accidentally deleted:
```bash
git checkout HEAD examples/data/
```

---

## Useful References

- **README.md**: User-facing quick start and metric definitions
- **CONTRIBUTING.md**: Setup and pull request guidelines
- **[AGENT.md](AGENT.md)**: Legacy AI agent guidance (being superseded by this file)
- **[labchart_txt_parser](https://github.com/Neures-1158/labchart_txt_parser)**: Dependency that handles LabChart parsing
- **Maintainer**: [Damien Bachasson](https://github.com/dambach) (NEURES lab)

---

## Summary for Claude Assistants

**When working in this repo:**

1. **Respect the research principle**: No automatic breath detection; cycles are comment-based by design.
2. **Keep scientific invariants intact**: Spontaneous = negative flow, Mechanical = positive flow, prefer NaN over invented values.
3. **Maintain >90% test coverage**: Add tests for behavior changes; use synthetic signals for isolation.
4. **Format and lint before committing**: Use Black, isort, Ruff; pre-commit is your friend.
5. **Update CHANGELOG.md** for user-visible changes.
6. **Update README.md** if the public API or metric definitions change.
7. **Keep PRs small and focused**: One logical change per PR.
8. **CI must pass**: Tests on Linux/macOS/Windows × 3.10/3.11/3.12.
9. **Avoid scope creep**: Don't add features or abstractions beyond what's requested.
10. **Document why, not what**: Comments for non-obvious constraints; code should be self-explanatory otherwise.

Good luck, and feel free to ask for clarification on scientific assumptions or architecture!
