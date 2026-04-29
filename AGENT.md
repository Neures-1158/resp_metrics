# AGENT.md

Guidance for AI assistants working in this repository.

## Project

`resp_metrics` computes cycle-by-cycle respiratory metrics from LabChart text
exports parsed by `labchart_parser`. Small lab tool, `src/` layout, package
`resp_metrics`.

This is a research tool. Keep behavior explicit and auditable. Cycle detection
is comment-based only: users mark `INSPI` and `EXPI` in LabChart after visual
inspection. Do not add automatic breath detection unless explicitly requested.

## Commands

```bash
pip install -e ".[dev]"                          # dev install
pytest                                            # all tests
pytest tests/test_ventilatory.py::TestVentilatoryCycleValidation -v
ruff check src/ tests/                            # lint
black --target-version py310 src/ tests/          # format
isort --profile black src/ tests/                 # imports
python -m build                                   # package build
twine check dist/*                                # package metadata check
```

CI runs Linux/macOS/Windows x Python 3.10/3.11/3.12. Floor is **Python >=3.10**;
keep `pyproject.toml`, `README.md`, `CONTRIBUTING.md`, and the CI matrix in sync.

## Architecture

Public API from `resp_metrics`:

- `cycles_from_comments`
- `ventilatory_from_cycles`
- `mechanical_from_cycles`
- `compute_from_labchart`

Pipeline:

1. `labchart_parser.LabChartFile.from_file(path)` parses the LabChart export.
2. `cycles_from_comments()` builds complete cycles: `INSPI -> EXPI -> next INSPI`.
3. `ventilatory_from_cycles()` computes spontaneous-breathing metrics.
4. `mechanical_from_cycles()` computes mechanical ventilation metrics.
5. `compute_from_labchart()` wraps the full pipeline and optional CSV export.

## Data Conventions

- LabChart block time column: `time_block`.
- Comment columns: `block`, `time_block`, `Comment`.
- Cycle columns: `block_name`, `block`, `n_cycle`, `t_inspi`, `t_expi`,
  `t_next_inspi`.
- Flow units accepted by code: `L/min`, `lpm`, `L/s`, `l/sec`, `ls`.
- Internal flow unit: L/s.
- Volumes: L.
- Pressures: cmH2O.
- Keep `block_name` and `block` as leading columns when present.

## Scientific Invariants

- Spontaneous breathing: inspiratory flow is negative.
- Mechanical ventilation: inspiratory flow is positive.
- `VT` uses `volume_col` when available; the volume signal must increase during
  inspiration. Otherwise `VT` is integrated from flow.
- `WOB` requires esophageal pressure (`pes_col`). Do not substitute airway pressure.
- `PTP` is inspiratory pressure-time product relative to baseline pressure.
- `Pplat`, `Cstat`, and `R` stay `NaN` without a valid low-flow inspiratory plateau.
- If `Pplat` is missing, `dP = Ppeak - PEEP` is only a fallback and overestimates
  true driving pressure.
- Prefer `NaN` for physiologically unavailable metrics. Do not invent defaults.

## Parser Dependency

`resp_metrics` depends on `labchart_parser`. Relevant parser invariants:

- `LabChartFile.comments` includes `Time`, `time_block`, `time_abs`, `block`,
  and `Comment`.
- `LabChartFile.get_block_df(block)` returns one block without resetting the
  index and includes `Time`, `time_block`, `time_abs`, `Comment`, and channels.
- `LabChartFile.blocks` returns `list[int]`.

## Test Data

Unit tests use synthetic signals in `tests/conftest.py`. Example LabChart exports
live in `examples/data/` and should not be removed:

- `labchart_file.example.txt`
- `labchart_file_vs.example.txt`
- `labchart_file_negTime.txt`

For behavior changes, add small synthetic tests rather than relying on large
fixture files.

## Versioning

`__version__` reads from package metadata via
`importlib.metadata.version("resp_metrics")`. Bump the `version` field in
`pyproject.toml` only.
