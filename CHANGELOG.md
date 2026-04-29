# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fixed cycle pairing logic: EXPI is now required to fall strictly between two
  consecutive INSPI markers; cycles with invalid timing (e.g. t_expi ≤ t_inspi)
  are skipped rather than producing wrong metrics.
- `nearest_idx` now raises a clear `ValueError` when passed an empty array
  instead of a cryptic NumPy error.
- Fixed fragile `n_cycle` row lookup in `ventilatory_from_cycles` to use the
  loop variable directly rather than re-indexing the DataFrame.

### Changed

- Aligned project tooling and documentation with `labchart_txt_parser`.
- Raised Python support floor to 3.10.
- Added blocking Black, isort, Ruff, and package build checks to CI.
- Switched package `__version__` to `importlib.metadata`.
- `compute_from_labchart` now accepts `os.PathLike[str]` for the `path`
  argument (in addition to `str`), matching `LabChartFile.from_file`.
- Trapezoidal integration uses `numpy.trapezoid` (NumPy ≥ 2.0) with automatic
  fallback to `numpy.trapz` for older NumPy versions.
- Added module-level `__all__` to each source module for consistent public API
  declarations.
- Added `Raises` sections to `ventilatory_from_cycles` and
  `mechanical_from_cycles` docstrings.

### Added

- Added `CLAUDE.md` with comprehensive AI assistant guidance covering architecture,
  data conventions, scientific invariants, development workflow, and key constraints.
- Added pre-commit configuration.
- Added Contributor Covenant Code of Conduct.

## [0.1.0] - 2025-01-XX

### Added

- Initial release
- Cycle detection from INSPI/EXPI comments via `cycles_from_comments()`
- Ventilatory metrics for spontaneous breathing via `ventilatory_from_cycles()`:
  - Timing: Ti, Te, Ttot, BF, I:E ratio
  - Volumes: VT, VE, PIF, PEF
  - Work: WOB (requires esophageal pressure), PTP
- Mechanical ventilation metrics via `mechanical_from_cycles()`:
  - Pressures: PEEP, Ppeak, Pplat, dP, MAP
  - Mechanics: Cstat, R (when plateau detected)
- High-level API `compute_from_labchart()` with multi-block support
- Example scripts and Jupyter notebook
- 92% test coverage