# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New `resp_metrics.effort` module with `effort_from_cycles()`, computing
  per-cycle indices of inspiratory effort from oesophageal, gastric and
  transdiaphragmatic pressures: `dPes`, `dPga`, `dPdi`, `WOB`, `PTPes`,
  `PTPga`, `PTPdi`, `PTPdi_PTPes` and `TTIdi`, plus `dPga_corr` and
  `PTPga_corr`, which reference the gastric swing to the Pga nadir rather than
  the end-expiratory baseline (see README "Limitations"). Definitions follow the ATS/ERS
  Statement on Respiratory Muscle Testing (2002) and the ERS statement on
  respiratory muscle testing at rest and during exercise (Laveneziana et al.,
  2019). `Pdi` is read from `pdi_col` when given, otherwise derived as
  `Pga - Pes`.
- `compute_from_labchart()` accepts `pga_col`, `pdi_col` and `pdi_max`.
  Providing `pes_col` merges the effort columns into the `ventilatory` table,
  adds an `effort` key to the returned dict, and writes an
  `<prefix>_effort_block<N>.csv` when `output_dir` is set.
- `VT_Ti` (mean inspiratory flow, L/s) and `Ti_Ttot` (inspiratory duty cycle)
  in both `ventilatory_from_cycles()` and `mechanical_from_cycles()`.
- `dPmo` (baseline-referenced inspiratory swing) and `Pmo_mean` (absolute mean
  inspiratory pressure) in `ventilatory_from_cycles()`, both from
  `pressure_col`.
- Expiratory effort in `effort_from_cycles()`: `dPes_exp`, `dPga_exp`,
  `PTPes_exp`, `PTPga_exp` and `TTIabd`, computed over
  `[t_expi, t_next_inspi]`. Both channels are referenced to the Pga nadir
  early in expiration, the relaxed abdominal level; neither boundary of the
  window is a resting instant. `TTIabd` needs the new `pga_max` argument,
  also exposed on `compute_from_labchart()`.
- `Pes_ee`, the end-expiratory oesophageal pressure, in `effort_from_cycles()`.
  Reported as an absolute value; an indirect marker of operating lung volume.

### Fixed

- Fixed cycle pairing logic: EXPI is now required to fall strictly between two
  consecutive INSPI markers; cycles with invalid timing (e.g. t_expi ≤ t_inspi)
  are skipped rather than producing wrong metrics.
- `nearest_idx` now raises a clear `ValueError` when passed an empty array
  instead of a cryptic NumPy error.
- Fixed fragile `n_cycle` row lookup in `ventilatory_from_cycles` to use the
  loop variable directly rather than re-indexing the DataFrame.

### Changed

- `WOB` moved from `ventilatory_from_cycles()` to `effort_from_cycles()`, and
  `ventilatory_from_cycles()` no longer takes `pes_col`. `WOB` still appears
  in the `ventilatory` table of `compute_from_labchart()` when `pes_col` is
  given, so the high-level API is unchanged.
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