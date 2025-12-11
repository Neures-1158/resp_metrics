# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
