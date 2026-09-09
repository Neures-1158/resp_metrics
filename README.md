# Resp Metrics

[![CI](https://github.com/Neures-1158/resp_metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/Neures-1158/resp_metrics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Cycle-by-cycle respiratory metrics from ADInstruments LabChart text exports.
`resp_metrics` builds on
[labchart_parser](https://github.com/Neures-1158/labchart_txt_parser).

## Export from LabChart

<img src="img/lc_signal_export.png" width="300" alt="LabChart export dialog">

Set time display to **"Start from Block"** before exporting, and make sure **"Block header"** is ticked.

Respiratory cycles are read from user-placed `INSPI` and `EXPI` comments. No
automatic breath detection is performed; this is deliberate so signals are
visually inspected before analysis.

<img src="img/lc_inspi-expi_comments.png" width="500" alt="LabChart respiratory cycle comments">

## Install

```bash
pip install git+https://github.com/Neures-1158/resp_metrics.git
```

For development: `pip install -e ".[dev]"` (adds pytest, ruff, build, and twine).

## Quick start

```python
from resp_metrics import compute_from_labchart

result = compute_from_labchart(
    "examples/data/labchart_file_vs.example.txt",
    block=1,
    flow_col="Flow",
    flow_unit="L/s",
    volume_col=None,
    pressure_col="Pressure",
    mechanically_ventilated=False,
)

result["cycles"].head()
result["ventilatory"].head()
```

Lower-level functions are available for custom pipelines:

```python
from labchart_parser import LabChartFile
from resp_metrics import cycles_from_comments, ventilatory_from_cycles

lc = LabChartFile.from_file("data/recording.txt")
cycles = cycles_from_comments(lc.comments, block=1)
metrics = ventilatory_from_cycles(
    lc.get_block_df(1),
    cycles,
    flow_col="Flow",
    flow_unit="L/s",
)
```

See [examples/example_notebook.ipynb](examples/example_notebook.ipynb).

## Metrics

- Spontaneous breathing: inspiration is negative flow.
- Mechanical ventilation: inspiration is positive flow.
- Standard outputs: `BF`, `VT`, `VT_Ti`, `VE`, `Ti`, `Te`, `Ttot`, `Ti_Ttot`,
  `IE`, `PIF`, `PEF`, `PTP`, `dPmo` and `Pmo_mean`.
- Mechanical ventilation also returns `PEEP`, `Ppeak`, `Pplat`, `dP`, `Cstat`,
  `R`, and `MAP` when signals support them.

### Respiratory effort (Pes / Pga / Pdi)

Pass `pes_col` (and optionally `pga_col`, `pdi_col`, `pdi_max`) to add
per-cycle indices of inspiratory effort. They are merged into the
`ventilatory` table and also returned as a standalone `effort` table.

| Metric | Definition | Unit |
| --- | --- | --- |
| `VT_Ti` | `VT / Ti`, mean inspiratory flow | L·s⁻¹ |
| `Ti_Ttot` | `Ti / Ttot`, inspiratory duty cycle | — |
| `Pes_ee` | end-expiratory Pes, absolute (indirect marker of operating lung volume) | cmH2O |
| `dPes` | `Pes_baseline − min(Pes)` over inspiration | cmH2O |
| `dPga` | `max(Pga) − Pga_baseline` | cmH2O |
| `dPga_corr` | same, referenced to the Pga nadir | cmH2O |
| `dPdi` | `max(Pdi) − Pdi_baseline` | cmH2O |
| `WOB` | `∫ (Pes_baseline − Pes) × (−Flow) dt`, work of breathing | J |
| `PTPes` | `∫ (Pes_baseline − Pes) dt` | cmH2O·s·breath⁻¹ |
| `PTPga` | `∫ (Pga − Pga_baseline) dt` | cmH2O·s·breath⁻¹ |
| `PTPga_corr` | same, integrated from the Pga nadir | cmH2O·s·breath⁻¹ |
| `PTPdi` | `∫ (Pdi − Pdi_baseline) dt` | cmH2O·s·breath⁻¹ |
| `PTPdi_PTPes` | `PTPdi / PTPes`, diaphragmatic share of the effort | — |
| `TTIdi` | `(mean inspiratory Pdi / Pdi_max) × (Ti / Ttot)` | — |

The baseline of each channel is the median over the 0.2 s preceding
inspiration onset, i.e. the resting end-expiratory level. `Pdi` is read from
`pdi_col` when given, otherwise derived as `Pga − Pes`. Definitions follow the
[ATS/ERS Statement on Respiratory Muscle Testing](https://www.atsjournals.org/doi/10.1164/rccm.166.4.518)
(*Am J Respir Crit Care Med* 2002;165:518-624) and the
[ERS statement on respiratory muscle testing at rest and during exercise](https://publications.ersnet.org/content/erj/53/6/1801214)
(Laveneziana et al., *Eur Respir J* 2019;53:1801214).

```python
from resp_metrics import compute_from_labchart

res = compute_from_labchart(
    "examples/data/labchart_file_pressures.example.txt",
    block=1,
    flow_col="flow",
    flow_unit="L/s",
    volume_col=None,
    pressure_col="Pmo",
    pes_col="Pes",
    pga_col="Pga",
    pdi_col="Pdi",
    pdi_max=97.0,  # cmH2O, measured during a maximal manoeuvre
)
res["effort"].head()
```

## Limitations

- `Pplat`, `Cstat`, and `R` require a low-flow inspiratory plateau.
- If `Pplat` is unavailable, `dP = Ppeak - PEEP` is a fallback and
  overestimates true driving pressure.
- `dPmo` (swing) and `Pmo_mean` (mean) come from `pressure_col`. Under
  inspiratory threshold loading that channel carries the applied load, which is
  why they sit with the ventilatory pattern rather than the effort indices.
  `dPmo` is referenced to the pre-inspiratory baseline and is immune to a DC
  offset; `Pmo_mean` is absolute, matching the average inspiratory mouth
  pressure (`PM`) reported by Bird et al., so comparing it with `PTP`/`Ti`
  reveals an offset on the channel.
- `WOB` requires esophageal pressure (`pes_col`) and flow; airway pressure is
  not substituted, as it would not represent patient effort.
- `PTPes` is not corrected for chest wall elastic recoil, which would require
  chest wall elastance. It is a practical within-subject index of global
  inspiratory effort, not an absolute measure.
- Gastric metrics are reported twice. When expiratory abdominal muscles are
  recruited, Pga is still elevated at the INSPI marker and falls as they relax
  at the start of inspiration, so the end-expiratory baseline sits above the
  relaxed level and `dPga` / `PTPga` are underestimated — an artifact ATS/ERS
  2002 explicitly flags. `dPga_corr` and `PTPga_corr` reference the Pga nadir
  instead (search window `pga_nadir_frac`, default the first third of
  inspiration) and match the uncorrected values when no such recruitment is
  present. `Pes` and `Pdi` are left uncorrected: their swings are an order of
  magnitude larger so the same offset is negligible, and `Pdi` is derived from
  `Pga` and `Pes` and cannot take an independent reference.
- `Pes_ee` is reported as an absolute value, so unlike every other effort
  column it carries any DC offset of the channel. ERS 2019 uses end-expiratory
  Poes to reveal intrinsic PEEP when hyperinflation is suspected, but it is a
  surrogate only — the reference method for end-expiratory lung volume is the
  inspiratory capacity manoeuvre. Read changes across conditions rather than a
  single absolute level.
- `TTIdi` needs `Pdi_max` from a maximal manoeuvre; it cannot be derived from
  tidal breathing and stays `NaN` when not supplied. The diaphragm fatigue
  threshold is a `TTIdi` of 0.15–0.18 (ATS/ERS).
- The final cycle in each block is excluded because the next inspiration onset
  is unknown.

## Tests

```bash
pytest
```

## Maintainer

Maintained under [NEURES](https://github.com/Neures-1158). Lead: Damien
Bachasson, PhD ([GitHub](https://github.com/dambach) |
[ORCID](https://orcid.org/0000-0001-6335-9916) |
[Lab](https://sante.sorbonne-universite.fr/structures-de-recherche/neurophysiologie-respiratoire-experimentale-et-clinique)).
Issues and PRs welcome.

MIT licensed. See [LICENSE](LICENSE).
