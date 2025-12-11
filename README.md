# Resp_metrics

[![CI](https://github.com/Neures-1158/resp_metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/Neures-1158/resp_metrics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](https://github.com/Neures-1158/resp_metrics)

Cycle-by-cycle ventilatory metrics (BF, VT, VE, Ti, Te, Ttot, I:E, ...) computed from respiratory signals exported from ADInstruments LabChart.

## Features

- Parse LabChart `.txt` exports using [labchart_parser](https://github.com/Neures-1158/labchart_txt_parser).  Export from labchart as follows:
- 
  <img src="img/lc_signal_export.png" width="300" alt="LabChart screenshot showing signal export dialog">

    - Before exporting in LabChart, make sure time is displayed as "Start from Block"

- Extract respiratory cycles from INSPI/EXPI comments obtained using a macro in labchart. Absence of automatic detection is deliberate as it forces thorough inspection of signals.

  <img src="img/lc_inspi-expi_comments.png" width="500" alt="Example of LabChart screenshot showing respiratory cycles detection with INSPI/EXPI comments">

- Compute ventilatory variables for:
  - **Spontaneous breathing** (Flow inspiration negative).
  - **Mechanical ventilation** (Flow inspiration positive).
- Outputs per-cycle DataFrames with standard metrics (BF, VT, VE, Ti, Te, I:E, PIF, PEF, PTP, WOB).
- For mechanical ventilation, also returns PEEP, Ppeak, ΔP, MAP, etc. (Pplat, Cstat, R left NaN for now in absence of inspiratory hold AND detection). 

## Limitations

- **Plateau pressure (Pplat) detection**: Requires an inspiratory hold maneuver with a period of near-zero flow. Without an adequate plateau, Pplat, static compliance (Cstat), and airway resistance (R) remain NaN.
- **Driving pressure fallback**: When Pplat is unavailable, dP is computed as Ppeak − PEEP. This **overestimates** the true driving pressure because it includes the resistive pressure component. Interpret with caution.
- **Work of breathing (WOB)**: Requires esophageal pressure (Pes) to compute the true pressure-volume work. If only airway pressure (Paw) is available, WOB returns NaN.
- **Pressure-time product (PTP)**: Computed as the integral of pressure above the value at inspiration onset, over the inspiratory phase only. Intended for estimating inspiratory muscle effort when using esophageal or airway pressure.
- **Last cycle of each block**: Intentionally excluded from output because the end of the cycle (next inspiration onset) is undefined.
- **Cycle detection**: Relies on user-placed INSPI/EXPI comments in LabChart. No automatic breath detection is performed—this is deliberate to encourage careful visual inspection of signals.

## Usage

See [`examples/example_usage.py`](examples/example_usage.py) and [`examples/example_notebook.ipynb`](examples/example_notebook.ipynb).

## Installation

You can install in two ways:

### Option 1 – Direct install from GitHub
Install directly into your environment with pip:

```bash
pip install git+https://github.com/Neures-1158/resp_metrics.git
```

### Option 2 – Development mode (editable install)
Clone the repository locally and install in editable mode:

```bash
git clone https://github.com/Neures-1158/resp_metrics.git
cd resp_metrics
pip install -e .
```


## Author

 Damien Bachasson, PhD | Inserm / Sorbonne Université | [![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github)](https://github.com/dambach) [![ORCID](https://img.shields.io/badge/-ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0001-6335-9916) [![Scholar](https://img.shields.io/badge/-Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.fr/citations?user=DNt--nsAAAAJ) |

## References

- Miller, M. R., Hankinson, J., Brusasco, V., Burgos, F., Casaburi, R., Coates, A., Crapo, R., Enright, P., van der Grinten, C. P., Gustafsson, P., Jensen, R., Johnson, D. C., MacIntyre, N., McKay, R., Navajas, D., Pedersen, O. F., Pellegrino, R., Viegi, G., Wanger, J., & ATS/ERS Task Force (2005). Standardisation of spirometry. *European Respiratory Journal*, 26(2), 319–338. https://doi.org/10.1183/09031936.05.00034805

## Contributions

Contributions from lab members, collaborators, and the wider community are very welcome. Please feel free to contribute by submitting issues or pull requests on GitHub.

## License

MIT License
Copyright © 2025 Damien Bachasson
(see the [LICENSE](LICENSE) file for details.)
