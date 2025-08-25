# resp_metrics

Cycle-by-cycle ventilatory metrics (BF, VT, VE, Ti, Te, Ttot, I:E, ...) computed from respiratory signals exported from ADInstruments LabChart.

## Features

- Parse LabChart `.txt` exports (via [labchart_parser](https://github.com/Neures-1158/labchart_txt_parser)).
- Extract respiratory cycles from INSPI/EXPI comments obtained using a macro in labchart.
- Compute ventilatory variables for:
  - **Spontaneous breathing** (Flow in L/min, inspiration negative).
  - **Mechanical ventilation** (Flow in L/min, inspiration positive, Pressure available).
- Outputs per-cycle DataFrames with standard metrics (BF, VT, VE, Ti, Te, I:E, PIF, PEF).
- For mechanical ventilation, also returns PEEP, Ppeak, ΔP, MAP, etc. (Pplat, Cstat, R left NaN for now in absence of inspiratory hold AND detection). 

## Installation

You can install in two ways:

### Option 1 – Development mode (editable install)
Clone the repository locally and install in editable mode:

```bash
git clone https://github.com/Neures-1158/resp_metrics.git
cd resp_metrics
pip install -e .
```

### Option 2 – Direct install from GitHub
Install directly into your environment with pip:

```bash
pip install git+https://github.com/Neures-1158/resp_metrics.git
```

## Usage

See [`examples/example_usage.py`](examples/example_usage.py) for full code.

Basic workflow:

```python
from resp_metrics import compute_from_labchart

# Example path to LabChart .txt export
path = "examples/data/33-01-0009-D2-AVI.txt"

# Spontaneous breathing
result = compute_from_labchart(
    path,
    block=1,
    flow_col="Flow",
    mechanically_ventilated=False
)
print(result["ventilatory"].head())

# Mechanical ventilation (requires Pressure channel)
result = compute_from_labchart(
    path,
    block=1,
    flow_col="Flow",
    pressure_col="Pressure",
    mechanically_ventilated=True
)
print(result["ventilatory"].head())
print(result["ventilator"].head())  # mechanical subset
```

## License

[MIT License](LICENSE)

## Contributors & Maintainers

This project is maintained under the [NEURES Lab GitHub organization](https://github.com/Neures-1158).  
Main maintainer: **Damien Bachasson** (author and lead developer).

Contributions from lab members, collaborators, and the wider community are very welcome. Please feel free to contribute by submitting issues or pull requests on GitHub.