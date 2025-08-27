"""
Minimal example usage of resp_metrics.

This script demonstrates how to:
1. Load a LabChart text export using labchart_parser.
2. Extract cycles based on INSPI/EXPI comments.
3. Compute ventilatory metrics for spontaneous breathing.
4. Use the high-level API for both spontaneous and mechanical ventilation.

Usage:
    python examples/data/example_usage.py
"""

import sys
from pathlib import Path

# Ensure local src/ is importable without installation
# ROOT = Path(__file__).resolve().parents[2]
# SRC = ROOT / "src"
# if SRC.exists():
#     sys.path.insert(0, str(SRC))

from labchart_parser import LabChartFile
from resp_metrics import cycles_from_comments

def main():
    path_mv = "examples/data/labchart_file.example.txt"
    path_vs = "examples/data/labchart_file_vs.example.txt"

    # --- Load LabChart file ---
    lc = LabChartFile.from_file(path_vs)
    
    print("=== LabChart File Information ===")
    print("Metadata:", lc.metadata)
    print("Channels:", lc.channels)
    print("Comments:", lc.comments)
    print()

    # --- Extract cycles from INSPI/EXPI comments ---
    cycles = cycles_from_comments(lc.comments, block=1,
                                  insp_label="INSPI", expi_label="EXPI")
    print("=== Detected Cycles ===")
    print(cycles.head())

    # --- Compute ventilatory metrics for spontaneous breathing ---
    from resp_metrics import ventilatory_from_cycles

    metrics = ventilatory_from_cycles(
        lc.get_block_df(1),
        cycles,
        flow_col="Flow",
        volume_col=None,
        pressure_col= "Pressure",
        flow_unit="L/s"
    )
    print("=== Ventilatory Metrics (Spontaneous Breathing) ===")
    print(metrics.head())

    # --- High-level API usage ---

    from resp_metrics import compute_from_labchart

    # Spontaneous breathing example
    result_spont = compute_from_labchart(
        path_vs,
        block=1,
        flow_col="Flow",
        flow_unit="L/s",
        volume_col=None,
        pressure_col="Pressure",               # No pressure channel for spontaneous
        mechanically_ventilated=False
    )
    print("=== High-level API Results: Spontaneous Breathing ===")
    print("Cycles head:\n", result_spont["cycles"].head())
    print("Ventilatory metrics head:\n", result_spont["ventilatory"].head())
    if result_spont["ventilator"] is not None:
        print("Ventilator mechanics head:\n", result_spont["ventilator"].head())
    else:
        print("No ventilator metrics (spontaneous or no pressure channel).")
    print()


    # Mechanical ventilation example using mechanical_from_cycles
    from resp_metrics import mechanical_from_cycles
    lc_mv = LabChartFile.from_file(path_mv)
    cycles_mv = cycles_from_comments(lc_mv.comments, block=2,
                                      insp_label="INSPI", expi_label="EXPI")
    print(cycles_mv)
    result_mech = mechanical_from_cycles(
        lc_mv.get_block_df(1),
        cycles_mv,
        flow_col="Flow",
        volume_col=None,
        pressure_col= "Pressure")
    print(result_mech)

    # Mechanical ventilation example using high-level API
    result_mech = compute_from_labchart(
        path_mv,
        block=1,
        flow_col="Flow",
        flow_unit="L/min",
        volume_col=None,
        pressure_col="Pressure",
        mechanically_ventilated=True
    )
    print("=== High-level API Results: Mechanical Ventilation ===")
    print("Cycles head:\n", result_mech["cycles"].head())
    print("Ventilatory metrics head:\n", result_mech["ventilatory"].head())
    if result_mech["ventilator"] is not None:
        print("Ventilator mechanics head:\n", result_mech["ventilator"].head())
    else:
        print("No ventilator metrics (spontaneous or no pressure channel).")
    print()

if __name__ == "__main__":
    main()