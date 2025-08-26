"""High-level convenience API for non-developers.

This module exposes a single function, :func:`compute_from_labchart`,
that orchestrates the full pipeline with *explicit* channel names and
comment-based cycles only. No auto-detection is performed.

Workflow:
1) Load a LabChart .txt export via labchart_parser.LabChartFile
2) Build cycles from INSPI/EXPI comments
3) Always compute ventilatory metrics from flow/volume
4) Optionally compute ventilator mechanics (PEEP, Pplat, etc.) if
   `mechanically_ventilated=True` *and* a pressure channel name is given
   *and* the mechanical_vent module is available.
"""

from __future__ import annotations
from typing import Optional, Dict

from labchart_parser import LabChartFile
from .cycles import cycles_from_comments
from .ventilatory import ventilatory_from_cycles

try:
    # Optional mechanical ventilation metrics
    from .mechanical_vent import mechanical_from_cycles
    _HAS_VENTILATOR = True
except Exception:  # pragma: no cover - absence is allowed
    mechanical_from_cycles = None  # type: ignore
    _HAS_VENTILATOR = False


def compute_from_labchart(
    path: str,
    *,
    block: int = 1,
    flow_col: str,
    flow_unit: str,
    volume_col: Optional[str] = None,
    pressure_col: Optional[str] = None,
    mechanically_ventilated: bool = False,
    insp_label: str = "INSPI",
    expi_label: str = "EXPI",
) -> Dict[str, object]:
    """One-call pipeline with explicit channels and comment-based cycles.

    Parameters
    ----------
    path : str
        Path to LabChart .txt export.
    block : int, default 1
        Block index to analyze.
    flow_col : str (required)
        Name of the flow column (L/min).
    volume_col : str or None, default None
        Name of the volume column (L). If None, VT will be integrated from flow.
    pressure_col : str or None, default None
        Name of the pressure column (cmH2O). Required only if ventilator mechanics
        are requested.
    mechanically_ventilated : bool, default False
        If True and pressure_col is provided (and ventilator module available),
        compute ventilator mechanics (PEEP, Pplat, dP, Cstat, R, MAP). Otherwise skip.
    insp_label, expi_label : str
        Comment labels used to build cycles (default "INSPI"/"EXPI").

    Returns
    -------
    dict
        {
          'meta': metadata dict,
          'cycles': DataFrame (columns: n_cycle, t_insp, t_expi),
          'ventilatory': DataFrame (per-cycle ventilatory variables),
          'ventilator': DataFrame or None (per-cycle ventilator mechanics)
        }
    """
    # 1) Load and select block
    lc = LabChartFile.from_file(path)
    df_block = lc.get_block_df(block)

    # 2) Cycles from comments (strictly)
    cycles = cycles_from_comments(
        lc.comments,
        block=block,
        insp_label=insp_label,
        expi_label=expi_label,
    )

    if mechanically_ventilated and _HAS_VENTILATOR and pressure_col is not None:
        # Mechanical ventilation path: compute combined ventilatory+mechanical table
        vent = mechanical_from_cycles(  # type: ignore[misc]
            df_block,
            cycles,
            flow_col=flow_col,
            pressure_col=pressure_col,
            volume_col=volume_col,
        )
        # Extract only mechanical columns for the 'ventilator' view
        mech_cols = [c for c in [
            'n_cycle','t_inspi','t_expi','PEEP','Ppeak','Pplat','dP','Cstat','R','MAP'
        ] if c in vent.columns]
        ventmech = vent[mech_cols].copy() if mech_cols else None
    else:
        # Spontaneous path: standard ventilatory variables only
        vent = ventilatory_from_cycles(
            df_block,
            cycles,
            flow_col=flow_col,
            pressure_col=pressure_col,
            volume_col=volume_col,
            flow_unit=flow_unit,
        )
        ventmech = None

    return {
        "meta": lc.metadata,
        "cycles": cycles,
        "ventilatory": vent,
        "ventilator": ventmech,
    }
