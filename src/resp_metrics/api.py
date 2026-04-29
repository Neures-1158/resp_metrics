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
   *and* the mechanical_vent module is available. (experimental)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from labchart_parser import LabChartFile

from .cycles import cycles_from_comments
from .ventilatory import ventilatory_from_cycles

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["compute_from_labchart"]

try:
    # Optional mechanical ventilation metrics
    from .mechanical_vent import mechanical_from_cycles

    _HAS_VENTILATOR = True
    _VENTILATOR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - absence is allowed
    mechanical_from_cycles = None  # type: ignore
    _HAS_VENTILATOR = False
    _VENTILATOR_IMPORT_ERROR = exc


def _process_single_block(
    lc: LabChartFile,
    block: int,
    *,
    flow_col: str,
    flow_unit: str,
    volume_col: str | None,
    pressure_col: str | None,
    pes_col: str | None,
    mechanically_ventilated: bool,
    insp_label: str,
    expi_label: str,
    output_dir: str | Path | None,
    output_prefix: str | None,
    block_name: str,
) -> dict[str, object]:
    """Process a single block and return results.

    Parameters
    ----------
    lc : LabChartFile
        The loaded LabChart file.
    block : int
        Block index to analyze.
    flow_col : str
        Name of the flow column.
    flow_unit : str
        Unit of the flow signal.
    volume_col : str or None
        Name of the volume column.
    pressure_col : str or None
        Name of the airway pressure column.
    pes_col : str or None
        Name of the esophageal pressure column.
    mechanically_ventilated : bool
        Whether to compute ventilator mechanics.
    insp_label, expi_label : str
        Comment labels for cycle detection.
    output_dir : str or Path or None
        If provided, save cycles and metrics to CSV files in this directory.
    output_prefix : str or None
        Prefix for output filenames.

    Returns
    -------
    dict
        {
          'cycles': DataFrame,
          'ventilatory': DataFrame,
          'ventilator': DataFrame or None
        }
    """
    # Get block data
    df_block = lc.get_block_df(block)

    # Cycles from comments (strictly)
    cycles = cycles_from_comments(
        lc.comments,
        block=block,
        insp_label=insp_label,
        expi_label=expi_label,
        block_name=block_name,
    )
    cycles = _with_block_columns(cycles, block, block_name)

    if mechanically_ventilated and pressure_col is None:
        warnings.warn(
            "mechanically_ventilated=True but pressure_col is None; "
            "falling back to ventilatory metrics only.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif mechanically_ventilated and not _HAS_VENTILATOR:
        detail = (
            f" Import error: {_VENTILATOR_IMPORT_ERROR}"
            if _VENTILATOR_IMPORT_ERROR is not None
            else ""
        )
        warnings.warn(
            "mechanically_ventilated=True but mechanical ventilation metrics "
            f"are unavailable; falling back to ventilatory metrics only.{detail}",
            RuntimeWarning,
            stacklevel=2,
        )

    if mechanically_ventilated and _HAS_VENTILATOR and pressure_col is not None:
        # Mechanical ventilation path: compute combined ventilatory+mechanical table
        vent = mechanical_from_cycles(  # type: ignore[misc]
            df_block,
            cycles,
            flow_col=flow_col,
            pressure_col=pressure_col,
            volume_col=volume_col,
            flow_unit=flow_unit,
            block=block,
            block_name=block_name,
        )
        vent = _with_block_columns(vent, block, block_name)
        # Extract only mechanical columns for the 'ventilator' view
        mech_cols = [
            c
            for c in [
                "n_cycle",
                "t_inspi",
                "t_expi",
                "PEEP",
                "Ppeak",
                "Pplat",
                "dP",
                "Cstat",
                "R",
                "MAP",
            ]
            if c in vent.columns
        ]
        ventmech = vent[mech_cols].copy() if mech_cols else None
        ventmech = _with_block_columns(ventmech, block, block_name)
    else:
        # Spontaneous path: standard ventilatory variables only
        vent = ventilatory_from_cycles(
            df_block,
            cycles,
            flow_col=flow_col,
            pressure_col=pressure_col,
            pes_col=pes_col,
            volume_col=volume_col,
            flow_unit=flow_unit,
            block=block,
            block_name=block_name,
        )
        vent = _with_block_columns(vent, block, block_name)
        ventmech = None

    _save_outputs(
        output_dir=output_dir,
        output_prefix=output_prefix,
        block=block,
        cycles=cycles,
        ventilatory=vent,
        ventilator=ventmech,
    )

    return {
        "cycles": cycles,
        "ventilatory": vent,
        "ventilator": ventmech,
    }


def _with_block_column(df: pd.DataFrame | None, block: int) -> pd.DataFrame | None:
    """Deprecated: use _with_block_columns."""
    return _with_block_columns(df, block, f"Block {block}")


def _with_block_columns(
    df: pd.DataFrame | None, block: int, block_name: str
) -> pd.DataFrame | None:
    """Ensure 'block_name' and 'block' columns are present and first."""
    if df is None:
        return None
    if "block" not in df.columns:
        df = df.copy()
        df.insert(0, "block", block)
    if "block_name" not in df.columns:
        df = df.copy()
        df.insert(0, "block_name", block_name)
    cols = ["block_name", "block"] + [
        c for c in df.columns if c not in ("block_name", "block")
    ]
    return df[cols]


def _save_outputs(
    *,
    output_dir: str | Path | None,
    output_prefix: str | None,
    block: int,
    cycles: pd.DataFrame,
    ventilatory: pd.DataFrame,
    ventilator: pd.DataFrame | None,
) -> None:
    """Optionally save outputs to CSV."""
    if output_dir is None:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or "resp_metrics"
    block_tag = f"block{block}"
    cycles.to_csv(out_dir / f"{prefix}_cycles_{block_tag}.csv", index=False)
    ventilatory.to_csv(out_dir / f"{prefix}_metrics_{block_tag}.csv", index=False)
    if ventilator is not None:
        ventilator.to_csv(out_dir / f"{prefix}_ventilator_{block_tag}.csv", index=False)


def compute_from_labchart(
    path: str | os.PathLike[str],
    *,
    block: int | list[int] | None = 1,
    flow_col: str,
    flow_unit: str = "L/min",
    volume_col: str | None = None,
    pressure_col: str | None = None,
    pes_col: str | None = None,
    mechanically_ventilated: bool = False,
    insp_label: str = "INSPI",
    expi_label: str = "EXPI",
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    block_names: dict[int, str] | None = None,
) -> dict[str, object]:
    """One-call pipeline with explicit channels and comment-based cycles.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to LabChart .txt export.
    block : int, list of int, or None, default 1
        Block index(es) to analyze.
        - int: analyze a single block, return DataFrames directly
        - list of int: analyze specified blocks, return dict keyed by block number
        - None: analyze ALL blocks in the file, return dict keyed by block number
    flow_col : str (required)
        Name of the flow column.
    flow_unit : str, default 'L/min'
        Unit of the flow signal. Accepted values are 'L/min' and 'L/s'.
    volume_col : str or None, default None
        Name of the volume column (L). If None, VT will be integrated from flow.
    pressure_col : str or None, default None
        Name of the airway pressure column (cmH2O). Required for PTP calculation
        and for ventilator mechanics if mechanically_ventilated=True.
    pes_col : str or None, default None
        Name of the esophageal pressure column (cmH2O). Required for WOB
        calculation. If not provided, WOB will be NaN.
    mechanically_ventilated : bool, default False
        If True and pressure_col is provided (and ventilator module available),
        compute ventilator mechanics (PEEP, Pplat, dP, Cstat, R, MAP). Otherwise skip.
    insp_label, expi_label : str
        Comment labels used to build cycles (default "INSPI"/"EXPI").
    output_dir : str or Path or None, default None
        If provided, save cycles and metrics to CSV files in this directory.
    output_prefix : str or None, default None
        Prefix for output filenames. Defaults to the input file stem.
    block_names : dict or None, default None
        Optional mapping from block number to block name. If not provided, a
        default name "Block {block}" is used.

    Returns
    -------
    dict
        For single block (int):
        {
          'meta': metadata dict,
          'cycles': DataFrame (columns: n_cycle, t_insp, t_expi),
          'ventilatory': DataFrame (per-cycle ventilatory variables),
          'ventilator': DataFrame or None (per-cycle ventilator mechanics)
        }

        For multiple blocks (list or None):
        {
          'meta': metadata dict,
          'cycles': {block_num: DataFrame, ...},
          'ventilatory': {block_num: DataFrame, ...},
          'ventilator': {block_num: DataFrame or None, ...}
        }
        All output DataFrames include leading ``block_name`` and ``block`` columns.
    """
    # Load the file
    lc = LabChartFile.from_file(path)

    if output_dir is not None and output_prefix is None:
        output_prefix = Path(path).stem

    # Determine which blocks to process
    if block is None:
        # Get all available blocks from the file
        blocks_to_process = sorted(lc.blocks)
    elif isinstance(block, list):
        blocks_to_process = block
    else:
        # Single block (int) - use original behavior
        result = _process_single_block(
            lc,
            block,
            flow_col=flow_col,
            flow_unit=flow_unit,
            volume_col=volume_col,
            pressure_col=pressure_col,
            pes_col=pes_col,
            mechanically_ventilated=mechanically_ventilated,
            insp_label=insp_label,
            expi_label=expi_label,
            output_dir=output_dir,
            output_prefix=output_prefix,
            block_name=(
                block_names.get(block, f"Block {block}")
                if block_names
                else f"Block {block}"
            ),
        )
        return {
            "meta": lc.metadata,
            "cycles": result["cycles"],
            "ventilatory": result["ventilatory"],
            "ventilator": result["ventilator"],
        }

    # Process multiple blocks
    cycles_dict = {}
    ventilatory_dict = {}
    ventilator_dict = {}

    for blk in blocks_to_process:
        result = _process_single_block(
            lc,
            blk,
            flow_col=flow_col,
            flow_unit=flow_unit,
            volume_col=volume_col,
            pressure_col=pressure_col,
            pes_col=pes_col,
            mechanically_ventilated=mechanically_ventilated,
            insp_label=insp_label,
            expi_label=expi_label,
            output_dir=output_dir,
            output_prefix=output_prefix,
            block_name=(
                block_names.get(blk, f"Block {blk}") if block_names else f"Block {blk}"
            ),
        )
        cycles_dict[blk] = result["cycles"]
        ventilatory_dict[blk] = result["ventilatory"]
        ventilator_dict[blk] = result["ventilator"]

    return {
        "meta": lc.metadata,
        "cycles": cycles_dict,
        "ventilatory": ventilatory_dict,
        "ventilator": ventilator_dict,
    }
