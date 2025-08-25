"""
Ventilatory metrics computed cycle-by-cycle.

This module provides a single high-level function:

    ventilatory_from_cycles(df_block, cycles_df, flow_col="Flow", volume_col="VolumeResp")

It returns a DataFrame with one row per cycle and the following columns:
  - n_cycle: 1-based cycle index within the block
  - t_insp, t_expi: absolute times (s) delimiting inspiration (from comments)
  - Ti, Te, Ttot: inspiratory, expiratory and total cycle durations (s)
  - BF: breathing frequency (breaths/min)
  - VT: tidal volume (L)
  - VE: minute ventilation (L/min)
  - PIF, PEF: peak inspiratory/expiratory flow (magnitudes, L/s)
  - IE: I:E ratio (dimensionless), Ti/Te when both are finite

Assumptions:
  - df_block contains at least 'time_abs' and the specified flow/volume columns
  - Flow is expressed in L/min (will be converted to L/s internally)
  - Flow is negative during inspiration (spontaneous breathing convention)
  - PIF/PEF are returned as magnitudes (L/s)
  - cycles_df contains at least 't_insp' and 't_expi' (from cycles_from_comments)

Notes:
  - If `volume_col` is available, VT is computed as ΔVolume on inspiration.
    Otherwise VT is estimated by trapezoidal integration of flow between
    t_insp and t_expi.
  - PEF is computed between t_expi and t_next_insp when available; otherwise NaN.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _nearest_idx(vec: np.ndarray, target: float) -> int:
    """Return index of element in `vec` nearest to `target` (assumes 1D arrays)."""
    return int(np.abs(vec - target).argmin())


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Safe trapezoidal integral; returns NaN when not enough samples."""
    if y.size < 2 or x.size < 2:
        return float("nan")
    return float(np.trapz(y, x))


def ventilatory_from_cycles(
    df_block: pd.DataFrame,
    cycles_df: pd.DataFrame,
    flow_col: str = "Flow",
    volume_col: Optional[str] = "VolumeResp",
) -> pd.DataFrame:
    """Compute ventilatory variables per cycle.

    Parameters
    ----------
    df_block : pandas.DataFrame
        A single-block DataFrame as returned by LabChartFile.get_block_df(b).
        Must contain 'time_abs' and the requested channel columns.
    cycles_df : pandas.DataFrame
        Output of cycles_from_comments, must contain 't_insp' and 't_expi'.
    flow_col : str, default 'Flow'
        Column name for flow signal (L/min).
    volume_col : str or None, default 'VolumeResp'
        Column name for volume signal (L). If None or missing, VT is
        estimated by integrating flow over inspiration.

    Returns
    -------
    pandas.DataFrame
        One row per cycle with columns:
        ['n_cycle','t_insp','t_expi','Ti','Ttot','Te','BF','VT','VE','PIF','PEF','IE']
    """
    # Guard clauses
    if df_block is None or df_block.empty:
        return pd.DataFrame(columns=[
            'n_cycle','t_insp','t_expi','Ti','Ttot','Te','BF','VT','VE','PIF','PEF','IE'
        ])
    if cycles_df is None or cycles_df.empty:
        return pd.DataFrame(columns=[
            'n_cycle','t_insp','t_expi','Ti','Ttot','Te','BF','VT','VE','PIF','PEF','IE'
        ])

    # Required time axis
    if 'time_abs' not in df_block.columns:
        raise KeyError("df_block must contain a 'time_abs' column")

    t = df_block['time_abs'].to_numpy()
    # Optional channels
    has_flow = flow_col in df_block.columns
    has_vol = (volume_col is not None) and (volume_col in df_block.columns)

    flow = df_block[flow_col].to_numpy() if has_flow else None
    if flow is not None:
        # Flow provided in L/min (spontaneous convention: inspiration negative)
        flow = flow.astype(float) / 60.0  # -> L/s, keep sign
    vol = df_block[volume_col].to_numpy() if has_vol else None

    # Determine inspiration time column name accepted: 't_insp' (preferred) or 't_inspi' (legacy)
    ti_col = 't_insp' if 't_insp' in cycles_df.columns else ('t_inspi' if 't_inspi' in cycles_df.columns else None)
    if ti_col is None:
        raise KeyError("cycles_df must contain a 't_insp' (or legacy 't_inspi') column")
    te_col = 't_expi'
    if te_col not in cycles_df.columns:
        raise KeyError("cycles_df must contain a 't_expi' column")

    base_cols = [ti_col, te_col]
    use_cols = base_cols + (['n_cycle'] if 'n_cycle' in cycles_df.columns else [])
    cyc = cycles_df[use_cols].dropna(subset=[ti_col, te_col]).copy()
    cyc = cyc.sort_values(ti_col).reset_index(drop=True)
    if 'n_cycle' not in cyc.columns:
        cyc.insert(0, 'n_cycle', range(1, len(cyc) + 1))
    cyc['t_next_insp'] = cyc[ti_col].shift(-1)

    rows = []
    for i, row in cyc.iterrows():
        ti = float(row[ti_col])
        te = float(row[te_col])
        t_next = row['t_next_insp']

        i_insp = _nearest_idx(t, ti)
        i_expi = _nearest_idx(t, te)

        # Durations
        Ti = float(t[i_expi] - t[i_insp])
        if pd.notna(t_next):
            i_next = _nearest_idx(t, float(t_next))
            Ttot = float(t[i_next] - t[i_insp])
        else:
            i_next = None
            Ttot = float('nan')
        Te = (Ttot - Ti) if np.isfinite(Ttot) else float('nan')
        BF = 60.0 / Ttot if (np.isfinite(Ttot) and Ttot > 0) else float('nan')

        # VT (ΔVolume if available, else integrate Flow)
        if has_vol:
            VT = float(vol[i_expi] - vol[i_insp])
        elif has_flow:
            # integrate flow between i_insp and i_expi (inclusive)
            i0, i1 = sorted((i_insp, i_expi))
            # flow is negative during inspiration -> negate to return positive VT
            VT = -_trapz(flow[i0:i1+1], t[i0:i1+1])
        else:
            VT = float('nan')

        VE = BF * VT if (np.isfinite(BF) and np.isfinite(VT)) else float('nan')

        # Peaks (magnitudes): inspiration negative -> use abs(min) for PIF; expiration positive -> max
        if has_flow:
            i0, i1 = sorted((i_insp, i_expi))
            seg_insp = flow[i0:i1+1]
            PIF = float(abs(np.nanmin(seg_insp))) if seg_insp.size else float('nan')

            if i_next is not None and i_next > i_expi:
                seg_exp = flow[i_expi:i_next+1]
                PEF = float(np.nanmax(seg_exp)) if seg_exp.size else float('nan')
            else:
                PEF = float('nan')
        else:
            PIF = float('nan')
            PEF = float('nan')

        IE = (Ti / Te) if (np.isfinite(Ti) and np.isfinite(Te) and Te > 0) else float('nan')

        rows.append({
            'n_cycle': int(cyc.loc[i, 'n_cycle']),
            't_insp': t[i_insp], 't_expi': t[i_expi],
            'Ti': Ti, 'Ttot': Ttot, 'Te': Te, 'BF': BF,
            'VT': VT, 'VE': VE, 'PIF': PIF, 'PEF': PEF, 'IE': IE,
        })

    return pd.DataFrame(rows)