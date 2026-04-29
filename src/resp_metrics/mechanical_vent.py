"""
Mechanical ventilation mechanics computed cycle-by-cycle.

This module provides:

    mechanical_from_cycles(df_block, cycles_df, flow_col="Flow",
                           pressure_col="Pressure", volume_col="VolumeResp")

It returns a DataFrame with one row per cycle and the following columns:
  - n_cycle: 1-based cycle index
  - t_inspi: absolute time of inspiration onset
  - t_expi: absolute time of expiration onset
  - Ti, Te, Ttot: inspiratory, expiratory and total cycle durations (s)
  - BF: breathing frequency (breaths/min)
  - VT: tidal volume (L)
  - VE: minute ventilation (L/min)
  - PIF, PEF: peak inspiratory/expiratory flow (L/s)
  - IE: I:E ratio (dimensionless)
  - PEEP: positive end-expiratory pressure (cmH2O)
  - Ppeak: peak inspiratory pressure (cmH2O)
  - Pplat: plateau pressure (cmH2O), if low-flow plateau detected (depends on presence of an inspiratory hold; may be NaN)
  - dP: driving pressure (Pplat - PEEP) (cmH2O). **WARNING**: When Pplat is unavailable
        (no inspiratory hold), a fallback value (Ppeak - PEEP) is returned; this
        OVERESTIMATES the true driving pressure as it includes the resistive component.
        Interpretation of fallback dP values requires caution.
  - Cstat: static compliance (VT / (Pplat - PEEP), in L/cmH2O) (depends on Pplat)
  - R: airway resistance estimate ((Ppeak - Pplat)/PIF, cmH2O·s/L) (depends on Pplat)
  - MAP: mean airway pressure over the cycle (cmH2O)

Notes:
  - Flow is assumed to be POSITIVE during inspiration (mechanical ventilation convention).
  - Flow unit can be specified via flow_unit parameter (default: L/min).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import convert_flow_unit, nearest_idx, trapz_safe

__all__ = ["mechanical_from_cycles"]


def mechanical_from_cycles(
    df_block: pd.DataFrame,
    cycles_df: pd.DataFrame,
    flow_col: str = "Flow",
    pressure_col: str = "Pressure",
    volume_col: str | None = "VolumeResp",
    *,
    flow_unit: str = "L/min",
    peep_window: float = 0.20,  # seconds before insp
    plateau_flow_thresh: float = 0.05,  # |Flow| < threshold = plateau
    plateau_min_dur: float = 0.10,  # minimum duration for plateau
    block: int | str | None = None,
    block_name: str | None = None,
) -> pd.DataFrame:
    """Compute mechanical ventilation metrics per cycle.

    Parameters
    ----------
    df_block : pandas.DataFrame
        A single-block DataFrame with 'time_block' and channel columns.
    cycles_df : pandas.DataFrame
        Output of cycles_from_comments with 't_inspi', 't_expi', 't_next_inspi'.
    flow_col : str, default 'Flow'
        Column name for flow signal.
    pressure_col : str, default 'Pressure'
        Column name for airway pressure signal (cmH2O).
    volume_col : str or None, default 'VolumeResp'
        Column name for volume signal (L). If None, VT is integrated from flow.
    flow_unit : str, default 'L/min'
        Unit of the flow signal. Accepted values are 'L/min' and 'L/s'.
    peep_window : float, default 0.20
        Window (seconds) before inspiration to compute PEEP.
    plateau_flow_thresh : float, default 0.05
        Flow threshold (L/s) below which plateau is detected.
    plateau_min_dur : float, default 0.10
        Minimum duration (seconds) for a valid plateau.
    block : int or str or None, default None
        Optional block identifier to prepend as a ``block`` column.
    block_name : str or None, default None
        Optional block name to prepend as a ``block_name`` column.

    Returns
    -------
    pandas.DataFrame
        One row per cycle with ventilatory and mechanical variables. If provided,
        a ``block`` column is prepended.

    Raises
    ------
    KeyError
        If ``cycles_df`` is missing ``t_inspi``, ``t_expi``, or
        ``t_next_inspi``.
    """
    # Define all output columns for empty DataFrame case
    all_columns = [
        "n_cycle",
        "t_inspi",
        "t_expi",
        "Ti",
        "Ttot",
        "Te",
        "BF",
        "VT",
        "VE",
        "PIF",
        "PEF",
        "IE",
        "PEEP",
        "Ppeak",
        "Pplat",
        "dP",
        "dP_fallback",
        "Cstat",
        "R",
        "MAP",
    ]
    include_block = block is not None or (
        cycles_df is not None and "block" in cycles_df.columns
    )
    include_block_name = block_name is not None or (
        cycles_df is not None and "block_name" in cycles_df.columns
    )
    if include_block:
        all_columns = ["block"] + all_columns
    if include_block_name:
        all_columns = ["block_name"] + all_columns

    needed = {"time_block", pressure_col, flow_col}
    if df_block is None or df_block.empty or not needed.issubset(df_block.columns):
        return pd.DataFrame(columns=all_columns)
    if cycles_df is None or cycles_df.empty:
        return pd.DataFrame(columns=all_columns)

    # Use t_inspi for inspiration time
    ti_col = "t_inspi"
    if ti_col not in cycles_df.columns:
        raise KeyError("cycles_df must contain 't_inspi'")
    te_col = "t_expi"
    if te_col not in cycles_df.columns:
        raise KeyError("cycles_df must contain 't_expi'")
    if "t_next_inspi" not in cycles_df.columns:
        raise KeyError("cycles_df must contain 't_next_inspi'")

    t = df_block["time_block"].to_numpy()
    pressure = df_block[pressure_col].to_numpy()
    flow = convert_flow_unit(df_block[flow_col].to_numpy(), flow_unit)  # Convert to L/s

    has_vol = (volume_col is not None) and (volume_col in df_block.columns)
    volume = df_block[volume_col].to_numpy() if has_vol else None

    use_cols = [ti_col, te_col, "t_next_inspi"] + (
        ["n_cycle"] if "n_cycle" in cycles_df.columns else []
    )
    cyc = cycles_df[use_cols].dropna(subset=[ti_col, te_col]).copy()
    cyc = cyc.sort_values(ti_col).reset_index(drop=True)
    if "n_cycle" not in cyc.columns:
        cyc.insert(0, "n_cycle", range(1, len(cyc) + 1))

    block_value = block
    if block_value is None and "block" in cycles_df.columns and not cycles_df.empty:
        block_value = cycles_df["block"].iloc[0]
    block_name_value = block_name
    if (
        block_name_value is None
        and "block_name" in cycles_df.columns
        and not cycles_df.empty
    ):
        block_name_value = cycles_df["block_name"].iloc[0]

    rows = []
    for _, r in cyc.iterrows():
        ncy = int(r["n_cycle"])
        ti, te = float(r[ti_col]), float(r[te_col])
        if te <= ti:
            # Invalid cycle ordering
            continue
        t_next = r["t_next_inspi"]

        i_insp = nearest_idx(t, ti)
        i_expi = nearest_idx(t, te)
        i0, i1 = sorted((i_insp, i_expi))
        if i1 <= i0:
            # Invalid or zero-length inspiration window
            continue

        # --- Ventilatory variables (mechanical ventilation: inspiration positive) ---
        ti_duration = float(t[i_expi] - t[i_insp])
        if pd.notna(t_next):
            i_next = nearest_idx(t, float(t_next))
            if i_next > i_expi:
                ttot = float(t[i_next] - t[i_insp])
            else:
                i_next = None
                t_next = float("nan")
                ttot = float("nan")
        else:
            i_next = None
            ttot = float("nan")
        te_duration = ttot - ti_duration if np.isfinite(ttot) else float("nan")
        bf = 60.0 / ttot if (np.isfinite(ttot) and ttot > 0) else float("nan")

        if has_vol:
            vt = float(volume[i1] - volume[i0])
        else:
            vt = trapz_safe(flow[i0 : i1 + 1], t[i0 : i1 + 1])
        ve = bf * vt if (np.isfinite(bf) and np.isfinite(vt)) else float("nan")

        seg_insp = flow[i0 : i1 + 1]
        pif = float(np.nanmax(seg_insp)) if seg_insp.size else float("nan")

        if i_next is not None and i_next > i_expi:
            seg_exp = flow[i_expi : i_next + 1]
            pef = float(abs(np.nanmin(seg_exp))) if seg_exp.size else float("nan")
        else:
            pef = float("nan")

        ie_ratio = (
            (ti_duration / te_duration)
            if (
                np.isfinite(ti_duration)
                and np.isfinite(te_duration)
                and te_duration > 0
            )
            else float("nan")
        )

        # --- PEEP: median pressure before insp ---
        t0_peep = max(t[0], ti - peep_window)
        m_peep = (t >= t0_peep) & (t < ti)
        peep = float(np.nanmedian(pressure[m_peep])) if np.any(m_peep) else float("nan")

        # --- Ppeak ---
        ppeak = float(np.nanmax(pressure[i0 : i1 + 1])) if i1 > i0 else float("nan")

        # --- Pplat: look for plateau near end-inspiration ---
        # Plateau detection may fail without an inspiratory hold; Pplat will remain NaN in that case.
        insp_dur = max(t[i1] - t[i0], 0.0)
        tail_win = max(0.15, 0.3 * insp_dur)
        t_start_tail = max(t[i0], t[i1] - tail_win)
        m_tail = (t >= t_start_tail) & (t <= t[i1])
        m_low = m_tail & (np.abs(flow) <= plateau_flow_thresh)

        pplat = float("nan")
        if np.any(m_low):
            idx = np.where(m_low)[0]
            gaps = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[0, gaps + 1]
            ends = np.r_[gaps, len(idx) - 1]
            best, best_dur = None, 0.0
            for s, e in zip(starts, ends, strict=False):
                a, b = idx[s], idx[e]
                dur = t[b] - t[a]
                if dur >= plateau_min_dur and dur > best_dur:
                    best, best_dur = (a, b), dur
            if best is not None:
                a, b = best
                pplat = float(np.nanmedian(pressure[a : b + 1]))

        # --- Driving pressure ---
        # dP_fallback=True means Pplat was unavailable and Ppeak - PEEP was used.
        # Fallback dP overestimates true driving pressure (includes resistive component).
        if np.isfinite(pplat) and np.isfinite(peep):
            dp = pplat - peep
            dp_fallback = False
        elif np.isfinite(ppeak) and np.isfinite(peep):
            dp = ppeak - peep
            dp_fallback = True
        else:
            dp = float("nan")
            dp_fallback = False

        # --- Compliance ---
        # Requires a valid Pplat; otherwise remains NaN.
        cstat = (
            (vt / (pplat - peep))
            if (
                np.isfinite(vt)
                and np.isfinite(pplat)
                and np.isfinite(peep)
                and (pplat > peep)
            )
            else float("nan")
        )

        # --- Resistance ---
        # Requires a valid Pplat; otherwise remains NaN.
        if np.isfinite(ppeak) and np.isfinite(pplat) and (ppeak > pplat):
            pif_res = float(np.nanmax(flow[i0 : i1 + 1])) if i1 > i0 else float("nan")
            resistance = (
                ((ppeak - pplat) / pif_res)
                if (np.isfinite(pif_res) and pif_res > 0)
                else float("nan")
            )
        else:
            resistance = float("nan")

        # --- Mean airway pressure ---
        # MAP requires the full cycle window (inspiration + expiration).
        # Without t_next_inspi the denominator is unknown; return NaN rather than
        # silently compute MAP over inspiration only (which overestimates MAP).
        if pd.notna(t_next) and i_next is not None and i_next > i_expi:
            i2 = max(i_insp, 0)
            i3 = min(i_next, len(t) - 1)
            map_int = trapz_safe(pressure[i2 : i3 + 1], t[i2 : i3 + 1])
            ttot_map = t[i3] - t[i2] if i3 > i2 else float("nan")
            map_pressure = (
                (map_int / ttot_map)
                if (np.isfinite(map_int) and np.isfinite(ttot_map) and ttot_map > 0)
                else float("nan")
            )
        else:
            map_pressure = float("nan")

        rows.append(
            {
                # Common identifiers
                "n_cycle": ncy,
                "t_inspi": t[i_insp],
                "t_expi": t[i_expi],
                # Ventilatory variables
                "Ti": ti_duration,
                "Ttot": ttot,
                "Te": te_duration,
                "BF": bf,
                "VT": vt,
                "VE": ve,
                "PIF": pif,
                "PEF": pef,
                "IE": ie_ratio,
                # Mechanical variables
                "PEEP": peep,
                "Ppeak": ppeak,
                "Pplat": pplat,
                "dP": dp,
                "dP_fallback": dp_fallback,
                "Cstat": cstat,
                "R": resistance,
                "MAP": map_pressure,
            }
        )

    if not rows:
        return pd.DataFrame(columns=all_columns)

    out = pd.DataFrame(rows)
    if block_value is not None:
        out.insert(0, "block", block_value)
    if block_name_value is not None:
        out.insert(0, "block_name", block_name_value)
    return out
