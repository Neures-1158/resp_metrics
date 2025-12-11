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
from typing import Optional
import numpy as np
import pandas as pd

from .utils import nearest_idx, trapz_safe, convert_flow_unit


def mechanical_from_cycles(
    df_block: pd.DataFrame,
    cycles_df: pd.DataFrame,
    flow_col: str = "Flow",
    pressure_col: str = "Pressure",
    volume_col: Optional[str] = "VolumeResp",
    *,
    flow_unit: str = "L/min",
    peep_window: float = 0.20,         # seconds before insp
    plateau_flow_thresh: float = 0.05, # |Flow| < threshold = plateau
    plateau_min_dur: float = 0.10      # minimum duration for plateau
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
    
    Returns
    -------
    pandas.DataFrame
        One row per cycle with ventilatory and mechanical variables.
    """
    # Define all output columns for empty DataFrame case
    all_columns = [
        "n_cycle", "t_inspi", "t_expi",
        "Ti", "Ttot", "Te", "BF", "VT", "VE", "PIF", "PEF", "IE",
        "PEEP", "Ppeak", "Pplat", "dP", "Cstat", "R", "MAP"
    ]

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
    P = df_block[pressure_col].to_numpy()
    F = convert_flow_unit(df_block[flow_col].to_numpy(), flow_unit)  # Convert to L/s
    
    has_vol = (volume_col is not None) and (volume_col in df_block.columns)
    V = df_block[volume_col].to_numpy() if has_vol else None

    use_cols = [ti_col, te_col, "t_next_inspi"] + (["n_cycle"] if "n_cycle" in cycles_df.columns else [])
    cyc = cycles_df[use_cols].dropna(subset=[ti_col, te_col]).copy()
    cyc = cyc.sort_values(ti_col).reset_index(drop=True)
    if "n_cycle" not in cyc.columns:
        cyc.insert(0, "n_cycle", range(1, len(cyc) + 1))

    rows = []
    for _, r in cyc.iterrows():
        ncy = int(r["n_cycle"])
        ti, te = float(r[ti_col]), float(r[te_col])
        t_next = r["t_next_inspi"]

        i_insp = nearest_idx(t, ti)
        i_expi = nearest_idx(t, te)
        i0, i1 = sorted((i_insp, i_expi))

        # --- Ventilatory variables (mechanical ventilation: inspiration positive) ---
        Ti = float(t[i_expi] - t[i_insp])
        if pd.notna(t_next):
            i_next = nearest_idx(t, float(t_next))
            Ttot = float(t[i_next] - t[i_insp])
        else:
            i_next = None
            Ttot = float('nan')
        Te = (Ttot - Ti) if np.isfinite(Ttot) else float('nan')
        BF = 60.0 / Ttot if (np.isfinite(Ttot) and Ttot > 0) else float('nan')

        if has_vol:
            VT = float(V[i1] - V[i0])
        else:
            VT = trapz_safe(F[i0:i1+1], t[i0:i1+1])  # insp positive -> VT > 0
        VE = BF * VT if (np.isfinite(BF) and np.isfinite(VT)) else float('nan')

        seg_insp = F[i0:i1+1]
        PIF = float(np.nanmax(seg_insp)) if seg_insp.size else float('nan')

        if i_next is not None and i_next > i_expi:
            seg_exp = F[i_expi:i_next+1]
            PEF = float(abs(np.nanmin(seg_exp))) if seg_exp.size else float('nan')
        else:
            PEF = float('nan')

        IE = (Ti / Te) if (np.isfinite(Ti) and np.isfinite(Te) and Te > 0) else float('nan')

        # --- PEEP: median pressure before insp ---
        t0_peep = max(t[0], ti - peep_window)
        m_peep = (t >= t0_peep) & (t < ti)
        PEEP = float(np.nanmedian(P[m_peep])) if np.any(m_peep) else float("nan")

        # --- Ppeak ---
        Ppeak = float(np.nanmax(P[i0:i1+1])) if i1 > i0 else float("nan")

        # --- Pplat: look for plateau near end-inspiration ---
        # Plateau detection may fail without an inspiratory hold; Pplat will remain NaN in that case.
        insp_dur = max(t[i1] - t[i0], 0.0)
        tail_win = max(0.15, 0.3 * insp_dur)
        t_start_tail = max(t[i0], t[i1] - tail_win)
        m_tail = (t >= t_start_tail) & (t <= t[i1])
        m_low = m_tail & (np.abs(F) <= plateau_flow_thresh)

        Pplat = float("nan")
        if np.any(m_low):
            idx = np.where(m_low)[0]
            gaps = np.where(np.diff(idx) > 1)[0]
            starts = np.r_[0, gaps + 1]
            ends = np.r_[gaps, len(idx) - 1]
            best, best_dur = None, 0.0
            for s, e in zip(starts, ends):
                a, b = idx[s], idx[e]
                dur = t[b] - t[a]
                if dur >= plateau_min_dur and dur > best_dur:
                    best, best_dur = (a, b), dur
            if best is not None:
                a, b = best
                Pplat = float(np.nanmedian(P[a:b+1]))

        # --- Driving pressure ---
        if np.isfinite(Pplat) and np.isfinite(PEEP):
            dP = Pplat - PEEP
        elif np.isfinite(Ppeak) and np.isfinite(PEEP):
            dP = Ppeak - PEEP
        else:
            dP = float("nan")

        # --- Compliance ---
        # Requires a valid Pplat; otherwise remains NaN.
        Cstat = (VT / (Pplat - PEEP)) if (np.isfinite(VT) and np.isfinite(Pplat) and np.isfinite(PEEP) and (Pplat > PEEP)) else float("nan")

        # --- Resistance ---
        # Requires a valid Pplat; otherwise remains NaN.
        if np.isfinite(Ppeak) and np.isfinite(Pplat) and (Ppeak > Pplat):
            PIF_res = float(np.nanmax(F[i0:i1+1])) if i1 > i0 else float("nan")
            R = ((Ppeak - Pplat) / PIF_res) if (np.isfinite(PIF_res) and PIF_res > 0) else float("nan")
        else:
            R = float("nan")

        # --- Mean airway pressure ---
        if pd.notna(t_next):
            i_end = max(i_expi, i_next)
        else:
            i_end = i_expi
        i2 = max(i_insp, 0)
        i3 = min(i_end, len(t) - 1)
        MAP_int = trapz_safe(P[i2:i3+1], t[i2:i3+1])
        Ttot_map = t[i3] - t[i2] if i3 > i2 else float("nan")
        MAP = (MAP_int / Ttot_map) if (np.isfinite(MAP_int) and np.isfinite(Ttot_map) and Ttot_map > 0) else float("nan")

        rows.append({
            # Common identifiers
            "n_cycle": ncy,
            "t_inspi": t[i_insp],
            "t_expi": t[i_expi],
            # Ventilatory variables
            "Ti": Ti, "Ttot": Ttot, "Te": Te, "BF": BF,
            "VT": VT, "VE": VE, "PIF": PIF, "PEF": PEF, "IE": IE,
            # Mechanical variables
            "PEEP": PEEP, "Ppeak": Ppeak, "Pplat": Pplat, "dP": dP,
            "Cstat": Cstat, "R": R, "MAP": MAP,
        })

    return pd.DataFrame(rows)