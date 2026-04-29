"""
Ventilatory metrics computed cycle-by-cycle.

This module provides a single high-level function:

    ventilatory_from_cycles(df_block, cycles_df, flow_col="Flow", volume_col="VolumeResp")

It returns a DataFrame with one row per cycle and the following columns:
  - n_cycle: 1-based cycle index within the block
  - t_inspi, t_expi: absolute times (s) delimiting inspiration (from comments)
  - Ti, Te, Ttot: inspiratory, expiratory and total cycle durations (s)
  - BF: breathing frequency (breaths/min)
  - VT: tidal volume (L)
  - VE: minute ventilation (L/min)
  - PIF, PEF: peak inspiratory/expiratory flow (magnitudes, L/s)
  - IE: I:E ratio (dimensionless), Ti/Te when both are finite
  - WOB: work of breathing (J) — requires esophageal pressure (Pes)
  - PTP: pressure-time product (cmH2O·s) — positive when pressure falls below baseline

Assumptions:
  - df_block contains at least 'time_block' and the specified flow/volume columns
  - Flow can be in L/min, L/s, mL/s, or mL/min (specified by flow_unit parameter)
  - Flow is negative during inspiration (spontaneous breathing convention)
  - PIF/PEF are returned as magnitudes (L/s)
  - cycles_df contains 't_inspi', 't_expi', and 't_next_inspi' (from cycles_from_comments)

Notes:
  - If `volume_col` is available, VT is computed as ΔVolume on inspiration;
    the volume signal must increase during inspiration to return positive VT.
    If VT is negative (volume decreases during inspiration, as happens when
    VolumeResp is the integral of negative-convention flow), a UserWarning is
    emitted and the result falls back to flow integration.
    Pass volume_col=None to always use flow integration.
  - PEF is computed between t_expi and t_next_inspi when available; otherwise NaN.
  - WOB requires esophageal pressure (pes_col). Pes must follow the standard
    subatmospheric convention: negative at rest (e.g. -5 cmH2O), more negative
    during inspiratory effort (e.g. -15 cmH2O). The formula is:
      WOB = ∫ (Pes_baseline − Pes) × (−Flow) dt
    where Pes_baseline is the median Pes in the pre-inspiratory window.
    WOB is positive when the patient generates inspiratory effort.
    If pes_col is not provided, WOB will be NaN.
  - PTP = ∫ (P_baseline − P) dt during inspiration, where P_baseline is the
    median pressure in a window before inspiration onset. PTP is positive when
    airway pressure falls below baseline (spontaneous inspiratory effort).
    If no samples are available in the baseline window, pressure at inspiration
    onset is used.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .utils import convert_flow_unit, nearest_idx, trapz_safe

__all__ = ["ventilatory_from_cycles"]


def ventilatory_from_cycles(
    df_block: pd.DataFrame,
    cycles_df: pd.DataFrame,
    flow_col: str = "Flow",
    volume_col: str | None = "VolumeResp",
    pressure_col: str | None = "Paw",
    pes_col: str | None = None,
    flow_unit: str = "L/min",
    ptp_window: float = 0.20,
    block: int | str | None = None,
    block_name: str | None = None,
) -> pd.DataFrame:
    """Compute ventilatory variables per cycle.

    Parameters
    ----------
    df_block : pandas.DataFrame
        A single-block DataFrame as returned by LabChartFile.get_block_df(b).
        Must contain 'time_block' and the requested channel columns.
    cycles_df : pandas.DataFrame
        Output of cycles_from_comments, must contain 't_inspi' and 't_expi'.
    flow_col : str, default 'Flow'
        Column name for flow signal.
    volume_col : str or None, default 'VolumeResp'
        Column name for volume signal (L). If present, VT is computed as
        ``volume[t_expi] - volume[t_inspi]``; the signal must increase during
        inspiration to return positive VT. If None or missing, VT is estimated
        by integrating flow over inspiration.
    pressure_col : str or None, default 'Paw'
        Column name for airway pressure signal (cmH2O). Used for PTP calculation.
    pes_col : str or None, default None
        Column name for esophageal pressure signal (cmH2O). Required for WOB
        calculation. If not provided, WOB will be NaN. Using airway pressure
        (Paw) for WOB would not represent patient effort correctly.
    flow_unit : str, default 'L/min'
        Unit of the flow signal. Accepted values include 'L/min', 'L/s',
        'mL/min', 'mL/s', and supported spelling/case variants handled by
        ``convert_flow_unit``.
    ptp_window : float, default 0.20
        Window (seconds) before inspiration onset to compute baseline pressure
        for PTP calculation. If the window has no samples, the pressure at
        inspiration onset is used.
    block : int or str or None, default None
        Optional block identifier to prepend as a ``block`` column.
    block_name : str or None, default None
        Optional block name to prepend as a ``block_name`` column.

    Returns
    -------
    pandas.DataFrame
        One row per cycle with columns. If block information is available,
        a leading ``block`` column is included.

    Raises
    ------
    KeyError
        If ``df_block`` is missing a ``time_block`` column, or if
        ``cycles_df`` is missing ``t_inspi``, ``t_expi``, or
        ``t_next_inspi``.
    """
    # Guard clauses
    include_block = block is not None or (
        cycles_df is not None and "block" in cycles_df.columns
    )
    include_block_name = block_name is not None or (
        cycles_df is not None and "block_name" in cycles_df.columns
    )
    base_columns = [
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
        "WOB",
        "PTP",
    ]
    if include_block:
        base_columns = ["block"] + base_columns
    if include_block_name:
        base_columns = ["block_name"] + base_columns

    if df_block is None or df_block.empty:
        return pd.DataFrame(columns=base_columns)
    if cycles_df is None or cycles_df.empty:
        return pd.DataFrame(columns=base_columns)

    # Required time axis
    if "time_block" not in df_block.columns:
        raise KeyError("df_block must contain a 'time_block' column")

    t = df_block["time_block"].to_numpy()
    # Optional channels
    has_flow = flow_col in df_block.columns
    has_vol = (volume_col is not None) and (volume_col in df_block.columns)
    has_pressure = (pressure_col is not None) and (pressure_col in df_block.columns)
    has_pes = (pes_col is not None) and (pes_col in df_block.columns)

    flow = df_block[flow_col].to_numpy() if has_flow else None
    pressure = df_block[pressure_col].to_numpy() if has_pressure else None
    pes = df_block[pes_col].to_numpy() if has_pes else None
    if flow is not None:
        # Convert flow to L/s if needed (spontaneous convention: inspiration negative)
        flow = convert_flow_unit(flow, flow_unit)
    vol = df_block[volume_col].to_numpy() if has_vol else None

    # Use t_inspi for inspiration time
    ti_col = "t_inspi"
    if ti_col not in cycles_df.columns:
        raise KeyError("cycles_df must contain a 't_inspi' column")
    te_col = "t_expi"
    if te_col not in cycles_df.columns:
        raise KeyError("cycles_df must contain a 't_expi' column")
    if "t_next_inspi" not in cycles_df.columns:
        raise KeyError("cycles_df must contain a 't_next_inspi' column")

    base_cols = [ti_col, te_col, "t_next_inspi"]
    use_cols = base_cols + (["n_cycle"] if "n_cycle" in cycles_df.columns else [])
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
    for _, row in cyc.iterrows():
        ti = float(row[ti_col])
        te = float(row[te_col])
        if te <= ti:
            # Invalid cycle ordering
            continue
        t_next = row["t_next_inspi"]

        i_insp = nearest_idx(t, ti)
        i_expi = nearest_idx(t, te)
        if i_expi <= i_insp:
            # Invalid or zero-length inspiration window
            continue

        # Durations
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

        # VT (ΔVolume if available, else integrate Flow)
        if has_vol:
            vt = float(vol[i_expi] - vol[i_insp])
            if vt < 0 and has_flow:
                # VolumeResp likely decreases during inspiration because it is the
                # cumulative integral of negative-convention flow. Fall back to flow
                # integration, which correctly negates the sign.
                warnings.warn(
                    f"VT from volume column is negative ({vt:.4f} L) at cycle "
                    f"{int(row['n_cycle'])}. VolumeResp likely decreases during "
                    "inspiration (e.g. it is the integral of negative-convention flow). "
                    "Falling back to flow integration. Pass volume_col=None to always "
                    "use flow integration and silence this warning.",
                    UserWarning,
                    stacklevel=3,
                )
                vt = -trapz_safe(flow[i_insp : i_expi + 1], t[i_insp : i_expi + 1])
        elif has_flow:
            # integrate flow between i_insp and i_expi (inclusive)
            i0, i1 = sorted((i_insp, i_expi))
            # flow is negative during inspiration -> negate to return positive VT
            vt = -trapz_safe(flow[i0 : i1 + 1], t[i0 : i1 + 1])
        else:
            vt = float("nan")

        ve = bf * vt if (np.isfinite(bf) and np.isfinite(vt)) else float("nan")

        # Peaks (magnitudes): inspiration negative -> use abs(min) for PIF; expiration positive -> max
        if has_flow:
            i0, i1 = sorted((i_insp, i_expi))
            seg_insp = flow[i0 : i1 + 1]
            pif = float(abs(np.nanmin(seg_insp))) if seg_insp.size else float("nan")

            if i_next is not None and i_next > i_expi:
                seg_exp = flow[i_expi : i_next + 1]
                pef = float(np.nanmax(seg_exp)) if seg_exp.size else float("nan")
            else:
                pef = float("nan")
        else:
            pif = float("nan")
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

        # WOB calculation (requires esophageal pressure for physiological accuracy)
        # Convention: Pes in cmH2O, standard subatmospheric values (e.g. -5 to -20 at rest).
        # Pmus = Pes_baseline - Pes > 0 when the patient generates inspiratory effort.
        # WOB = ∫ Pmus × (-Flow) dt; -Flow > 0 during inspiration (negative-flow convention).
        # Units: kPa × L/s × s = kPa·L = J  (1 kPa·L = 1 J).
        if has_pes and has_flow:
            i0, i1 = sorted((i_insp, i_expi))
            # Pes baseline: median in the pre-inspiratory window (same window as PTP)
            t0_pes = max(t[0], ti - ptp_window)
            m_pes = (t >= t0_pes) & (t < ti)
            if np.any(m_pes):
                pes_baseline = float(np.nanmedian(pes[m_pes]))
            else:
                pes_baseline = float(pes[i_insp])
            pmus_kpa = (pes_baseline - pes[i0 : i1 + 1]) * 0.0980665
            wob = trapz_safe(pmus_kpa * (-flow[i0 : i1 + 1]), t[i0 : i1 + 1])
        else:
            wob = float("nan")

        # PTP calculation (cmH2O·s) relative to baseline pressure
        if has_pressure:
            i0, i1 = sorted((i_insp, i_expi))
            # Compute baseline as median pressure in window before inspiration
            t0_baseline = max(t[0], ti - ptp_window)
            m_baseline = (t >= t0_baseline) & (t < ti)
            if np.any(m_baseline):
                p_baseline = float(np.nanmedian(pressure[m_baseline]))
            else:
                # Fall back to pressure at inspiration onset to avoid arbitrary baseline
                p_baseline = float(pressure[i_insp])
            # PTP = ∫ (P_baseline - P) dt during inspiration.
            # Positive when airway pressure falls below baseline (spontaneous effort).
            p_drop = p_baseline - pressure[i0 : i1 + 1]
            ptp = trapz_safe(p_drop, t[i0 : i1 + 1])
        else:
            ptp = float("nan")

        rows.append(
            {
                "n_cycle": int(row["n_cycle"]),
                "t_inspi": t[i_insp],
                "t_expi": t[i_expi],
                "Ti": ti_duration,
                "Ttot": ttot,
                "Te": te_duration,
                "BF": bf,
                "VT": vt,
                "VE": ve,
                "PIF": pif,
                "PEF": pef,
                "IE": ie_ratio,
                "WOB": wob,
                "PTP": ptp,
            }
        )

    if not rows:
        return pd.DataFrame(columns=base_columns)

    out = pd.DataFrame(rows)
    if block_value is not None:
        out.insert(0, "block", block_value)
    if block_name_value is not None:
        out.insert(0, "block_name", block_name_value)
    return out
