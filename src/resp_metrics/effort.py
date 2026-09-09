"""
Respiratory effort indices computed cycle-by-cycle from oesophageal, gastric
and transdiaphragmatic pressures.

This module provides a single high-level function:

    effort_from_cycles(df_block, cycles_df, pes_col="Pes", pga_col="Pga")

It returns a DataFrame with one row per cycle and the following columns:
  - n_cycle: 1-based cycle index within the block
  - t_inspi, t_expi: absolute times (s) delimiting inspiration (from comments)
  - dPes: inspiratory oesophageal pressure swing (cmH2O)
  - dPga: inspiratory gastric pressure swing (cmH2O)
  - dPga_corr: same, referenced to the Pga nadir (cmH2O)
  - dPdi: inspiratory transdiaphragmatic pressure swing (cmH2O)
  - WOB: work of breathing (J) — requires Pes and flow
  - PTPes: oesophageal pressure-time product (cmH2O·s per breath)
  - PTPga: gastric pressure-time product (cmH2O·s per breath)
  - PTPga_corr: same, referenced to the Pga nadir (cmH2O·s per breath)
  - PTPdi: transdiaphragmatic pressure-time product (cmH2O·s per breath)
  - PTPdi_PTPes: ratio of the two pressure-time products (dimensionless)
  - TTIdi: tension-time index of the diaphragm (dimensionless)

Definitions follow the reference statements on respiratory muscle testing:

  - ATS/ERS Statement on Respiratory Muscle Testing.
    Am J Respir Crit Care Med 2002;165:518-624.
    PTP is the integration of respiratory pressure over time. The tension-time
    index of the diaphragm is TTdi = (Pdi/Pdi,max) x (TI/Ttot), "where Pdi is
    the mean transdiaphragmatic pressure generated per breath". The diaphragm
    fatigue threshold is a TTdi of 0.15-0.18.
  - Laveneziana P, et al. ERS statement on respiratory muscle testing at rest
    and during exercise. Eur Respir J 2019;53:1801214.
    PTPdi is "the time integral of the area between baseline Pdi (resting
    end-expiratory Pdi) and Pdi during the inspiratory effort"; TTdi is "the
    product of mean inspiratory Pdi divided by Pdi,max and TI/TTOT".

Assumptions:
  - df_block contains at least 'time_block' and the requested pressure columns
  - Pressures are in cmH2O; no unit conversion is applied
  - Flow, used only for WOB, is negative during inspiration (spontaneous
    breathing convention) and converted to L/s from `flow_unit`
  - Inspiration is delimited by the INSPI/EXPI comments, i.e. the window
    [t_inspi, t_expi] of cycles_df. No automatic effort-onset detection.
  - cycles_df contains 't_inspi', 't_expi' and 't_next_inspi'

Notes:
  - Each channel gets its own baseline: the median value over
    [t_inspi - baseline_window, t_inspi), i.e. the resting end-expiratory
    level. When that window holds no sample, the value at onset is used.
  - Swings are reported as positive magnitudes for a normal inspiratory
    effort: dPes = Pes_baseline - min(Pes) (Pes falls), while dPga and dPdi
    are max(P) - P_baseline (Pga and Pdi rise). dPga may come out negative
    when gastric pressure falls during inspiration, which is informative
    (abdominal paradox), and is therefore not clipped.
  - WOB = integral of (Pes_baseline - Pes) x (-Flow) dt over inspiration, with
    the pressure converted from cmH2O to kPa so that kPa.L = J. It is positive
    when the subject generates an inspiratory effort. Airway pressure must not
    be substituted for Pes: it would not represent patient effort.
  - dPga_corr and PTPga_corr use the Pga nadir within the first
    `pga_nadir_frac` of inspiration as reference, instead of the
    end-expiratory baseline, and integrate from that instant. When expiratory
    abdominal muscles are recruited, Pga is still elevated at the INSPI marker
    and falls as those muscles relax at the start of inspiration; the
    end-expiratory baseline then sits above the relaxed level and
    underestimates the gastric swing. ATS/ERS 2002 flags this artifact
    ("recruitment of abdominal muscles during expiration is followed by a
    sudden relaxation at the beginning of the next inspiratory effort"), and
    ERS 2019 notes that recording Pga is what allows the beginning of
    inhalation to be defined accurately. Without such recruitment Pga is
    already at its nadir at the marker and the corrected values match the
    uncorrected ones. Pes and Pdi are deliberately left uncorrected: their
    swings are an order of magnitude larger, so the same offset is
    negligible, and Pdi is derived from Pga and Pes and cannot take an
    independent reference.
  - PTPes is NOT corrected for chest wall elastic recoil, which would require
    chest wall elastance. It is a practical within-subject index of global
    inspiratory effort, not an absolute measure.
  - Pdi is read from `pdi_col` when available, otherwise derived as Pga - Pes.
  - TTIdi = (mean inspiratory Pdi / Pdi_max) x (Ti / Ttot), which is
    equivalent to PTPdi / (Pdi_max x Ttot). Pdi_max cannot be derived from
    tidal breathing: it must be supplied via `pdi_max` from a maximal
    manoeuvre, otherwise TTIdi is NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import _baseline_before, convert_flow_unit, nearest_idx, trapz_safe

# cmH2O -> kPa, so that pressure x volume comes out in joules
_CMH2O_TO_KPA = 0.0980665

__all__ = ["effort_from_cycles"]


def effort_from_cycles(
    df_block: pd.DataFrame,
    cycles_df: pd.DataFrame,
    pes_col: str | None = "Pes",
    pga_col: str | None = "Pga",
    pdi_col: str | None = None,
    flow_col: str = "Flow",
    *,
    flow_unit: str = "L/min",
    baseline_window: float = 0.20,
    pga_nadir_frac: float = 1.0 / 3.0,
    pdi_max: float | None = None,
    block: int | str | None = None,
    block_name: str | None = None,
) -> pd.DataFrame:
    """Compute respiratory effort indices per cycle.

    Parameters
    ----------
    df_block : pandas.DataFrame
        A single-block DataFrame as returned by LabChartFile.get_block_df(b).
        Must contain 'time_block' and the requested pressure columns.
    cycles_df : pandas.DataFrame
        Output of cycles_from_comments. Must contain 't_inspi', 't_expi' and
        't_next_inspi'.
    pes_col : str or None, default 'Pes'
        Column name for oesophageal pressure (cmH2O). If None or missing,
        dPes and PTPes are NaN.
    pga_col : str or None, default 'Pga'
        Column name for gastric pressure (cmH2O). If None or missing, dPga
        and PTPga are NaN.
    pdi_col : str or None, default None
        Column name for a recorded transdiaphragmatic pressure (cmH2O). When
        None or missing, Pdi is derived as ``Pga - Pes`` if both channels are
        available; otherwise the Pdi metrics are NaN.
    flow_col : str, default 'Flow'
        Column name for the flow signal, used only for WOB. If missing, WOB
        is NaN.
    flow_unit : str, default 'L/min'
        Unit of the flow signal, handled by ``convert_flow_unit``.
    baseline_window : float, default 0.20
        Window (seconds) before inspiration onset used to compute the resting
        end-expiratory baseline of each pressure channel.
    pga_nadir_frac : float, default 1/3
        Fraction of inspiration searched for the Pga nadir used by
        ``dPga_corr`` and ``PTPga_corr``.
    pdi_max : float or None, default None
        Maximal transdiaphragmatic pressure (cmH2O) from a maximal manoeuvre,
        used to normalise TTIdi. If None or non-positive, TTIdi is NaN.
    block : int or str or None, default None
        Optional block identifier to prepend as a ``block`` column.
    block_name : str or None, default None
        Optional block name to prepend as a ``block_name`` column.

    Returns
    -------
    pandas.DataFrame
        One row per cycle. When block information is available, optional
        leading columns are prepended in this order: ``block_name``,
        ``block``, followed by the per-cycle metric columns.

    Raises
    ------
    KeyError
        If ``df_block`` is missing a ``time_block`` column, or if
        ``cycles_df`` is missing ``t_inspi``, ``t_expi``, or
        ``t_next_inspi``.
    """
    all_columns = [
        "n_cycle",
        "t_inspi",
        "t_expi",
        "dPes",
        "dPga",
        "dPga_corr",
        "dPdi",
        "WOB",
        "PTPes",
        "PTPga",
        "PTPga_corr",
        "PTPdi",
        "PTPdi_PTPes",
        "TTIdi",
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

    if df_block is None or df_block.empty:
        return pd.DataFrame(columns=all_columns)
    if cycles_df is None or cycles_df.empty:
        return pd.DataFrame(columns=all_columns)

    if "time_block" not in df_block.columns:
        raise KeyError("df_block must contain a 'time_block' column")

    ti_col = "t_inspi"
    if ti_col not in cycles_df.columns:
        raise KeyError("cycles_df must contain a 't_inspi' column")
    te_col = "t_expi"
    if te_col not in cycles_df.columns:
        raise KeyError("cycles_df must contain a 't_expi' column")
    if "t_next_inspi" not in cycles_df.columns:
        raise KeyError("cycles_df must contain a 't_next_inspi' column")

    t = df_block["time_block"].to_numpy()

    has_pes = (pes_col is not None) and (pes_col in df_block.columns)
    has_pga = (pga_col is not None) and (pga_col in df_block.columns)
    pes = df_block[pes_col].to_numpy(dtype=float) if has_pes else None
    pga = df_block[pga_col].to_numpy(dtype=float) if has_pga else None

    has_flow = flow_col in df_block.columns
    flow = (
        convert_flow_unit(df_block[flow_col].to_numpy(), flow_unit)
        if has_flow
        else None
    )

    # Pdi: recorded channel when available, otherwise derived as Pga - Pes.
    if (pdi_col is not None) and (pdi_col in df_block.columns):
        pdi = df_block[pdi_col].to_numpy(dtype=float)
    elif has_pes and has_pga:
        pdi = pga - pes
    else:
        pdi = None

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
        i0, i1 = sorted((i_insp, i_expi))

        # Total cycle duration, needed for TTIdi. Ti cancels out of the TTIdi
        # expression below, so it is not recomputed here; ventilatory_from_cycles
        # already reports Ti, Ttot and Ti/Ttot.
        if pd.notna(t_next):
            i_next = nearest_idx(t, float(t_next))
            ttot = float(t[i_next] - t[i_insp]) if i_next > i_expi else float("nan")
        else:
            ttot = float("nan")

        seg_t = t[i0 : i1 + 1]

        # --- Oesophageal pressure: falls during inspiratory effort ---
        if has_pes:
            pes_base = _baseline_before(t, pes, ti, baseline_window, i_insp)
            seg_pes = pes[i0 : i1 + 1]
            d_pes = pes_base - float(np.nanmin(seg_pes))
            ptp_pes = trapz_safe(pes_base - seg_pes, seg_t)
            # Work of breathing: Pmus x inspired volume, in joules.
            if has_flow:
                pmus_kpa = (pes_base - seg_pes) * _CMH2O_TO_KPA
                wob = trapz_safe(pmus_kpa * (-flow[i0 : i1 + 1]), seg_t)
            else:
                wob = float("nan")
        else:
            d_pes = float("nan")
            ptp_pes = float("nan")
            wob = float("nan")

        # --- Gastric pressure: rises during inspiratory effort ---
        if has_pga:
            pga_base = _baseline_before(t, pga, ti, baseline_window, i_insp)
            seg_pga = pga[i0 : i1 + 1]
            d_pga = float(np.nanmax(seg_pga)) - pga_base
            ptp_pga = trapz_safe(seg_pga - pga_base, seg_t)
            # Corrected variant: reference the swing to the Pga nadir, i.e. the
            # relaxed abdominal level once expiratory muscles have let go. The
            # search stays inside inspiration whatever pga_nadir_frac is, so the
            # nadir can never land in expiration and leave an empty segment.
            i_end = min(i1, i0 + max(1, int((i1 - i0) * pga_nadir_frac)))
            search = pga[i0 : i_end + 1]
            if np.all(np.isnan(search)):
                d_pga_corr = float("nan")
                ptp_pga_corr = float("nan")
            else:
                i_nadir = i0 + int(np.nanargmin(search))
                seg_corr = pga[i_nadir : i1 + 1]
                d_pga_corr = float(np.nanmax(seg_corr)) - float(pga[i_nadir])
                ptp_pga_corr = trapz_safe(seg_corr - pga[i_nadir], t[i_nadir : i1 + 1])
        else:
            d_pga = float("nan")
            ptp_pga = float("nan")
            d_pga_corr = float("nan")
            ptp_pga_corr = float("nan")

        # --- Transdiaphragmatic pressure: rises during inspiratory effort ---
        if pdi is not None:
            pdi_base = _baseline_before(t, pdi, ti, baseline_window, i_insp)
            seg_pdi = pdi[i0 : i1 + 1]
            d_pdi = float(np.nanmax(seg_pdi)) - pdi_base
            ptp_pdi = trapz_safe(seg_pdi - pdi_base, seg_t)
        else:
            d_pdi = float("nan")
            ptp_pdi = float("nan")

        # --- Diaphragmatic share of the inspiratory effort ---
        ptp_ratio = (
            (ptp_pdi / ptp_pes)
            if (np.isfinite(ptp_pdi) and np.isfinite(ptp_pes) and ptp_pes > 0)
            else float("nan")
        )

        # --- Tension-time index of the diaphragm ---
        # TTIdi = (mean inspiratory Pdi / Pdi_max) x (Ti / Ttot)
        #       = (PTPdi / Ti) / Pdi_max x (Ti / Ttot)
        #       = PTPdi / (Pdi_max x Ttot)
        tti_di = (
            (ptp_pdi / (pdi_max * ttot))
            if (
                pdi_max is not None
                and pdi_max > 0
                and np.isfinite(ptp_pdi)
                and np.isfinite(ttot)
                and ttot > 0
            )
            else float("nan")
        )

        rows.append(
            {
                "n_cycle": int(row["n_cycle"]),
                "t_inspi": t[i_insp],
                "t_expi": t[i_expi],
                "dPes": d_pes,
                "dPga": d_pga,
                "dPga_corr": d_pga_corr,
                "dPdi": d_pdi,
                "WOB": wob,
                "PTPes": ptp_pes,
                "PTPga": ptp_pga,
                "PTPga_corr": ptp_pga_corr,
                "PTPdi": ptp_pdi,
                "PTPdi_PTPes": ptp_ratio,
                "TTIdi": tti_di,
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
