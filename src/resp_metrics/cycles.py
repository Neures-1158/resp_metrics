"""
Cycle detection utilities for respiratory metrics.

This module defines helper functions for detecting respiratory cycles
from comment annotations. A cycle is defined as an inspiration start
followed by expiration start and the next inspiration start. This relies
on comment labels exported from LabChart (e.g. ``INSPI`` for inspiration,
``EXPI`` for expiration) and the time stamps associated with those comments.
"""

from __future__ import annotations
import pandas as pd


def cycles_from_comments(
    comments_df: pd.DataFrame,
    block: int,
    insp_label: str = "INSPI",
    expi_label: str = "EXPI",
) -> pd.DataFrame:
    """
    Build a cycles DataFrame by identifying complete respiratory cycles 
    (INSPI → EXPI → next INSPI) within a block.

    Parameters
    ----------
    comments_df : pandas.DataFrame
        DataFrame of comments with at least columns ``time_block``, ``block``,
        and ``Comment``.
    block : int
        Block index to extract cycles from.
    insp_label : str, default="INSPI"
        Label text used for inspiration onset.
    expi_label : str, default="EXPI"
        Label text used for expiration onset.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
          - ``n_cycle``: 1-based cycle index within the block
          - ``t_inspi``: absolute time of inspiration onset
          - ``t_expi``: absolute time of expiration onset
          - ``t_next_inspi``: absolute time of next inspiration onset
    """
    if comments_df is None or comments_df.empty:
        return pd.DataFrame(columns=["n_cycle", "t_inspi", "t_expi", "t_next_inspi"])

    # filter
    c = comments_df.loc[comments_df["block"] == block].copy()
    if c.empty:
        return pd.DataFrame(columns=["n_cycle", "t_inspi", "t_expi", "t_next_inspi"])
    lab = c["Comment"].astype(str).str.strip().str.upper()
    t = c["time_block"].to_numpy()

    # identify INSPI and EXPI times
    t_inspi = t[lab == insp_label.upper()]
    t_expi = t[lab == expi_label.upper()]

    rows = []
    for i, ti in enumerate(t_inspi[:-1]):  # Skip last INSPI as it won't have a next INSPI
        ex_after = t_expi[t_expi > ti]
        if ex_after.size:
            next_inspi = t_inspi[i + 1]  # Get next INSPI
            rows.append({
                "t_inspi": float(ti),
                "t_expi": float(ex_after[0]),
                "t_next_inspi": float(next_inspi)
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "n_cycle", range(1, len(out) + 1))
    return out