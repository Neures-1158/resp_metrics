"""
Utility functions for respiratory metrics computation.

This module provides common helper functions used across the package.
"""

from __future__ import annotations

import numpy as np

# np.trapezoid was introduced in NumPy 2.0; np.trapz was removed in NumPy 2.2+.
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz  # type: ignore[attr-defined]

__all__ = ["nearest_idx", "trapz_safe", "convert_flow_unit"]


def nearest_idx(vec: np.ndarray, target: float) -> int:
    """Return index of element in `vec` nearest to `target`.

    Parameters
    ----------
    vec : np.ndarray
        1D array of values to search.
    target : float
        Target value to find the nearest element to.

    Returns
    -------
    int
        Index of the element in `vec` closest to `target`.

    Raises
    ------
    ValueError
        If ``vec`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> nearest_idx(np.array([0.0, 1.0, 2.0, 3.0]), 1.7)
    2
    """
    if vec.size == 0:
        raise ValueError("vec must be non-empty")
    return int(np.abs(vec - target).argmin())


def trapz_safe(y: np.ndarray, x: np.ndarray) -> float:
    """Safe trapezoidal integral; returns NaN when not enough samples.

    Parameters
    ----------
    y : np.ndarray
        Array of y values (function values).
    x : np.ndarray
        Array of x values (independent variable).

    Returns
    -------
    float
        Trapezoidal integral of y over x, or NaN if insufficient data.

    Notes
    -----
    Uses ``numpy.trapezoid`` (NumPy ≥ 2.0) or ``numpy.trapz`` as a
    fallback. Returns NaN if either array has fewer than 2 elements.

    Examples
    --------
    >>> import numpy as np
    >>> trapz_safe(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]))
    4.0
    >>> trapz_safe(np.array([1.0]), np.array([0.0]))
    nan
    """
    if y.size < 2 or x.size < 2:
        return float("nan")
    return float(_trapz(y, x))


def convert_flow_unit(flow: np.ndarray, flow_unit: str) -> np.ndarray:
    """Convert flow array to L/s from the specified unit.

    Parameters
    ----------
    flow : np.ndarray
        Flow values in the original unit.
    flow_unit : str
        Unit of the input flow. Accepted values: 'L/min', 'lpm', 'L/s', 'l/sec', 'ls',
        'mL/s', 'ml/sec', 'mls', 'mL/min', 'mlpm'.

    Returns
    -------
    np.ndarray
        Flow values in L/s.

    Raises
    ------
    ValueError
        If flow_unit is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> convert_flow_unit(np.array([60.0, 120.0]), 'L/min')
    array([1., 2.])
    """
    flow = flow.astype(float)
    unit_lower = flow_unit.lower()

    if unit_lower in ["l/s", "l/sec", "ls"]:
        return flow
    elif unit_lower in ["l/min", "lpm"]:
        return flow / 60.0
    elif unit_lower in ["ml/s", "ml/sec", "mls"]:
        return flow / 1000.0
    elif unit_lower in ["ml/min", "mlpm"]:
        return flow / 60000.0
    else:
        raise ValueError(
            f"Unsupported flow unit: {flow_unit}. "
            "Use 'L/s', 'L/min', 'mL/s', or 'mL/min'."
        )


def _baseline_before(
    t: np.ndarray,
    sig: np.ndarray,
    t_onset: float,
    window: float,
    i_onset: int,
) -> float:
    """Median signal value over the window preceding inspiration onset.

    Used as the reference ("resting end-expiratory") level for pressure swings
    and pressure-time products.

    Parameters
    ----------
    t : np.ndarray
        Time axis (s), monotonically increasing.
    sig : np.ndarray
        Signal sampled on ``t``.
    t_onset : float
        Inspiration onset time (s).
    window : float
        Window duration (s) before ``t_onset`` used for the median.
    i_onset : int
        Index of ``t_onset`` in ``t``, used as a fallback when the window
        contains no sample.

    Returns
    -------
    float
        Median of ``sig`` over ``[t_onset - window, t_onset)``, or
        ``sig[i_onset]`` when that window is empty.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0.0, 0.1, 0.2, 0.3])
    >>> sig = np.array([1.0, 3.0, 5.0, 7.0])
    >>> _baseline_before(t, sig, 0.2, 0.2, 2)
    2.0
    """
    t0 = max(t[0], t_onset - window)
    mask = (t >= t0) & (t < t_onset)
    if np.any(mask):
        return float(np.nanmedian(sig[mask]))
    return float(sig[i_onset])
