"""
Utility functions for respiratory metrics computation.

This module provides common helper functions used across the package.
"""

from __future__ import annotations

import numpy as np


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
    
    Examples
    --------
    >>> import numpy as np
    >>> nearest_idx(np.array([0.0, 1.0, 2.0, 3.0]), 1.7)
    2
    """
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
    Uses numpy.trapz internally. Returns NaN if either array has fewer
    than 2 elements.
    
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
    return float(np.trapz(y, x))


def convert_flow_unit(flow: np.ndarray, flow_unit: str) -> np.ndarray:
    """Convert flow array to L/s from the specified unit.
    
    Parameters
    ----------
    flow : np.ndarray
        Flow values in the original unit.
    flow_unit : str
        Unit of the input flow. Accepted values: 'L/min', 'lpm', 'L/s', 'l/sec', 'ls'.
    
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
    
    if unit_lower in ['l/s', 'l/sec', 'ls']:
        return flow
    elif unit_lower in ['l/min', 'lpm']:
        return flow / 60.0
    else:
        raise ValueError(f"Unsupported flow unit: {flow_unit}. Use 'L/s' or 'L/min'.")
