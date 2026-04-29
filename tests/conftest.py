"""
Pytest fixtures for resp_metrics tests.

This module provides synthetic signals with known, manually calculable values
for validating the scientific accuracy of respiratory metrics calculations.
"""

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# COMMENT FIXTURES
# =============================================================================


@pytest.fixture
def empty_comments_df():
    """Empty comments DataFrame."""
    return pd.DataFrame(columns=["block", "time_block", "Comment"])


@pytest.fixture
def sample_comments_df():
    """DataFrame with 3 complete respiratory cycles in block 1.

    Cycles:
    - Cycle 1: INSPI at 0.0s, EXPI at 1.0s, next INSPI at 2.0s
    - Cycle 2: INSPI at 2.0s, EXPI at 3.0s, next INSPI at 4.0s
    - Cycle 3: INSPI at 4.0s, EXPI at 5.0s (incomplete - no next INSPI)
    """
    return pd.DataFrame(
        {
            "block": [1, 1, 1, 1, 1, 1],
            "time_block": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Comment": ["INSPI", "EXPI", "INSPI", "EXPI", "INSPI", "EXPI"],
        }
    )


@pytest.fixture
def comments_df_block2():
    """Comments in block 2 for testing block filtering."""
    return pd.DataFrame(
        {
            "block": [2, 2, 2, 2],
            "time_block": [10.0, 11.0, 12.0, 13.0],
            "Comment": ["INSPI", "EXPI", "INSPI", "EXPI"],
        }
    )


@pytest.fixture
def comments_df_lowercase():
    """Comments with lowercase labels to test case insensitivity."""
    return pd.DataFrame(
        {
            "block": [1, 1, 1, 1],
            "time_block": [0.0, 1.0, 2.0, 3.0],
            "Comment": ["inspi", "expi", "inspi", "expi"],
        }
    )


@pytest.fixture
def comments_df_missing_expi():
    """Comments where one INSPI has no following EXPI."""
    return pd.DataFrame(
        {
            "block": [1, 1, 1, 1],
            "time_block": [0.0, 1.0, 2.0, 4.0],
            "Comment": ["INSPI", "EXPI", "INSPI", "INSPI"],  # No EXPI after 2nd INSPI
        }
    )


# =============================================================================
# CYCLES FIXTURES
# =============================================================================


@pytest.fixture
def sample_cycles_df():
    """Pre-computed cycles DataFrame with 2 complete cycles.

    Cycle 1: Ti=1.0s, Te=1.0s, Ttot=2.0s, BF=30/min
    Cycle 2: Ti=1.0s, Te=1.0s, Ttot=2.0s, BF=30/min
    """
    return pd.DataFrame(
        {
            "n_cycle": [1, 2],
            "t_inspi": [0.0, 2.0],
            "t_expi": [1.0, 3.0],
            "t_next_inspi": [2.0, 4.0],
        }
    )


@pytest.fixture
def single_cycle_df():
    """Single cycle for edge case testing.

    Cycle 1: Ti=1.5s, Te=2.5s, Ttot=4.0s, BF=15/min, I:E=0.6
    """
    return pd.DataFrame(
        {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.5], "t_next_inspi": [4.0]}
    )


# =============================================================================
# SPONTANEOUS BREATHING SIGNAL FIXTURES
# =============================================================================


@pytest.fixture
def spontaneous_signal_df():
    """Synthetic spontaneous breathing signal with KNOWN values.

    Convention: Flow is NEGATIVE during inspiration.

    Parameters used to create signal:
    - Sample rate: 100 Hz
    - Duration: 10 s
    - Breathing frequency: ~15 breaths/min (Ttot = 4.0s)
    - Ti = 1.5s, Te = 2.5s
    - Flow: triangular wave, peak = -0.5 L/s during inspiration
    - VT = 0.375 L (area of triangle: 0.5 * 1.5 * 0.5)

    Known expected values for cycle at t=0:
    - Ti = 1.5 s
    - Te = 2.5 s
    - Ttot = 4.0 s
    - BF = 15 breaths/min
    - I:E = 0.6
    - VT ≈ 0.375 L (triangular inspiration)
    - VE = 15 * 0.375 = 5.625 L/min
    - PIF = 0.5 L/s
    - PEF = 0.3 L/s
    """
    fs = 100  # Hz
    duration = 10.0  # seconds
    t = np.arange(0, duration, 1 / fs)
    n_samples = len(t)

    # Create triangular flow pattern
    # Inspiration: 0 to 1.5s, flow goes from 0 to -0.5 to 0
    # Expiration: 1.5 to 4.0s, flow goes from 0 to +0.3 to 0
    cycle_period = 4.0  # Ttot
    ti_duration = 1.5

    flow = np.zeros(n_samples)
    for i, ti in enumerate(t):
        t_in_cycle = ti % cycle_period
        if t_in_cycle < ti_duration:
            # Inspiration (negative flow) - triangular
            if t_in_cycle < ti_duration / 2:
                flow[i] = -0.5 * (t_in_cycle / (ti_duration / 2))  # ramp down to -0.5
            else:
                flow[i] = -0.5 * (
                    1 - (t_in_cycle - ti_duration / 2) / (ti_duration / 2)
                )  # ramp up to 0
        else:
            # Expiration (positive flow) - triangular
            te_duration = cycle_period - ti_duration
            t_exp = t_in_cycle - ti_duration
            if t_exp < te_duration / 2:
                flow[i] = 0.3 * (t_exp / (te_duration / 2))  # ramp up to 0.3
            else:
                flow[i] = 0.3 * (
                    1 - (t_exp - te_duration / 2) / (te_duration / 2)
                )  # ramp down to 0

    # Integrate flow to get volume (cumulative)
    volume = np.cumsum(flow) / fs

    # Pressure: simple constant with small variation
    pressure = 2.0 + 0.5 * np.sin(2 * np.pi * t / cycle_period)

    # Esophageal pressure for WOB calculation (more negative during inspiration)
    pes = -5.0 + 3.0 * flow  # More negative when flow is negative (inspiration)

    return pd.DataFrame(
        {
            "time_block": t,
            "Flow": flow * 60,  # Convert to L/min for storage
            "VolumeResp": volume,
            "Paw": pressure,
            "Pes": pes,
        }
    )


@pytest.fixture
def spontaneous_cycles_for_signal():
    """Cycles matching the spontaneous_signal_df fixture."""
    return pd.DataFrame(
        {
            "n_cycle": [1, 2],
            "t_inspi": [0.0, 4.0],
            "t_expi": [1.5, 5.5],
            "t_next_inspi": [4.0, 8.0],
        }
    )


# =============================================================================
# MECHANICAL VENTILATION SIGNAL FIXTURES
# =============================================================================


@pytest.fixture
def mechanical_signal_df():
    """Synthetic mechanical ventilation signal with KNOWN values.

    Convention: Flow is POSITIVE during inspiration.

    Parameters:
    - Sample rate: 100 Hz
    - Duration: 10 s
    - Breathing frequency: ~24 breaths/min (Ttot = 2.5s)
    - Ti = 0.8s, Te = 1.7s
    - Active flow phase: 0.6s at 0.6 L/s; plateau phase: 0.2s at 0 L/s
    - VT = 0.6 * 0.6 = 0.36 L  (flow active for 0.6 s only, not the full 0.8 s Ti)
    - PEEP = 5 cmH2O
    - Ppeak = 25 cmH2O
    - Pplat = 20 cmH2O (with 0.2s plateau at end of inspiration)

    Known expected values:
    - Ti = 0.8 s
    - Te = 1.7 s
    - Ttot = 2.5 s
    - BF = 24 breaths/min
    - I:E ≈ 0.47
    - VT = 0.36 L  (0.6 L/s × 0.6 s active phase)
    - VE = 24 * 0.36 = 8.64 L/min
    - PIF = 0.6 L/s
    - PEEP = 5 cmH2O
    - Ppeak = 25 cmH2O
    - dP = 20 - 5 = 15 cmH2O (using Pplat)
    - Cstat = 0.36 / 15 = 0.024 L/cmH2O
    """
    fs = 100  # Hz
    duration = 10.0
    t = np.arange(0, duration, 1 / fs)
    n_samples = len(t)

    cycle_period = 2.5  # Ttot
    ti_duration = 0.8
    flow_phase_duration = 0.6

    flow = np.zeros(n_samples)
    pressure = np.zeros(n_samples)

    for i, ti in enumerate(t):
        t_in_cycle = ti % cycle_period
        if t_in_cycle < flow_phase_duration:
            # Active inspiration with flow
            flow[i] = 0.6  # L/s positive
            # Pressure ramps up during flow
            pressure[i] = 5 + 20 * (t_in_cycle / flow_phase_duration)
        elif t_in_cycle < ti_duration:
            # Inspiratory plateau (no flow, constant pressure)
            flow[i] = 0.0  # Low flow for plateau detection
            pressure[i] = 20  # Pplat
        else:
            # Expiration
            te_duration = cycle_period - ti_duration
            t_exp = t_in_cycle - ti_duration
            # Exponential decay of flow
            flow[i] = -0.4 * np.exp(-3 * t_exp / te_duration)
            # Pressure returns to PEEP
            pressure[i] = 5 + 15 * np.exp(-5 * t_exp / te_duration)

    # Integrate flow to get volume
    volume = np.cumsum(np.clip(flow, 0, None)) / fs  # Only positive flow for volume

    return pd.DataFrame(
        {
            "time_block": t,
            "Flow": flow * 60,  # Convert to L/min
            "VolumeResp": volume,
            "Pressure": pressure,
        }
    )


@pytest.fixture
def mechanical_cycles_for_signal():
    """Cycles matching the mechanical_signal_df fixture."""
    return pd.DataFrame(
        {
            "n_cycle": [1, 2, 3],
            "t_inspi": [0.0, 2.5, 5.0],
            "t_expi": [0.8, 3.3, 5.8],
            "t_next_inspi": [2.5, 5.0, 7.5],
        }
    )


# =============================================================================
# EDGE CASE FIXTURES
# =============================================================================


@pytest.fixture
def empty_signal_df():
    """Empty signal DataFrame."""
    return pd.DataFrame(columns=["time_block", "Flow", "VolumeResp", "Paw"])


@pytest.fixture
def signal_without_flow():
    """Signal DataFrame missing flow column."""
    t = np.linspace(0, 5, 100)
    return pd.DataFrame(
        {"time_block": t, "VolumeResp": np.zeros(100), "Paw": np.ones(100) * 5}
    )


@pytest.fixture
def signal_without_volume():
    """Signal DataFrame missing volume column."""
    t = np.linspace(0, 5, 100)
    flow = np.sin(2 * np.pi * t / 2)
    return pd.DataFrame(
        {"time_block": t, "Flow": flow * 60, "Paw": np.ones(100) * 5}  # L/min
    )


# =============================================================================
# KNOWN VALUE CONSTANTS FOR ASSERTIONS
# =============================================================================


class ExpectedSpontaneous:
    """Expected values for spontaneous_signal_df fixture."""

    Ti = 1.5  # s
    Te = 2.5  # s
    Ttot = 4.0  # s
    BF = 15.0  # breaths/min
    IE = 0.6  # ratio
    VT = 0.375  # L (triangular: 0.5 * 1.5 * 0.5)
    VE = 5.625  # L/min
    PIF = 0.5  # L/s
    PEF = 0.3  # L/s


class ExpectedMechanical:
    """Expected values for mechanical_signal_df fixture."""

    Ti = 0.8  # s
    Te = 1.7  # s
    Ttot = 2.5  # s
    BF = 24.0  # breaths/min
    IE = 0.47  # ratio (approx)
    VT = 0.36  # L (0.6 * 0.6 = 0.36, only active flow phase)
    VE = 8.64  # L/min (approx)
    PIF = 0.6  # L/s
    PEEP = 5.0  # cmH2O
    Ppeak = 25.0  # cmH2O (approx)
    Pplat = 20.0  # cmH2O
    d_p = 15.0  # cmH2O
    Cstat = 0.024  # L/cmH2O (VT / dP)
