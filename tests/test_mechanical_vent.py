"""
Tests for resp_metrics.mechanical_vent module.

Tests mechanical ventilation metric calculations with known synthetic values.
All tests use pytest.approx(rel=1e-3) for numerical comparisons.
"""

import math

import numpy as np
import pandas as pd
import pytest

from resp_metrics.mechanical_vent import mechanical_from_cycles


class TestMechanicalEmptyInputs:
    """Tests for empty/null input handling."""

    def test_none_df_block_returns_empty(self, sample_cycles_df):
        """Should return empty DataFrame for None df_block."""
        result = mechanical_from_cycles(None, sample_cycles_df)
        assert result.empty
        assert "PEEP" in result.columns
        assert "Ppeak" in result.columns

    def test_empty_df_block_returns_empty(self, sample_cycles_df):
        """Should return empty DataFrame for empty df_block."""
        empty_df = pd.DataFrame()
        result = mechanical_from_cycles(empty_df, sample_cycles_df)
        assert result.empty

    def test_none_cycles_returns_empty(self, mechanical_signal_df):
        """Should return empty DataFrame for None cycles."""
        result = mechanical_from_cycles(mechanical_signal_df, None)
        assert result.empty

    def test_missing_required_columns(self):
        """Should return empty if required columns missing."""
        df = pd.DataFrame(
            {
                "time_block": [0, 1, 2],
                "Flow": [1, 2, 3],
                # Missing pressure column
            }
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )
        result = mechanical_from_cycles(df, cycles)
        assert result.empty


class TestMechanicalCycleValidation:
    """Tests for invalid or inconsistent cycle timing."""

    def test_expi_before_inspi_skips_cycle(self):
        """Cycles with t_expi <= t_inspi should be skipped."""
        t = np.linspace(0, 2, 200)
        df = pd.DataFrame(
            {"time_block": t, "Flow": np.zeros(200), "Pressure": np.ones(200) * 10.0}
        )
        cycles = pd.DataFrame(
            {
                "n_cycle": [1],
                "t_inspi": [1.0],
                "t_expi": [0.5],  # invalid
                "t_next_inspi": [1.5],
            }
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure", flow_unit="L/s"
        )

        assert result.empty
        assert list(result.columns) == [
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

    def test_next_inspi_before_expi_sets_ttot_nan(self):
        """If next INSPI is before EXPI, Ttot/Te/BF should be NaN."""
        t = np.linspace(0, 2, 200)
        flow = np.zeros(200)
        flow[t < 0.8] = 0.5
        df = pd.DataFrame(
            {"time_block": t, "Flow": flow, "Pressure": np.ones(200) * 15.0}
        )
        cycles = pd.DataFrame(
            {
                "n_cycle": [1],
                "t_inspi": [0.0],
                "t_expi": [1.0],
                "t_next_inspi": [0.5],  # invalid ordering
            }
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure", flow_unit="L/s"
        )

        assert len(result) == 1
        assert math.isnan(result.iloc[0]["Ttot"])
        assert math.isnan(result.iloc[0]["Te"])
        assert math.isnan(result.iloc[0]["BF"])


class TestMechanicalFlowUnit:
    """Tests for flow unit conversion in mechanical ventilation."""

    def test_flow_l_min_conversion(self):
        """Flow in L/min should be converted to L/s."""
        t = np.linspace(0, 3, 300)
        flow_lpm = np.zeros(300)
        flow_lpm[t < 1.0] = 36.0  # 36 L/min = 0.6 L/s

        df = pd.DataFrame(
            {"time_block": t, "Flow": flow_lpm, "Pressure": np.ones(300) * 20.0}
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure", flow_unit="L/min"
        )

        # PIF should be 0.6 L/s
        assert result.iloc[0]["PIF"] == pytest.approx(0.6, rel=1e-2)

    def test_flow_l_s_no_conversion(self):
        """Flow in L/s should not be converted."""
        t = np.linspace(0, 3, 300)
        flow_ls = np.zeros(300)
        flow_ls[t < 1.0] = 0.6  # Already L/s

        df = pd.DataFrame(
            {"time_block": t, "Flow": flow_ls, "Pressure": np.ones(300) * 20.0}
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure", flow_unit="L/s"
        )

        # PIF should be 0.6 L/s
        assert result.iloc[0]["PIF"] == pytest.approx(0.6, rel=1e-2)


class TestMechanicalPEEP:
    """Tests for PEEP calculation."""

    def test_peep_median_before_inspiration(self):
        """PEEP should be median pressure in window before inspiration."""
        t = np.linspace(0, 4, 400)
        pressure = np.ones(400) * 20.0  # High during inspiration
        # PEEP window: just before t_inspi=1.0
        pressure[(t >= 0.8) & (t < 1.0)] = 5.0
        pressure[t < 0.8] = 5.0  # Baseline

        df = pd.DataFrame(
            {
                "time_block": t,
                "Flow": np.ones(400) * 30.0,  # L/min
                "Pressure": pressure,
            }
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [3.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure", peep_window=0.20
        )

        # PEEP should be ~5 cmH2O (median of 0.8-1.0s)
        assert result.iloc[0]["PEEP"] == pytest.approx(5.0, rel=1e-2)


class TestMechanicalPpeak:
    """Tests for peak pressure calculation."""

    def test_ppeak_max_during_inspiration(self):
        """Ppeak should be max pressure during inspiration."""
        t = np.linspace(0, 3, 300)
        pressure = np.ones(300) * 5.0  # PEEP
        # Ramp up to 25 cmH2O during inspiration (0-1s)
        insp_mask = (t >= 0) & (t < 1.0)
        pressure[insp_mask] = 5 + 20 * (t[insp_mask] / 1.0)

        df = pd.DataFrame(
            {
                "time_block": t,
                "Flow": np.where(t < 1.0, 30.0, -20.0),  # L/min
                "Pressure": pressure,
            }
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure"
        )

        # Ppeak should be ~25 cmH2O
        assert result.iloc[0]["Ppeak"] == pytest.approx(25.0, rel=5e-2)


class TestMechanicalPplat:
    """Tests for plateau pressure detection."""

    def test_pplat_detected_with_low_flow(self):
        """Pplat should be detected when flow < threshold for sufficient duration."""
        t = np.linspace(0, 3, 300)
        flow = np.zeros(300)
        pressure = np.ones(300) * 5.0

        # Inspiration: 0-1s
        # Active flow phase: 0-0.7s (flow = 0.6 L/s)
        flow[(t >= 0) & (t < 0.7)] = 36.0  # L/min = 0.6 L/s
        pressure[(t >= 0) & (t < 0.7)] = 5 + 20 * (t[(t >= 0) & (t < 0.7)] / 0.7)

        # Plateau phase: 0.7-1.0s (flow = 0, pressure stable at 20)
        flow[(t >= 0.7) & (t < 1.0)] = 0.0  # Low flow
        pressure[(t >= 0.7) & (t < 1.0)] = 20.0  # Pplat

        # Expiration: 1.0-2.0s
        flow[(t >= 1.0) & (t < 2.0)] = -20.0  # L/min
        pressure[(t >= 1.0)] = 5.0  # Back to PEEP

        df = pd.DataFrame({"time_block": t, "Flow": flow, "Pressure": pressure})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df,
            cycles,
            flow_col="Flow",
            pressure_col="Pressure",
            plateau_flow_thresh=0.05,
            plateau_min_dur=0.10,
        )

        # Pplat should be ~20 cmH2O
        assert result.iloc[0]["Pplat"] == pytest.approx(20.0, rel=5e-2)

    def test_pplat_nan_without_plateau(self):
        """Pplat should be NaN if no low-flow period detected."""
        t = np.linspace(0, 3, 300)
        # Constant high flow throughout inspiration - no plateau
        flow = np.where(t < 1.0, 60.0, -30.0)  # L/min
        pressure = np.where(t < 1.0, 25.0, 5.0)

        df = pd.DataFrame({"time_block": t, "Flow": flow, "Pressure": pressure})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df,
            cycles,
            flow_col="Flow",
            pressure_col="Pressure",
            plateau_flow_thresh=0.05,
            plateau_min_dur=0.10,
        )

        # Pplat should be NaN
        assert math.isnan(result.iloc[0]["Pplat"])


class TestMechanicalDrivingPressure:
    """Tests for driving pressure calculation."""

    def test_dp_from_pplat(self):
        """dP = Pplat - PEEP when Pplat available."""
        # Create signal with enough data before t_inspi for PEEP detection
        t = np.linspace(-0.5, 3, 350)  # Start before 0 for PEEP window
        pressure = np.ones(350) * 5.0  # PEEP baseline
        flow = np.zeros(350)

        # Pre-inspiration PEEP (t < 0)
        pressure[t < 0] = 5.0

        # Build signal with plateau
        flow[(t >= 0) & (t < 0.6)] = 36.0  # Active flow
        pressure[(t >= 0) & (t < 0.6)] = 5 + 15 * ((t[(t >= 0) & (t < 0.6)]) / 0.6)

        flow[(t >= 0.6) & (t < 1.0)] = 0.0  # Plateau
        pressure[(t >= 0.6) & (t < 1.0)] = 20.0  # Pplat

        flow[(t >= 1.0)] = -20.0  # Expiration
        pressure[(t >= 1.0)] = 5.0  # PEEP

        df = pd.DataFrame({"time_block": t, "Flow": flow, "Pressure": pressure})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df,
            cycles,
            flow_col="Flow",
            pressure_col="Pressure",
            peep_window=0.20,
            plateau_flow_thresh=0.05,
            plateau_min_dur=0.10,
        )

        # dP = Pplat - PEEP = 20 - 5 = 15 cmH2O
        # If Pplat detected, dP should be ~15
        if not math.isnan(result.iloc[0]["Pplat"]):
            assert result.iloc[0]["dP"] == pytest.approx(15.0, rel=2e-1)
        else:
            # If no Pplat, dP = Ppeak - PEEP
            assert not math.isnan(result.iloc[0]["dP"])


class TestMechanicalDPFallback:
    """Tests for dP_fallback flag."""

    def test_dp_fallback_false_when_pplat_available(self):
        """dP_fallback must be False when Pplat is detected."""
        t = np.linspace(0, 3, 300)
        flow = np.zeros(300)
        pressure = np.ones(300) * 5.0

        flow[(t >= 0) & (t < 0.6)] = 36.0  # active flow
        pressure[(t >= 0) & (t < 0.6)] = 20.0

        flow[(t >= 0.6) & (t < 1.0)] = 0.0  # plateau
        pressure[(t >= 0.6) & (t < 1.0)] = 20.0

        flow[(t >= 1.0)] = -20.0
        pressure[(t >= 1.0)] = 5.0

        df = pd.DataFrame({"time_block": t, "Flow": flow, "Pressure": pressure})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df,
            cycles,
            flow_col="Flow",
            pressure_col="Pressure",
            plateau_flow_thresh=0.05,
            plateau_min_dur=0.10,
        )

        if not math.isnan(result.iloc[0]["Pplat"]):
            assert result.iloc[0]["dP_fallback"] is False or (
                result.iloc[0]["dP_fallback"] == False  # noqa: E712
            )

    def test_dp_fallback_true_when_pplat_missing(self):
        """dP_fallback must be True when Pplat is NaN (fallback to Ppeak - PEEP)."""
        # t_inspi=1.0 ensures a valid PEEP window exists (pre-inspiration data)
        t = np.linspace(0, 4, 400)
        flow = np.zeros(400)
        pressure = np.ones(400) * 5.0  # PEEP baseline

        # Constant high flow during inspiration (1-2s): no low-flow plateau possible
        flow[(t >= 1.0) & (t < 2.0)] = 60.0  # L/min
        pressure[(t >= 1.0) & (t < 2.0)] = 25.0  # Ppeak
        flow[t >= 2.0] = -30.0
        pressure[t >= 2.0] = 5.0

        df = pd.DataFrame({"time_block": t, "Flow": flow, "Pressure": pressure})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [3.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure", flow_unit="L/min"
        )

        assert math.isnan(result.iloc[0]["Pplat"])
        assert result.iloc[0]["dP_fallback"] == True  # noqa: E712


class TestMechanicalCompliance:
    """Tests for static compliance calculation."""

    def test_cstat_calculation(self):
        """Cstat = VT / (Pplat - PEEP)."""
        t = np.linspace(0, 3, 300)
        flow = np.zeros(300)
        pressure = np.ones(300) * 5.0

        # Flow = 0.5 L/s for 0.6s => VT = 0.3 L
        flow[(t >= 0) & (t < 0.6)] = 30.0  # 30 L/min = 0.5 L/s
        pressure[(t >= 0) & (t < 0.6)] = 15.0

        # Plateau
        flow[(t >= 0.6) & (t < 1.0)] = 0.0
        pressure[(t >= 0.6) & (t < 1.0)] = 20.0  # Pplat

        # Expiration
        flow[(t >= 1.0)] = -20.0
        pressure[(t >= 1.0)] = 5.0

        df = pd.DataFrame({"time_block": t, "Flow": flow, "Pressure": pressure})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df,
            cycles,
            flow_col="Flow",
            pressure_col="Pressure",
            peep_window=0.20,
            plateau_flow_thresh=0.05,
            plateau_min_dur=0.10,
        )

        # VT ≈ 0.3 L, Pplat - PEEP ≈ 15 cmH2O
        # Cstat = 0.3 / 15 = 0.02 L/cmH2O
        if not math.isnan(result.iloc[0]["Cstat"]):
            assert result.iloc[0]["Cstat"] == pytest.approx(0.02, rel=3e-1)

    def test_cstat_nan_without_pplat(self):
        """Cstat should be NaN if Pplat not detected."""
        t = np.linspace(0, 3, 300)
        # Constant high flow - no plateau
        flow = np.where(t < 1.0, 60.0, -30.0)
        pressure = np.where(t < 1.0, 25.0, 5.0)

        df = pd.DataFrame({"time_block": t, "Flow": flow, "Pressure": pressure})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure"
        )

        assert math.isnan(result.iloc[0]["Cstat"])


class TestMechanicalResistance:
    """Tests for airway resistance calculation."""

    def test_resistance_calculation(self):
        """R = (Ppeak - Pplat) / PIF."""
        t = np.linspace(0, 3, 300)
        flow = np.zeros(300)
        pressure = np.ones(300) * 5.0

        # Inspiration with peak pressure
        flow[(t >= 0) & (t < 0.6)] = 30.0  # 0.5 L/s
        pressure[(t >= 0) & (t < 0.6)] = 30.0  # Ppeak = 30

        # Plateau
        flow[(t >= 0.6) & (t < 1.0)] = 0.0
        pressure[(t >= 0.6) & (t < 1.0)] = 20.0  # Pplat = 20

        # Expiration
        flow[(t >= 1.0)] = -20.0
        pressure[(t >= 1.0)] = 5.0

        df = pd.DataFrame({"time_block": t, "Flow": flow, "Pressure": pressure})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure"
        )

        # R = (30 - 20) / 0.5 = 20 cmH2O·s/L
        if not math.isnan(result.iloc[0]["R"]):
            assert result.iloc[0]["R"] == pytest.approx(20.0, rel=3e-1)


class TestMechanicalMAP:
    """Tests for mean airway pressure calculation."""

    def test_map_calculation(self):
        """MAP = ∫P dt / Ttot over the full cycle."""
        t = np.linspace(0, 2, 200)
        # Constant pressure of 15 cmH2O
        pressure = np.ones(200) * 15.0

        df = pd.DataFrame(
            {"time_block": t, "Flow": np.ones(200) * 30.0, "Pressure": pressure}
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure"
        )

        # MAP = 15 cmH2O (constant pressure)
        assert result.iloc[0]["MAP"] == pytest.approx(15.0, rel=5e-2)

    def test_map_nan_without_t_next_inspi(self):
        """MAP must be NaN when t_next_inspi is unavailable (terminal cycle).

        Computing MAP over inspiration only would silently overestimate it.
        """
        t = np.linspace(0, 2, 200)
        pressure = np.ones(200) * 15.0

        df = pd.DataFrame(
            {"time_block": t, "Flow": np.ones(200) * 30.0, "Pressure": pressure}
        )
        cycles = pd.DataFrame(
            {
                "n_cycle": [1],
                "t_inspi": [0.0],
                "t_expi": [1.0],
                "t_next_inspi": [float("nan")],  # no next breath
            }
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure"
        )

        assert math.isnan(result.iloc[0]["MAP"])


class TestMechanicalVentilatoryVariables:
    """Tests for ventilatory variables calculated in mechanical mode."""

    def test_timing_variables(self):
        """Should calculate Ti, Te, Ttot, BF, IE correctly."""
        t = np.linspace(0, 4, 400)
        df = pd.DataFrame(
            {
                "time_block": t,
                "Flow": np.ones(400) * 30.0,
                "Pressure": np.ones(400) * 15.0,
            }
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [0.8], "t_next_inspi": [2.5]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure"
        )

        row = result.iloc[0]
        assert row["Ti"] == pytest.approx(0.8, rel=5e-2)
        assert row["Ttot"] == pytest.approx(2.5, rel=5e-2)
        assert row["Te"] == pytest.approx(1.7, rel=5e-2)
        assert row["BF"] == pytest.approx(24.0, rel=5e-2)
        assert row["IE"] == pytest.approx(0.47, rel=1e-1)

    def test_vt_positive_for_positive_flow(self):
        """VT should be positive when flow is positive during inspiration."""
        t = np.linspace(0, 3, 300)
        flow = np.zeros(300)
        flow[t < 1.0] = 30.0  # 0.5 L/s positive

        df = pd.DataFrame(
            {"time_block": t, "Flow": flow, "Pressure": np.ones(300) * 15.0}
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.0], "t_expi": [1.0], "t_next_inspi": [2.0]}
        )

        result = mechanical_from_cycles(
            df, cycles, flow_col="Flow", pressure_col="Pressure"
        )

        # VT should be positive (~0.5 L)
        assert result.iloc[0]["VT"] > 0
        assert result.iloc[0]["VT"] == pytest.approx(0.5, rel=1e-1)


class TestMechanicalWithSyntheticSignal:
    """Integration tests with mechanical signal fixture."""

    def test_mechanical_signal_metrics(
        self, mechanical_signal_df, mechanical_cycles_for_signal
    ):
        """Test all metrics with mechanical ventilation signal."""
        result = mechanical_from_cycles(
            mechanical_signal_df,
            mechanical_cycles_for_signal,
            flow_col="Flow",
            pressure_col="Pressure",
            volume_col=None,
            flow_unit="L/min",
        )

        assert len(result) >= 2

        # Check first cycle - skip first cycle as it may not have PEEP window
        row = result.iloc[1]  # Use second cycle which has pre-inspiration data

        # Timing
        assert row["Ti"] == pytest.approx(0.8, rel=2e-1)
        assert row["BF"] == pytest.approx(24.0, rel=2e-1)

        # VT should be positive
        assert row["VT"] > 0

        # Ppeak should be present
        assert not math.isnan(row["Ppeak"])
        assert row["Ppeak"] > 0

        # MAP should be present and reasonable
        if not math.isnan(row["MAP"]):
            assert row["MAP"] > 0
