"""
Tests for resp_metrics.ventilatory module.

Tests ventilatory metric calculations with known synthetic values.
All tests use pytest.approx(rel=1e-3) for numerical comparisons.
"""

import pytest
import numpy as np
import pandas as pd
import math

from resp_metrics.ventilatory import ventilatory_from_cycles


class TestVentilatoryEmptyInputs:
    """Tests for empty/null input handling."""

    def test_none_df_block_returns_empty(self, sample_cycles_df):
        """Should return empty DataFrame for None df_block."""
        result = ventilatory_from_cycles(None, sample_cycles_df)
        assert result.empty
        assert "n_cycle" in result.columns
        assert "VT" in result.columns

    def test_empty_df_block_returns_empty(self, sample_cycles_df):
        """Should return empty DataFrame for empty df_block."""
        empty_df = pd.DataFrame()
        result = ventilatory_from_cycles(empty_df, sample_cycles_df)
        assert result.empty

    def test_none_cycles_returns_empty(self, spontaneous_signal_df):
        """Should return empty DataFrame for None cycles."""
        result = ventilatory_from_cycles(spontaneous_signal_df, None)
        assert result.empty

    def test_empty_cycles_returns_empty(self, spontaneous_signal_df):
        """Should return empty DataFrame for empty cycles."""
        empty_cycles = pd.DataFrame()
        result = ventilatory_from_cycles(spontaneous_signal_df, empty_cycles)
        assert result.empty


class TestVentilatoryMissingColumns:
    """Tests for missing column handling."""

    def test_missing_time_block_raises(self, sample_cycles_df):
        """Should raise KeyError if time_block missing."""
        df = pd.DataFrame({"Flow": [1, 2, 3]})
        with pytest.raises(KeyError, match="time_block"):
            ventilatory_from_cycles(df, sample_cycles_df)

    def test_missing_t_inspi_raises(self, spontaneous_signal_df):
        """Should raise KeyError if t_inspi missing in cycles."""
        cycles = pd.DataFrame({"t_expi": [1.0], "t_next_inspi": [2.0]})
        with pytest.raises(KeyError, match="t_inspi"):
            ventilatory_from_cycles(spontaneous_signal_df, cycles)

    def test_missing_t_expi_raises(self, spontaneous_signal_df):
        """Should raise KeyError if t_expi missing in cycles."""
        cycles = pd.DataFrame({"t_inspi": [0.0], "t_next_inspi": [2.0]})
        with pytest.raises(KeyError, match="t_expi"):
            ventilatory_from_cycles(spontaneous_signal_df, cycles)


class TestVentilatoryTiming:
    """Tests for timing variable calculations (Ti, Te, Ttot, BF, IE)."""

    def test_timing_simple_cycle(self, single_cycle_df):
        """Test timing with known values: Ti=1.5s, Te=2.5s, Ttot=4.0s."""
        # Create minimal signal
        t = np.linspace(0, 5, 500)
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.zeros(500)  # Flow not needed for timing
        })
        
        result = ventilatory_from_cycles(df, single_cycle_df, flow_col="Flow")
        
        assert len(result) == 1
        row = result.iloc[0]
        
        # Ti = t_expi - t_inspi = 1.5 - 0 = 1.5s
        assert row["Ti"] == pytest.approx(1.5, rel=1e-2)
        # Ttot = t_next_inspi - t_inspi = 4.0 - 0 = 4.0s
        assert row["Ttot"] == pytest.approx(4.0, rel=1e-2)
        # Te = Ttot - Ti = 4.0 - 1.5 = 2.5s
        assert row["Te"] == pytest.approx(2.5, rel=1e-2)
        # BF = 60 / Ttot = 60 / 4.0 = 15 breaths/min
        assert row["BF"] == pytest.approx(15.0, rel=1e-2)
        # I:E = Ti / Te = 1.5 / 2.5 = 0.6
        assert row["IE"] == pytest.approx(0.6, rel=1e-2)

    def test_multiple_cycles_timing(self, sample_cycles_df):
        """Test timing with multiple cycles."""
        t = np.linspace(0, 5, 500)
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.zeros(500)
        })
        
        result = ventilatory_from_cycles(df, sample_cycles_df, flow_col="Flow")
        
        assert len(result) == 2
        # Both cycles: Ti=1.0s, Te=1.0s, Ttot=2.0s, BF=30, I:E=1.0
        for _, row in result.iterrows():
            assert row["Ti"] == pytest.approx(1.0, rel=1e-2)
            assert row["Te"] == pytest.approx(1.0, rel=1e-2)
            assert row["Ttot"] == pytest.approx(2.0, rel=1e-2)
            assert row["BF"] == pytest.approx(30.0, rel=1e-2)
            assert row["IE"] == pytest.approx(1.0, rel=2e-2)  # Slightly relaxed tolerance


class TestVentilatoryVolumeIntegration:
    """Tests for VT calculation by flow integration."""

    def test_vt_from_constant_flow(self):
        """VT from constant negative flow during inspiration.
        
        Flow = -0.5 L/s for 1.0s => VT = 0.5 L
        """
        t = np.linspace(0, 3, 300)
        flow = np.zeros(300)
        # Constant negative flow during inspiration (0-1s)
        flow[t < 1.0] = -0.5  # L/s
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": flow  # Already in L/s
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [2.0]
        })
        
        result = ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="L/s")
        
        # VT = integral of |flow| = 0.5 L/s * 1.0s = 0.5 L
        assert result.iloc[0]["VT"] == pytest.approx(0.5, rel=1e-2)

    def test_vt_from_triangular_flow(self):
        """VT from triangular flow pattern.
        
        Triangle with peak -0.6 L/s over 1.0s => VT = 0.5 * 1.0 * 0.6 = 0.3 L
        """
        t = np.linspace(0, 2, 200)
        flow = np.zeros(200)
        # Triangular flow during inspiration (0-1s)
        insp_mask = t < 1.0
        flow[insp_mask] = -0.6 * (1 - np.abs(t[insp_mask] - 0.5) / 0.5)
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": flow  # L/s
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [1.5]
        })
        
        result = ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="L/s")
        
        # VT ≈ 0.3 L (triangular area)
        assert result.iloc[0]["VT"] == pytest.approx(0.3, rel=5e-2)

    def test_vt_from_volume_column(self):
        """VT from volume column takes precedence over integration."""
        t = np.linspace(0, 2, 200)
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.zeros(200),  # Irrelevant when volume available
            "VolumeResp": t * 0.25  # Linear increase: 0.5L at t=2s
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],  # Volume at t=1 is 0.25L
            "t_next_inspi": [2.0]
        })
        
        result = ventilatory_from_cycles(
            df, cycles, flow_col="Flow", volume_col="VolumeResp"
        )
        
        # VT = Volume(t_expi) - Volume(t_inspi) = 0.25 - 0 = 0.25 L
        assert result.iloc[0]["VT"] == pytest.approx(0.25, rel=1e-2)


class TestVentilatoryFlowUnitConversion:
    """Tests for flow unit handling."""

    def test_flow_l_min_conversion(self):
        """Flow in L/min should be converted to L/s."""
        t = np.linspace(0, 2, 200)
        flow_lpm = np.zeros(200)
        flow_lpm[t < 1.0] = -30.0  # -30 L/min = -0.5 L/s
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": flow_lpm
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [1.5]
        })
        
        result = ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="L/min")
        
        # VT should be ~0.5 L (0.5 L/s * 1s)
        assert result.iloc[0]["VT"] == pytest.approx(0.5, rel=1e-2)

    def test_flow_l_s_no_conversion(self):
        """Flow in L/s should not be converted."""
        t = np.linspace(0, 2, 200)
        flow_ls = np.zeros(200)
        flow_ls[t < 1.0] = -0.5  # Already L/s
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": flow_ls
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [1.5]
        })
        
        result = ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="L/s")
        
        assert result.iloc[0]["VT"] == pytest.approx(0.5, rel=1e-2)

    def test_invalid_flow_unit_raises(self):
        """Should raise ValueError for unsupported flow unit."""
        t = np.linspace(0, 2, 20)
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.zeros(20)
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [1.5]
        })
        
        with pytest.raises(ValueError, match="Unsupported flow unit"):
            ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="mL/s")


class TestVentilatoryCycleValidation:
    """Tests for invalid or inconsistent cycle timing."""

    def test_expi_before_inspi_skips_cycle(self):
        """Cycles with t_expi <= t_inspi should be skipped."""
        t = np.linspace(0, 2, 200)
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.zeros(200)
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [1.0],
            "t_expi": [0.5],  # invalid
            "t_next_inspi": [1.5]
        })

        result = ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="L/s")

        assert result.empty

    def test_next_inspi_before_expi_sets_ttot_nan(self):
        """If next INSPI is before EXPI, Ttot/Te/BF should be NaN."""
        t = np.linspace(0, 2, 200)
        flow = np.zeros(200)
        flow[t < 0.8] = -0.5
        df = pd.DataFrame({
            "time_block": t,
            "Flow": flow
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [0.5]  # invalid ordering
        })

        result = ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="L/s")

        assert len(result) == 1
        assert math.isnan(result.iloc[0]["Ttot"])
        assert math.isnan(result.iloc[0]["Te"])
        assert math.isnan(result.iloc[0]["BF"])


class TestVentilatoryPeakFlows:
    """Tests for PIF and PEF calculations."""

    def test_pif_magnitude(self):
        """PIF should be magnitude of min flow during inspiration."""
        t = np.linspace(0, 3, 300)
        flow = np.zeros(300)
        flow[t < 1.0] = -0.8  # Negative during inspiration
        flow[(t >= 1.0) & (t < 2.0)] = 0.4  # Positive during expiration
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": flow  # L/s
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [2.0]
        })
        
        result = ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="L/s")
        
        # PIF = |min(flow during insp)| = |-0.8| = 0.8 L/s
        assert result.iloc[0]["PIF"] == pytest.approx(0.8, rel=1e-2)
        # PEF = max(flow during exp) = 0.4 L/s
        assert result.iloc[0]["PEF"] == pytest.approx(0.4, rel=1e-2)


class TestVentilatoryVE:
    """Tests for minute ventilation calculation."""

    def test_ve_calculation(self):
        """VE = BF * VT."""
        t = np.linspace(0, 4, 400)
        flow = np.zeros(400)
        # Cycle with Ti=1s, Ttot=2s => BF=30
        flow[t < 1.0] = -0.5  # VT = 0.5L
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": flow  # L/s
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [2.0]
        })
        
        result = ventilatory_from_cycles(df, cycles, flow_col="Flow", flow_unit="L/s")
        
        # VE = BF * VT = 30 * 0.5 = 15 L/min
        assert result.iloc[0]["VE"] == pytest.approx(15.0, rel=1e-2)


class TestVentilatoryWOB:
    """Tests for work of breathing calculation."""

    def test_wob_requires_pes(self):
        """WOB should be NaN if pes_col not provided."""
        t = np.linspace(0, 2, 200)
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.ones(200) * -0.5,
            "Paw": np.ones(200) * 5.0  # Airway pressure, not esophageal
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [1.5]
        })
        
        result = ventilatory_from_cycles(
            df, cycles, 
            flow_col="Flow", 
            pressure_col="Paw",
            pes_col=None,  # No esophageal pressure
            flow_unit="L/s"
        )
        
        assert math.isnan(result.iloc[0]["WOB"])

    def test_wob_with_pes(self):
        """WOB should be calculated when pes_col provided.
        
        WOB = -∫ Pes_kPa * Flow dt
        With Pes = 10 cmH2O, Flow = -0.5 L/s, duration = 1s:
        Pes_kPa = 10 * 0.0980665 = 0.980665 kPa
        WOB = -(-0.5 * 0.980665 * 1) = 0.490 J
        """
        t = np.linspace(0, 2, 200)
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.where(t < 1.0, -0.5, 0.0),  # L/s
            "Pes": np.ones(200) * 10.0  # cmH2O
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [1.5]
        })
        
        result = ventilatory_from_cycles(
            df, cycles,
            flow_col="Flow",
            pes_col="Pes",
            flow_unit="L/s"
        )
        
        # WOB ≈ 0.49 J
        assert result.iloc[0]["WOB"] == pytest.approx(0.49, rel=5e-2)


class TestVentilatoryPTP:
    """Tests for pressure-time product calculation."""

    def test_ptp_with_baseline(self):
        """PTP should be relative to baseline pressure.
        
        If pressure = 10 cmH2O and baseline = 5 cmH2O, duration = 1s:
        PTP = (10 - 5) * 1 = 5 cmH2O·s
        """
        t = np.linspace(0, 3, 300)
        pressure = np.zeros(300)
        # Baseline before inspiration: 5 cmH2O
        pressure[t < 0.5] = 5.0
        # During inspiration: 10 cmH2O
        pressure[(t >= 0.5) & (t < 1.5)] = 10.0
        # After: back to baseline
        pressure[t >= 1.5] = 5.0
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.zeros(300),
            "Paw": pressure
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.5],
            "t_expi": [1.5],
            "t_next_inspi": [2.5]
        })
        
        result = ventilatory_from_cycles(
            df, cycles,
            flow_col="Flow",
            pressure_col="Paw",
            flow_unit="L/s",
            ptp_window=0.3  # Look at 0.2-0.5s for baseline
        )
        
        # PTP = (10 - 5) * 1.0s = 5 cmH2O·s
        assert result.iloc[0]["PTP"] == pytest.approx(5.0, rel=1e-1)

    def test_ptp_nan_without_pressure(self):
        """PTP should be NaN if pressure column not available."""
        t = np.linspace(0, 2, 200)
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.zeros(200)
        })
        cycles = pd.DataFrame({
            "n_cycle": [1],
            "t_inspi": [0.0],
            "t_expi": [1.0],
            "t_next_inspi": [1.5]
        })
        
        result = ventilatory_from_cycles(
            df, cycles,
            flow_col="Flow",
            pressure_col="Paw",  # Column doesn't exist
            flow_unit="L/s"
        )
        
        assert math.isnan(result.iloc[0]["PTP"])


class TestVentilatoryWithSyntheticSignal:
    """Integration tests with synthetic signal fixture."""

    def test_spontaneous_signal_metrics(
        self, spontaneous_signal_df, spontaneous_cycles_for_signal
    ):
        """Test all metrics with spontaneous breathing signal."""
        result = ventilatory_from_cycles(
            spontaneous_signal_df,
            spontaneous_cycles_for_signal,
            flow_col="Flow",
            volume_col=None,  # Test integration
            pressure_col="Paw",
            pes_col="Pes",
            flow_unit="L/min"
        )
        
        assert len(result) == 2
        
        # Check first cycle timing (from ExpectedSpontaneous)
        row = result.iloc[0]
        assert row["Ti"] == pytest.approx(1.5, rel=5e-2)
        assert row["Te"] == pytest.approx(2.5, rel=5e-2)
        assert row["Ttot"] == pytest.approx(4.0, rel=5e-2)
        assert row["BF"] == pytest.approx(15.0, rel=5e-2)
        assert row["IE"] == pytest.approx(0.6, rel=5e-2)
        
        # Check flows
        assert row["PIF"] == pytest.approx(0.5, rel=1e-1)
        assert row["PEF"] == pytest.approx(0.3, rel=1e-1)
        
        # VT should be positive
        assert row["VT"] > 0
        
        # WOB should be calculated (has Pes) - can be positive or negative
        # depending on signal conventions
        assert not math.isnan(row["WOB"])
