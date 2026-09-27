"""Tests for respiratory effort indices (Pes / Pga / Pdi)."""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from resp_metrics.effort import effort_from_cycles

EFFORT_COLUMNS = [
    "n_cycle",
    "t_inspi",
    "t_expi",
    "Pes_ee",
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
    "dPga_exp",
    "PTPga_exp",
    "TTIabd",
    "pes_artifact",
]


def _rect_effort(pes_amp=-10.0, pga_amp=4.0, fs=100, dur=6.0):
    """One cycle: baseline 0, inspiration [1, 2] held at the given amplitudes.

    The window is inclusive of t_expi, matching how the package slices cycles.
    """
    t = np.arange(0, dur, 1 / fs)
    pes = np.zeros_like(t)
    pga = np.zeros_like(t)
    insp = (t >= 1.0) & (t <= 2.0)
    pes[insp] = pes_amp
    pga[insp] = pga_amp
    df = pd.DataFrame({"time_block": t, "Pes": pes, "Pga": pga})
    cycles = pd.DataFrame(
        {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [4.0]}
    )
    return df, cycles


class TestEffortEmptyInputs:
    """Empty or missing inputs return an empty frame with the full schema."""

    def test_empty_signal_returns_empty_with_columns(self):
        """An empty df_block yields an empty frame carrying every column."""
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [4.0]}
        )
        result = effort_from_cycles(pd.DataFrame(), cycles)
        assert result.empty
        assert list(result.columns) == EFFORT_COLUMNS

    def test_empty_cycles_returns_empty_with_columns(self):
        """An empty cycles_df yields an empty frame carrying every column."""
        df, _ = _rect_effort()
        result = effort_from_cycles(df, pd.DataFrame())
        assert result.empty
        assert list(result.columns) == EFFORT_COLUMNS

    def test_none_inputs_return_empty(self):
        """None inputs are tolerated and yield an empty frame."""
        assert effort_from_cycles(None, None).empty


class TestEffortMissingColumns:
    """Missing required columns raise, missing channels yield NaN."""

    def test_missing_time_block_raises(self):
        """df_block without 'time_block' raises KeyError."""
        _, cycles = _rect_effort()
        df = pd.DataFrame({"Pes": [0.0, 1.0], "Pga": [0.0, 1.0]})
        with pytest.raises(KeyError, match="time_block"):
            effort_from_cycles(df, cycles)

    @pytest.mark.parametrize("missing", ["t_inspi", "t_expi", "t_next_inspi"])
    def test_missing_cycle_column_raises(self, missing):
        """cycles_df without a required time column raises KeyError."""
        df, cycles = _rect_effort()
        with pytest.raises(KeyError, match=missing):
            effort_from_cycles(df, cycles.drop(columns=[missing]))

    def test_no_pressure_channels_gives_nan(self):
        """Without any pressure channel every metric is NaN, not a default."""
        df, cycles = _rect_effort()
        result = effort_from_cycles(df[["time_block"]], cycles)
        row = result.iloc[0]
        for col in ("dPes", "dPga", "dPdi", "PTPes", "PTPga", "PTPdi", "TTIdi"):
            assert math.isnan(row[col])

    def test_pes_only_gives_pes_metrics(self):
        """With Pes alone, only the Pes metrics are finite."""
        df, cycles = _rect_effort()
        result = effort_from_cycles(df.drop(columns=["Pga"]), cycles, pga_col=None)
        row = result.iloc[0]
        assert row["dPes"] == pytest.approx(10.0, rel=1e-2)
        assert math.isnan(row["dPga"])
        assert math.isnan(row["dPdi"])


class TestEffortSwings:
    """Inspiratory pressure swings, reported as positive magnitudes."""

    def test_swings_match_known_amplitudes(self):
        """Pes falls 10, Pga rises 4, so Pdi rises 14 cmH2O."""
        df, cycles = _rect_effort()
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["dPes"] == pytest.approx(10.0, rel=1e-2)
        assert row["dPga"] == pytest.approx(4.0, rel=1e-2)
        assert row["dPdi"] == pytest.approx(14.0, rel=1e-2)

    def test_dpdi_equals_dpes_plus_dpga(self):
        """With a synchronous rectangular effort, dPdi = dPes + dPga."""
        df, cycles = _rect_effort(pes_amp=-7.5, pga_amp=2.5)
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["dPdi"] == pytest.approx(row["dPes"] + row["dPga"], rel=1e-2)

    def test_falling_pga_gives_negative_swing(self):
        """A gastric pressure that falls during inspiration is not clipped."""
        df, cycles = _rect_effort(pga_amp=-3.0)
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["dPga"] < 0


class TestEffortPTP:
    """Pressure-time products, integrated over inspiration."""

    def test_ptp_equals_amplitude_times_duration(self):
        """A 10 cmH2O deflection held 1.0 s gives PTPes = 10 cmH2O.s."""
        df, cycles = _rect_effort()
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["PTPes"] == pytest.approx(10.0, rel=5e-2)
        assert row["PTPga"] == pytest.approx(4.0, rel=5e-2)
        assert row["PTPdi"] == pytest.approx(14.0, rel=5e-2)

    def test_ptp_positive_for_inspiratory_effort(self):
        """All three PTPs are positive for a normal inspiratory effort."""
        df, cycles = _rect_effort()
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["PTPes"] > 0
        assert row["PTPga"] > 0
        assert row["PTPdi"] > 0

    def test_baseline_offset_cancels(self):
        """A constant DC offset on both channels leaves the metrics unchanged."""
        df, cycles = _rect_effort()
        shifted = df.copy()
        shifted["Pes"] = shifted["Pes"] + 25.0
        shifted["Pga"] = shifted["Pga"] - 13.0
        base = effort_from_cycles(df, cycles).iloc[0]
        offset = effort_from_cycles(shifted, cycles).iloc[0]
        for col in ("dPes", "dPga", "dPdi", "PTPes", "PTPga", "PTPdi"):
            assert offset[col] == pytest.approx(base[col], rel=1e-6)


class TestEffortPdiDerivation:
    """Pdi is read from its channel or derived as Pga - Pes."""

    def test_recorded_pdi_matches_derived(self):
        """Reading Pdi from a channel matches deriving it from Pga - Pes."""
        df, cycles = _rect_effort()
        derived = effort_from_cycles(df, cycles).iloc[0]
        with_channel = df.assign(Pdi=df["Pga"] - df["Pes"])
        recorded = effort_from_cycles(with_channel, cycles, pdi_col="Pdi").iloc[0]
        assert recorded["dPdi"] == pytest.approx(derived["dPdi"], rel=1e-9)
        assert recorded["PTPdi"] == pytest.approx(derived["PTPdi"], rel=1e-9)

    def test_pdi_channel_takes_precedence(self):
        """An explicit pdi_col is used rather than the Pga - Pes derivation."""
        df, cycles = _rect_effort()
        df = df.assign(Pdi=(df["Pga"] - df["Pes"]) / 2.0)
        row = effort_from_cycles(df, cycles, pdi_col="Pdi").iloc[0]
        assert row["dPdi"] == pytest.approx(7.0, rel=1e-2)

    def test_missing_pdi_col_falls_back_to_derivation(self):
        """A pdi_col absent from df_block falls back to Pga - Pes."""
        df, cycles = _rect_effort()
        row = effort_from_cycles(df, cycles, pdi_col="NotThere").iloc[0]
        assert row["dPdi"] == pytest.approx(14.0, rel=1e-2)


class TestEffortRatios:
    """PTPdi:PTPes, the diaphragmatic share of the inspiratory effort."""

    def test_ratio_matches_component_ptps(self):
        """PTPdi_PTPes equals PTPdi divided by PTPes."""
        df, cycles = _rect_effort()
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["PTPdi_PTPes"] == pytest.approx(
            row["PTPdi"] / row["PTPes"], rel=1e-9
        )
        assert row["PTPdi_PTPes"] == pytest.approx(1.4, rel=1e-2)

    def test_ratio_nan_when_ptpes_not_positive(self):
        """A null Pes effort makes the ratio undefined rather than infinite."""
        df, cycles = _rect_effort(pes_amp=0.0)
        row = effort_from_cycles(df, cycles).iloc[0]
        assert math.isnan(row["PTPdi_PTPes"])


class TestEffortTTIdi:
    """Tension-time index of the diaphragm (ATS/ERS definition)."""

    def test_ttidi_equals_ptpdi_over_pdimax_ttot(self):
        """TTIdi = (mean inspiratory Pdi / Pdi_max) x (Ti/Ttot) = PTPdi/(Pdi_max.Ttot)."""
        df, cycles = _rect_effort()
        row = effort_from_cycles(df, cycles, pdi_max=100.0).iloc[0]
        # Ttot = 4 - 1 = 3 s, PTPdi ~ 14 cmH2O.s
        assert row["TTIdi"] == pytest.approx(row["PTPdi"] / (100.0 * 3.0), rel=1e-9)
        assert row["TTIdi"] == pytest.approx(14.0 / 300.0, rel=5e-2)

    def test_ttidi_nan_without_pdi_max(self):
        """Pdi_max is not derivable from tidal breathing: TTIdi stays NaN."""
        df, cycles = _rect_effort()
        assert math.isnan(effort_from_cycles(df, cycles).iloc[0]["TTIdi"])

    def test_ttidi_nan_for_non_positive_pdi_max(self):
        """A non-positive Pdi_max is rejected rather than producing a sign flip."""
        df, cycles = _rect_effort()
        assert math.isnan(effort_from_cycles(df, cycles, pdi_max=0.0).iloc[0]["TTIdi"])

    def test_ttidi_nan_without_next_inspi(self):
        """Without t_next_inspi, Ttot is unknown and TTIdi is NaN."""
        df, cycles = _rect_effort()
        cycles["t_next_inspi"] = np.nan
        assert math.isnan(
            effort_from_cycles(df, cycles, pdi_max=100.0).iloc[0]["TTIdi"]
        )

    def test_ttidi_scales_inversely_with_pdi_max(self):
        """Doubling Pdi_max halves TTIdi."""
        df, cycles = _rect_effort()
        a = effort_from_cycles(df, cycles, pdi_max=50.0).iloc[0]["TTIdi"]
        b = effort_from_cycles(df, cycles, pdi_max=100.0).iloc[0]["TTIdi"]
        assert b == pytest.approx(a / 2.0, rel=1e-9)


class TestEffortCycleValidation:
    """Invalid cycles are dropped, as elsewhere in the package."""

    def test_expi_before_inspi_skips_cycle(self):
        """A cycle whose EXPI precedes its INSPI is dropped."""
        df, _ = _rect_effort()
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [2.0], "t_expi": [1.0], "t_next_inspi": [4.0]}
        )
        result = effort_from_cycles(df, cycles)
        assert result.empty
        assert list(result.columns) == EFFORT_COLUMNS

    def test_zero_length_inspiration_skips_cycle(self):
        """A cycle whose inspiration snaps to a single sample is dropped."""
        df, _ = _rect_effort()
        cycles = pd.DataFrame(
            {
                "n_cycle": [1],
                "t_inspi": [1.0],
                "t_expi": [1.0 + 1e-9],
                "t_next_inspi": [4.0],
            }
        )
        assert effort_from_cycles(df, cycles).empty

    def test_n_cycle_synthesised_when_absent(self):
        """A cycles_df without n_cycle gets a 1-based index."""
        df, cycles = _rect_effort()
        result = effort_from_cycles(df, cycles.drop(columns=["n_cycle"]))
        assert result.iloc[0]["n_cycle"] == 1


class TestEffortBlockColumns:
    """block_name / block are prepended and stay leading."""

    def test_block_columns_prepended(self):
        """Explicit block identifiers appear first, block_name before block."""
        df, cycles = _rect_effort()
        result = effort_from_cycles(df, cycles, block=2, block_name="Block 2")
        assert list(result.columns)[:2] == ["block_name", "block"]
        assert result.iloc[0]["block"] == 2

    def test_block_columns_inherited_from_cycles(self):
        """Block identifiers carried by cycles_df are reused."""
        df, cycles = _rect_effort()
        cycles = cycles.assign(block=3, block_name="Block 3")
        result = effort_from_cycles(df, cycles)
        assert result.iloc[0]["block_name"] == "Block 3"


class TestEffortWithSyntheticSignal:
    """End-to-end check against the shared fixture."""

    def test_effort_signal_metrics(self, effort_signal_df, effort_cycles_for_signal):
        """Both cycles of the fixture reproduce the expected constants."""
        # Expected values are the ExpectedEffort constants in conftest.py
        result = effort_from_cycles(
            effort_signal_df, effort_cycles_for_signal, pdi_max=100.0
        )
        assert len(result) == 2
        for _, row in result.iterrows():
            assert row["dPes"] == pytest.approx(10.0, rel=1e-2)
            assert row["dPga"] == pytest.approx(4.0, rel=1e-2)
            assert row["dPdi"] == pytest.approx(14.0, rel=1e-2)
            assert row["PTPes"] == pytest.approx(10.0, rel=5e-2)
            assert row["PTPga"] == pytest.approx(4.0, rel=5e-2)
            assert row["PTPdi"] == pytest.approx(14.0, rel=5e-2)
            assert row["PTPdi_PTPes"] == pytest.approx(1.4, rel=1e-2)
            assert row["TTIdi"] == pytest.approx(0.035, rel=5e-2)


class TestEffortWOB:
    """Work of breathing, which needs both Pes and flow."""

    @staticmethod
    def _pes_and_flow():
        """Pes drops to -10 cmH2O while inspiratory flow is -0.5 L/s for 1.0 s.

        Pmus = Pes_baseline - Pes = 0 - (-10) = 10 cmH2O = 0.980665 kPa
        -Flow = 0.5 L/s, so WOB = 0.980665 x 0.5 x 1.0 s ~ 0.490 J.
        """
        t = np.arange(0, 3, 0.01)
        insp = (t >= 0.5) & (t <= 1.5)
        df = pd.DataFrame(
            {
                "time_block": t,
                "Flow": np.where(insp, -0.5, 0.0),
                "Pes": np.where(insp, -10.0, 0.0),
            }
        )
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [0.5], "t_expi": [1.5], "t_next_inspi": [2.5]}
        )
        return df, cycles

    def test_wob_matches_hand_computed_value(self):
        """A 10 cmH2O effort at 0.5 L/s for 1.0 s gives about 0.49 J."""
        df, cycles = self._pes_and_flow()
        row = effort_from_cycles(
            df, cycles, pga_col=None, flow_col="Flow", flow_unit="L/s"
        ).iloc[0]
        assert row["WOB"] == pytest.approx(0.49, rel=5e-2)
        assert row["WOB"] > 0

    def test_wob_nan_without_pes(self):
        """Airway pressure is not substituted for Pes: no Pes means no WOB."""
        df, cycles = self._pes_and_flow()
        df = df.drop(columns=["Pes"]).assign(Paw=5.0)
        row = effort_from_cycles(
            df, cycles, pes_col=None, pga_col=None, flow_col="Flow", flow_unit="L/s"
        ).iloc[0]
        assert math.isnan(row["WOB"])

    def test_wob_nan_without_flow(self):
        """WOB needs the inspired volume, hence the flow signal."""
        df, cycles = self._pes_and_flow()
        row = effort_from_cycles(
            df.drop(columns=["Flow"]), cycles, pga_col=None, flow_unit="L/s"
        ).iloc[0]
        assert math.isnan(row["WOB"])

    def test_wob_respects_flow_unit(self):
        """The same flow expressed in L/min yields the same WOB."""
        df, cycles = self._pes_and_flow()
        in_lps = effort_from_cycles(
            df, cycles, pga_col=None, flow_col="Flow", flow_unit="L/s"
        ).iloc[0]["WOB"]
        in_lpm = effort_from_cycles(
            df.assign(Flow=df["Flow"] * 60),
            cycles,
            pga_col=None,
            flow_col="Flow",
            flow_unit="L/min",
        ).iloc[0]["WOB"]
        assert in_lpm == pytest.approx(in_lps, rel=1e-9)


class TestEffortPgaCorrected:
    """dPga_corr / PTPga_corr, referenced to the Pga nadir."""

    @staticmethod
    def _with_abdominal_relaxation():
        """Pga elevated at end-expiration, relaxing to 2, then rising to 12.

        Baseline (median of the 0.2 s before onset) is 5. During inspiration
        Pga first falls to 2 (abdominal relaxation) then ramps to 12.
        Uncorrected: dPga = 12 - 5 = 7. Corrected: dPga = 12 - 2 = 10.
        """
        t = np.arange(0, 4, 0.01)
        pga = np.full_like(t, 5.0)
        insp = (t >= 1.0) & (t <= 2.0)
        # relaxation over the first 0.3 s, then a linear ramp to 12
        rel = (t >= 1.0) & (t < 1.3)
        pga[rel] = 5.0 + (2.0 - 5.0) * (t[rel] - 1.0) / 0.3
        ramp = (t >= 1.3) & (t <= 2.0)
        pga[ramp] = 2.0 + (12.0 - 2.0) * (t[ramp] - 1.3) / 0.7
        pga[~insp & (t > 2.0)] = 5.0
        df = pd.DataFrame({"time_block": t, "Pes": np.zeros_like(t), "Pga": pga})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [3.0]}
        )
        return df, cycles

    def test_corrected_swing_uses_nadir(self):
        """With an abdominal relaxation dip, the corrected swing is larger."""
        df, cycles = self._with_abdominal_relaxation()
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["dPga"] == pytest.approx(7.0, rel=2e-2)
        assert row["dPga_corr"] == pytest.approx(10.0, rel=2e-2)
        assert row["dPga_corr"] > row["dPga"]

    def test_corrected_ptp_is_positive(self):
        """PTPga_corr integrates from the nadir and stays positive."""
        df, cycles = self._with_abdominal_relaxation()
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["PTPga_corr"] > 0

    def test_matches_uncorrected_without_relaxation(self):
        """With no expiratory recruitment the correction is a no-op."""
        t = np.arange(0, 4, 0.01)
        pga = np.zeros_like(t)
        ramp = (t >= 1.0) & (t <= 2.0)
        pga[ramp] = 8.0 * (t[ramp] - 1.0)
        df = pd.DataFrame({"time_block": t, "Pes": np.zeros_like(t), "Pga": pga})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [3.0]}
        )
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["dPga_corr"] == pytest.approx(row["dPga"], rel=1e-6)

    def test_nan_without_pga(self):
        """Both corrected columns are NaN when Pga is absent."""
        df, cycles = _rect_effort()
        row = effort_from_cycles(df, cycles, pga_col=None).iloc[0]
        assert math.isnan(row["dPga_corr"])
        assert math.isnan(row["PTPga_corr"])

    def test_pdi_and_pes_untouched_by_correction(self):
        """The correction is local to Pga: Pes and Pdi metrics are unchanged."""
        df, cycles = self._with_abdominal_relaxation()
        wide = effort_from_cycles(df, cycles, pga_nadir_frac=0.5).iloc[0]
        narrow = effort_from_cycles(df, cycles, pga_nadir_frac=0.2).iloc[0]
        for col in ("dPes", "PTPes", "dPdi", "PTPdi"):
            assert wide[col] == pytest.approx(narrow[col], rel=1e-9)
        assert wide["dPga"] == pytest.approx(narrow["dPga"], rel=1e-9)

    def test_nadir_search_stays_inside_inspiration(self):
        """pga_nadir_frac > 1 must not let the nadir land in expiration."""
        t = np.arange(0, 4, 0.01)
        pga = np.full_like(t, 5.0)
        pga[(t >= 1.0) & (t <= 2.0)] = 8.0
        pga[t > 2.0] = 1.0  # Pga collapses in expiration
        df = pd.DataFrame({"time_block": t, "Pes": np.zeros_like(t), "Pga": pga})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [3.0]}
        )
        for frac in (1.0, 1.5, 3.0):
            row = effort_from_cycles(df, cycles, pga_nadir_frac=frac).iloc[0]
            # the expiratory collapse to 1.0 must never be picked as reference
            assert row["dPga_corr"] == pytest.approx(0.0, abs=1e-9)
            assert np.isfinite(row["PTPga_corr"])

    def test_all_nan_pga_gives_nan_not_crash(self):
        """An unusable Pga channel degrades to NaN, as everywhere else."""
        df, cycles = _rect_effort()
        df = df.assign(Pga=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = effort_from_cycles(df, cycles).iloc[0]
        assert math.isnan(row["dPga_corr"])
        assert math.isnan(row["PTPga_corr"])


class TestEffortEndExpiratoryPes:
    """Pes_ee, the end-expiratory oesophageal pressure."""

    @staticmethod
    def _cycle_with_eelv(rest_level):
        """Pes sits at `rest_level` at rest and drops 10 cmH2O on inspiration."""
        t = np.arange(0, 4, 0.01)
        pes = np.full_like(t, rest_level)
        pes[(t >= 1.0) & (t <= 2.0)] = rest_level - 10.0
        df = pd.DataFrame({"time_block": t, "Pes": pes})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [3.0]}
        )
        return df, cycles

    def test_reports_the_resting_level(self):
        """Pes_ee is the pre-inspiratory level, not a difference."""
        df, cycles = self._cycle_with_eelv(-5.0)
        row = effort_from_cycles(df, cycles, pga_col=None).iloc[0]
        assert row["Pes_ee"] == pytest.approx(-5.0, rel=1e-6)

    def test_is_absolute_while_the_swing_is_not(self):
        """A rise in resting level moves Pes_ee but leaves dPes and PTPes alone."""
        low = effort_from_cycles(*self._cycle_with_eelv(-5.0), pga_col=None).iloc[0]
        high = effort_from_cycles(*self._cycle_with_eelv(+2.0), pga_col=None).iloc[0]
        assert high["Pes_ee"] - low["Pes_ee"] == pytest.approx(7.0, rel=1e-6)
        assert high["dPes"] == pytest.approx(low["dPes"], rel=1e-9)
        assert high["PTPes"] == pytest.approx(low["PTPes"], rel=1e-9)

    def test_equals_the_ptp_baseline(self):
        """Pes_ee is exactly the baseline PTPes and dPes are referenced to."""
        df, cycles = self._cycle_with_eelv(-5.0)
        row = effort_from_cycles(df, cycles, pga_col=None).iloc[0]
        # baseline - min(Pes) = dPes, so min(Pes) = Pes_ee - dPes
        assert row["Pes_ee"] - row["dPes"] == pytest.approx(-15.0, rel=1e-6)

    def test_nan_without_pes(self):
        """No Pes channel means no end-expiratory value."""
        df, cycles = self._cycle_with_eelv(-5.0)
        row = effort_from_cycles(
            df.drop(columns=["Pes"]), cycles, pes_col=None, pga_col=None
        ).iloc[0]
        assert math.isnan(row["Pes_ee"])


class TestEffortExpiratory:
    """Expiratory effort over [t_expi, t_next_inspi], referenced to the Pga nadir."""

    @staticmethod
    def _expiratory_cycle():
        """Pga relaxes to 2 early in expiration then ramps to 12 as abdominals contract.

        Inspiration [1, 2] holds Pga at 6. Expiration [2, 4]: Pga drops to 2 over
        the first 0.2 s (relaxation), then ramps linearly to 12 at t = 4.
        Referenced to the nadir (2), dPga_exp = 10.
        """
        t = np.arange(0, 6, 0.01)
        pga = np.full_like(t, 5.0)
        pga[(t >= 1.0) & (t <= 2.0)] = 6.0
        rel = (t > 2.0) & (t < 2.2)
        pga[rel] = 6.0 + (2.0 - 6.0) * (t[rel] - 2.0) / 0.2
        ramp = (t >= 2.2) & (t <= 4.0)
        pga[ramp] = 2.0 + (12.0 - 2.0) * (t[ramp] - 2.2) / 1.8
        pes = np.zeros_like(t)
        pes[(t >= 1.0) & (t <= 2.0)] = -10.0
        df = pd.DataFrame({"time_block": t, "Pes": pes, "Pga": pga})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [4.0]}
        )
        return df, cycles

    def test_swing_referenced_to_nadir(self):
        """Pga relaxes to 2 then peaks at 12, so dPga_exp = 10."""
        df, cycles = self._expiratory_cycle()
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["dPga_exp"] == pytest.approx(10.0, rel=2e-2)

    def test_ptp_positive_and_below_peak(self):
        """PTPga_exp is positive, and its mean over Te stays below the peak swing."""
        df, cycles = self._expiratory_cycle()
        row = effort_from_cycles(df, cycles).iloc[0]
        assert row["PTPga_exp"] > 0
        te = 4.0 - 2.0
        assert row["PTPga_exp"] / te < row["dPga_exp"]

    def test_ttiabd_matches_formula(self):
        """TTIabd = PTPga_exp / (Pga_max x Ttot)."""
        df, cycles = self._expiratory_cycle()
        row = effort_from_cycles(df, cycles, pga_max=100.0).iloc[0]
        ttot = 4.0 - 1.0
        assert row["TTIabd"] == pytest.approx(
            row["PTPga_exp"] / (100.0 * ttot), rel=1e-9
        )

    def test_ttiabd_nan_without_pga_max(self):
        """Pga_max comes from a maximal manoeuvre and cannot be inferred."""
        df, cycles = self._expiratory_cycle()
        assert math.isnan(effort_from_cycles(df, cycles).iloc[0]["TTIabd"])

    def test_ttiabd_nan_for_non_positive_pga_max(self):
        """A non-positive Pga_max is rejected rather than flipping the sign."""
        df, cycles = self._expiratory_cycle()
        assert math.isnan(effort_from_cycles(df, cycles, pga_max=0.0).iloc[0]["TTIabd"])

    def test_all_nan_without_next_inspi(self):
        """Without t_next_inspi the expiratory window is undefined."""
        df, cycles = self._expiratory_cycle()
        cycles["t_next_inspi"] = np.nan
        row = effort_from_cycles(df, cycles, pga_max=100.0).iloc[0]
        for col in ("dPga_exp", "PTPga_exp", "TTIabd"):
            assert math.isnan(row[col])

    def test_all_nan_pga_gives_nan_not_crash(self):
        """An unusable Pga channel degrades to NaN, as on the inspiratory side."""
        df, cycles = self._expiratory_cycle()
        df = df.assign(Pga=np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = effort_from_cycles(df, cycles, pga_max=100.0).iloc[0]
        for col in ("dPga_exp", "PTPga_exp", "TTIabd"):
            assert math.isnan(row[col])

    def test_nan_without_pga_column(self):
        """No Pga channel means no expiratory reference, hence no expiratory metrics."""
        df, cycles = self._expiratory_cycle()
        row = effort_from_cycles(df.drop(columns=["Pga"]), cycles, pga_col=None).iloc[0]
        for col in ("dPga_exp", "PTPga_exp"):
            assert math.isnan(row[col])

    def test_inspiratory_columns_untouched(self):
        """Adding the expiratory block leaves the inspiratory metrics unchanged."""
        df, cycles = self._expiratory_cycle()
        a = effort_from_cycles(df, cycles, pga_max=100.0).iloc[0]
        b = effort_from_cycles(df, cycles).iloc[0]
        for col in ("Pes_ee", "dPes", "dPga", "dPga_corr", "dPdi", "PTPes", "PTPdi"):
            assert a[col] == pytest.approx(b[col], rel=1e-9, nan_ok=True)


class TestEffortPesArtifactFlag:
    """pes_artifact, raised when end-expiratory Pdi is implausibly negative."""

    @staticmethod
    def _cycle(pga_rest=8.0, pes_rest=3.0):
        """Relaxed Pdi at rest is pga_rest - pes_rest; inspiration drops Pes."""
        t = np.arange(0, 4, 0.01)
        pes = np.full_like(t, pes_rest)
        pes[(t >= 1.0) & (t <= 2.0)] = pes_rest - 20.0
        pga = np.full_like(t, pga_rest)
        df = pd.DataFrame({"time_block": t, "Pes": pes, "Pga": pga})
        cycles = pd.DataFrame(
            {"n_cycle": [1], "t_inspi": [1.0], "t_expi": [2.0], "t_next_inspi": [3.0]}
        )
        return df, cycles

    def test_not_raised_on_a_clean_cycle(self):
        """Resting Pdi of +5 is physiological, so the flag stays down."""
        row = effort_from_cycles(*self._cycle()).iloc[0]
        assert row["pes_artifact"] is False or row["pes_artifact"] == False  # noqa: E712

    def test_raised_when_resting_pdi_is_negative(self):
        """A swallow lifts Pes above Pga at rest, driving Pdi negative."""
        # Pes at rest 30 against Pga 8 gives a resting Pdi of -22
        row = effort_from_cycles(*self._cycle(pes_rest=30.0)).iloc[0]
        assert row["pes_artifact"] == True  # noqa: E712

    def test_threshold_is_configurable(self):
        """pdi_ee_min moves the decision boundary."""
        df, cycles = self._cycle(pes_rest=11.0)  # resting Pdi = -3
        assert effort_from_cycles(df, cycles).iloc[0]["pes_artifact"] == False  # noqa: E712
        lenient = effort_from_cycles(df, cycles, pdi_ee_min=-1.0).iloc[0]
        assert lenient["pes_artifact"] == True  # noqa: E712

    def test_missing_rather_than_false_without_pdi(self):
        """With no Pdi the check cannot run, so the flag is missing, not False."""
        df, cycles = self._cycle()
        row = effort_from_cycles(df.drop(columns=["Pga"]), cycles, pga_col=None).iloc[0]
        assert pd.isna(row["pes_artifact"])

    def test_gastric_columns_stay_valid_on_a_flagged_cycle(self):
        """The flag concerns Pes and Pdi; the gastric columns are unaffected."""
        clean = effort_from_cycles(*self._cycle()).iloc[0]
        flagged = effort_from_cycles(*self._cycle(pes_rest=30.0)).iloc[0]
        assert flagged["pes_artifact"] == True  # noqa: E712
        assert flagged["dPga"] == pytest.approx(clean["dPga"], rel=1e-9)
        assert flagged["PTPga"] == pytest.approx(clean["PTPga"], rel=1e-9)
