"""
Tests for resp_metrics.utils module.

Tests utility functions with known values and edge cases.
"""

import math

import numpy as np
import pytest

from resp_metrics.utils import convert_flow_unit, nearest_idx, trapz_safe


class TestNearestIdx:
    """Tests for nearest_idx function."""

    def test_exact_match(self):
        """Should return index of exact match."""
        vec = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert nearest_idx(vec, 2.0) == 2

    def test_nearest_below(self):
        """Should return index of nearest element when target is between values."""
        vec = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert nearest_idx(vec, 1.3) == 1  # Closer to 1.0

    def test_nearest_above(self):
        """Should return index of nearest element when target is above midpoint."""
        vec = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert nearest_idx(vec, 1.7) == 2  # Closer to 2.0

    def test_midpoint_returns_lower(self):
        """At exact midpoint, should return lower index (argmin behavior)."""
        vec = np.array([0.0, 1.0, 2.0])
        # 0.5 is equidistant from 0.0 and 1.0, argmin returns first
        assert nearest_idx(vec, 0.5) == 0

    def test_target_below_range(self):
        """Should return 0 when target is below all values."""
        vec = np.array([1.0, 2.0, 3.0])
        assert nearest_idx(vec, 0.0) == 0

    def test_target_above_range(self):
        """Should return last index when target is above all values."""
        vec = np.array([1.0, 2.0, 3.0])
        assert nearest_idx(vec, 10.0) == 2

    def test_single_element(self):
        """Should return 0 for single element array."""
        vec = np.array([5.0])
        assert nearest_idx(vec, 0.0) == 0
        assert nearest_idx(vec, 10.0) == 0

    def test_negative_values(self):
        """Should work with negative values."""
        vec = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        assert nearest_idx(vec, -2.3) == 1


class TestTrapzSafe:
    """Tests for trapz_safe function."""

    def test_simple_integral(self):
        """Integral of constant function y=2 from x=0 to x=3 should be 6."""
        y = np.array([2.0, 2.0, 2.0, 2.0])
        x = np.array([0.0, 1.0, 2.0, 3.0])
        result = trapz_safe(y, x)
        assert result == pytest.approx(6.0, rel=1e-10)

    def test_triangular_integral(self):
        """Integral of triangle from 0 to 2 with peak 1 at x=1 should be 1."""
        y = np.array([0.0, 1.0, 0.0])
        x = np.array([0.0, 1.0, 2.0])
        result = trapz_safe(y, x)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_linear_integral(self):
        """Integral of y=x from 0 to 2 should be 2."""
        x = np.array([0.0, 1.0, 2.0])
        y = x.copy()
        result = trapz_safe(y, x)
        assert result == pytest.approx(2.0, rel=1e-10)

    def test_single_point_returns_nan(self):
        """Should return NaN for single point."""
        y = np.array([1.0])
        x = np.array([0.0])
        result = trapz_safe(y, x)
        assert math.isnan(result)

    def test_empty_arrays_return_nan(self):
        """Should return NaN for empty arrays."""
        y = np.array([])
        x = np.array([])
        result = trapz_safe(y, x)
        assert math.isnan(result)

    def test_mismatched_sizes(self):
        """Should handle arrays with different sizes (uses shorter)."""
        y = np.array([1.0, 2.0, 3.0])
        x = np.array([0.0, 1.0])  # Shorter
        # numpy.trapz uses the shorter array
        result = trapz_safe(y[:2], x)
        assert result == pytest.approx(1.5, rel=1e-10)

    def test_negative_values(self):
        """Should work with negative values."""
        y = np.array([-1.0, -2.0, -1.0])
        x = np.array([0.0, 1.0, 2.0])
        result = trapz_safe(y, x)
        assert result == pytest.approx(-3.0, rel=1e-10)


class TestConvertFlowUnit:
    """Tests for convert_flow_unit function."""

    def test_l_min_to_l_s(self):
        """Should convert L/min to L/s by dividing by 60."""
        flow = np.array([60.0, 120.0, 30.0])
        result = convert_flow_unit(flow, "L/min")
        expected = np.array([1.0, 2.0, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_lpm_to_l_s(self):
        """Should handle 'lpm' as L/min."""
        flow = np.array([60.0])
        result = convert_flow_unit(flow, "lpm")
        assert result[0] == pytest.approx(1.0)

    def test_l_s_no_conversion(self):
        """Should not convert when already in L/s."""
        flow = np.array([1.0, 2.0, 0.5])
        result = convert_flow_unit(flow, "L/s")
        np.testing.assert_array_equal(result, flow)

    def test_l_sec_no_conversion(self):
        """Should handle 'l/sec' as L/s."""
        flow = np.array([1.0])
        result = convert_flow_unit(flow, "l/sec")
        assert result[0] == 1.0

    def test_case_insensitive(self):
        """Should be case insensitive."""
        flow = np.array([60.0])
        assert convert_flow_unit(flow, "L/MIN")[0] == pytest.approx(1.0)
        assert convert_flow_unit(flow, "l/min")[0] == pytest.approx(1.0)
        assert convert_flow_unit(flow, "L/S")[0] == 60.0

    def test_ml_s_to_l_s(self):
        """Should convert mL/s to L/s by dividing by 1000."""
        flow = np.array([500.0, 1000.0])
        result = convert_flow_unit(flow, "mL/s")
        np.testing.assert_array_almost_equal(result, [0.5, 1.0])

    def test_ml_min_to_l_s(self):
        """Should convert mL/min to L/s by dividing by 60000."""
        flow = np.array([60000.0])
        result = convert_flow_unit(flow, "mL/min")
        assert result[0] == pytest.approx(1.0)

    def test_ml_sec_variant(self):
        """Should accept 'ml/sec' as mL/s."""
        flow = np.array([1000.0])
        result = convert_flow_unit(flow, "ml/sec")
        assert result[0] == pytest.approx(1.0)

    def test_unsupported_unit_raises(self):
        """Should raise ValueError for unsupported units."""
        flow = np.array([1.0])
        with pytest.raises(ValueError, match="Unsupported flow unit"):
            convert_flow_unit(flow, "cfm")

    def test_converts_to_float(self):
        """Should convert integer arrays to float."""
        flow = np.array([60, 120, 30], dtype=np.int32)
        result = convert_flow_unit(flow, "L/min")
        assert result.dtype == np.float64
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 0.5])
