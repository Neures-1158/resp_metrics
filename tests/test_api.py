"""
Tests for resp_metrics.api module.

Tests the high-level compute_from_labchart API.
Uses mocking to avoid dependency on labchart_parser files.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from resp_metrics.api import compute_from_labchart


class MockLabChartFile:
    """Mock LabChartFile for testing without file dependency."""
    
    def __init__(self, df_block, comments, metadata=None):
        self._df_block = df_block
        self._comments = comments
        self._metadata = metadata or {"title": "Test", "date": "2024-01-01"}
    
    @classmethod
    def from_file(cls, path):
        """Mock class method."""
        # Create test data based on path
        t = np.linspace(0, 10, 1000)
        flow = -0.5 * np.sin(2 * np.pi * t / 4)
        
        df = pd.DataFrame({
            "time_block": t,
            "Flow": flow * 60,  # L/min
            "Paw": np.ones(1000) * 5.0,
            "Pressure": np.ones(1000) * 15.0
        })
        
        comments = pd.DataFrame({
            "block": [1, 1, 1, 1, 1, 1],
            "time_block": [0.0, 1.0, 4.0, 5.0, 8.0, 9.0],
            "Comment": ["INSPI", "EXPI", "INSPI", "EXPI", "INSPI", "EXPI"]
        })
        
        return cls(df, comments)
    
    def get_block_df(self, block):
        return self._df_block
    
    @property
    def comments(self):
        return self._comments
    
    @property
    def metadata(self):
        return self._metadata


class TestComputeFromLabchartReturnStructure:
    """Tests for return structure of compute_from_labchart."""

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_returns_dict_with_required_keys(self):
        """Should return dict with meta, cycles, ventilatory, ventilator."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        assert isinstance(result, dict)
        assert "meta" in result
        assert "cycles" in result
        assert "ventilatory" in result
        assert "ventilator" in result

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_meta_is_dict(self):
        """meta should be a dictionary."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        assert isinstance(result["meta"], dict)

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_cycles_is_dataframe(self):
        """cycles should be a DataFrame."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        assert isinstance(result["cycles"], pd.DataFrame)

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_ventilatory_is_dataframe(self):
        """ventilatory should be a DataFrame."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        assert isinstance(result["ventilatory"], pd.DataFrame)


class TestComputeFromLabchartSpontaneous:
    """Tests for spontaneous breathing mode."""

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_spontaneous_mode_no_ventilator(self):
        """In spontaneous mode, ventilator should be None."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min",
            mechanically_ventilated=False
        )
        
        assert result["ventilator"] is None

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_spontaneous_ventilatory_columns(self):
        """Ventilatory DataFrame should have expected columns."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min",
            mechanically_ventilated=False
        )
        
        expected_cols = ["n_cycle", "t_inspi", "t_expi", "Ti", "Te", "Ttot", 
                        "BF", "VT", "VE", "PIF", "PEF", "IE", "WOB", "PTP"]
        for col in expected_cols:
            assert col in result["ventilatory"].columns


class TestComputeFromLabchartMechanical:
    """Tests for mechanical ventilation mode."""

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_mechanical_mode_with_pressure(self):
        """In mechanical mode with pressure, ventilator should not be None."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min",
            pressure_col="Pressure",
            mechanically_ventilated=True
        )
        
        assert result["ventilator"] is not None
        assert isinstance(result["ventilator"], pd.DataFrame)

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_mechanical_ventilator_columns(self):
        """Ventilator DataFrame should have mechanical columns."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min",
            pressure_col="Pressure",
            mechanically_ventilated=True
        )
        
        if result["ventilator"] is not None:
            expected_cols = ["n_cycle", "t_inspi", "t_expi", "PEEP", "Ppeak", 
                           "Pplat", "dP", "Cstat", "R", "MAP"]
            for col in expected_cols:
                assert col in result["ventilator"].columns

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_mechanical_without_pressure_no_ventilator(self):
        """Mechanical mode without pressure_col should not compute ventilator."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min",
            pressure_col=None,  # No pressure
            mechanically_ventilated=True
        )
        
        # Should fall back to spontaneous since no pressure
        assert result["ventilator"] is None


class TestComputeFromLabchartParameters:
    """Tests for parameter handling."""

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_flow_unit_passed_to_spontaneous(self):
        """flow_unit should be passed to ventilatory_from_cycles."""
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/s",  # Different unit
            mechanically_ventilated=False
        )
        
        # Should complete without error
        assert result["ventilatory"] is not None

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_block_selection(self):
        """Should use specified block."""
        result = compute_from_labchart(
            "test.txt",
            block=1,
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        # Should complete for block 1
        assert len(result["cycles"]) > 0

    @patch('resp_metrics.api.LabChartFile', MockLabChartFile)
    def test_custom_labels(self):
        """Should respect custom INSPI/EXPI labels."""
        # This would require a mock with different labels
        # For now, just verify the parameter is accepted
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min",
            insp_label="INSPI",
            expi_label="EXPI"
        )
        
        assert result is not None


class TestComputeFromLabchartPesPropagation:
    """Tests for pes_col parameter propagation."""

    @patch('resp_metrics.api.LabChartFile')
    def test_pes_col_passed_to_ventilatory(self, mock_lc_class):
        """pes_col should be passed to ventilatory_from_cycles."""
        # Create mock with Pes column
        t = np.linspace(0, 10, 1000)
        df = pd.DataFrame({
            "time_block": t,
            "Flow": np.sin(2 * np.pi * t / 4) * -30,
            "Paw": np.ones(1000) * 5.0,
            "Pes": np.ones(1000) * -5.0
        })
        comments = pd.DataFrame({
            "block": [1, 1, 1, 1],
            "time_block": [0.0, 1.0, 4.0, 5.0],
            "Comment": ["INSPI", "EXPI", "INSPI", "EXPI"]
        })
        
        mock_lc = MagicMock()
        mock_lc.get_block_df.return_value = df
        mock_lc.comments = comments
        mock_lc.metadata = {}
        mock_lc_class.from_file.return_value = mock_lc
        
        result = compute_from_labchart(
            "test.txt",
            flow_col="Flow",
            flow_unit="L/min",
            pes_col="Pes",
            mechanically_ventilated=False
        )
        
        # WOB should be calculated when Pes is provided
        # (actual value depends on signal)
        assert "WOB" in result["ventilatory"].columns


class MockLabChartFileMultiBlock:
    """Mock LabChartFile with multiple blocks for testing."""
    
    def __init__(self):
        t = np.linspace(0, 10, 1000)
        flow = -0.5 * np.sin(2 * np.pi * t / 4)
        
        # Create data for 3 blocks
        self._data = pd.DataFrame({
            "block": [1] * 1000 + [2] * 1000 + [3] * 1000,
            "time_block": list(t) * 3,
            "Flow": list(flow * 60) * 3,
            "Paw": [5.0] * 3000,
            "Pressure": [15.0] * 3000
        })
        
        self._comments = pd.DataFrame({
            "block": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time_block": [0.0, 1.0, 4.0, 5.0] * 3,
            "Comment": ["INSPI", "EXPI", "INSPI", "EXPI"] * 3
        })
        
        self._metadata = {"title": "MultiBlock Test", "date": "2024-01-01"}
    
    @classmethod
    def from_file(cls, path):
        return cls()
    
    def get_block_df(self, block):
        t = np.linspace(0, 10, 1000)
        flow = -0.5 * np.sin(2 * np.pi * t / 4)
        return pd.DataFrame({
            "time_block": t,
            "Flow": flow * 60,
            "Paw": np.ones(1000) * 5.0,
            "Pressure": np.ones(1000) * 15.0
        })
    
    @property
    def comments(self):
        return self._comments
    
    @property
    def metadata(self):
        return self._metadata
    
    @property
    def blocks(self):
        """Return list of available blocks."""
        return [1, 2, 3]
    
    @property
    def data(self):
        return self._data


class TestComputeFromLabchartMultiBlock:
    """Tests for multi-block functionality."""

    @patch('resp_metrics.api.LabChartFile', MockLabChartFileMultiBlock)
    def test_single_block_unchanged(self):
        """Verify that behavior with block=1 (int) hasn't changed."""
        result = compute_from_labchart(
            "test.txt",
            block=1,
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        # Should return DataFrames directly, not dicts
        assert isinstance(result, dict)
        assert isinstance(result["meta"], dict)
        assert isinstance(result["cycles"], pd.DataFrame)
        assert isinstance(result["ventilatory"], pd.DataFrame)
        # ventilator should be None or DataFrame, not dict
        assert result["ventilator"] is None or isinstance(result["ventilator"], pd.DataFrame)

    @patch('resp_metrics.api.LabChartFile', MockLabChartFileMultiBlock)
    def test_multiple_blocks_list(self):
        """Test with block=[1, 2] returns dict of DataFrames."""
        result = compute_from_labchart(
            "test.txt",
            block=[1, 2],
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        # Should return dicts keyed by block number
        assert isinstance(result, dict)
        assert isinstance(result["meta"], dict)
        
        # cycles should be a dict with keys 1 and 2
        assert isinstance(result["cycles"], dict)
        assert 1 in result["cycles"]
        assert 2 in result["cycles"]
        assert isinstance(result["cycles"][1], pd.DataFrame)
        assert isinstance(result["cycles"][2], pd.DataFrame)
        
        # ventilatory should be a dict with keys 1 and 2
        assert isinstance(result["ventilatory"], dict)
        assert 1 in result["ventilatory"]
        assert 2 in result["ventilatory"]
        assert isinstance(result["ventilatory"][1], pd.DataFrame)
        assert isinstance(result["ventilatory"][2], pd.DataFrame)
        
        # ventilator should be a dict with keys 1 and 2
        assert isinstance(result["ventilator"], dict)
        assert 1 in result["ventilator"]
        assert 2 in result["ventilator"]

    @patch('resp_metrics.api.LabChartFile', MockLabChartFileMultiBlock)
    def test_all_blocks_none(self):
        """Test with block=None returns all blocks."""
        result = compute_from_labchart(
            "test.txt",
            block=None,
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        # Should return dicts keyed by all available block numbers
        assert isinstance(result, dict)
        assert isinstance(result["meta"], dict)
        
        # cycles should be a dict with keys 1, 2, and 3
        assert isinstance(result["cycles"], dict)
        assert 1 in result["cycles"]
        assert 2 in result["cycles"]
        assert 3 in result["cycles"]
        
        # ventilatory should be a dict with keys 1, 2, and 3
        assert isinstance(result["ventilatory"], dict)
        assert 1 in result["ventilatory"]
        assert 2 in result["ventilatory"]
        assert 3 in result["ventilatory"]
        
        # ventilator should be a dict with keys 1, 2, and 3
        assert isinstance(result["ventilator"], dict)
        assert 1 in result["ventilator"]
        assert 2 in result["ventilator"]
        assert 3 in result["ventilator"]

    @patch('resp_metrics.api.LabChartFile', MockLabChartFileMultiBlock)
    def test_multiple_blocks_mechanical(self):
        """Test multi-block with mechanical ventilation mode."""
        result = compute_from_labchart(
            "test.txt",
            block=[1, 2],
            flow_col="Flow",
            flow_unit="L/min",
            pressure_col="Pressure",
            mechanically_ventilated=True
        )
        
        # ventilator should be a dict with DataFrames
        assert isinstance(result["ventilator"], dict)
        assert 1 in result["ventilator"]
        assert 2 in result["ventilator"]
        # Each should be a DataFrame (not None) since mechanical mode is on
        assert result["ventilator"][1] is not None
        assert result["ventilator"][2] is not None

    @patch('resp_metrics.api.LabChartFile', MockLabChartFileMultiBlock)
    def test_single_block_in_list(self):
        """Test with block=[1] (list with single element) returns dict format."""
        result = compute_from_labchart(
            "test.txt",
            block=[1],
            flow_col="Flow",
            flow_unit="L/min"
        )
        
        # Even with single element list, should return dict format
        assert isinstance(result["cycles"], dict)
        assert 1 in result["cycles"]
        assert isinstance(result["cycles"][1], pd.DataFrame)
