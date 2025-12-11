"""
Tests for resp_metrics.cycles module.

Tests cycle detection from comments with various edge cases.
"""

import pytest
import pandas as pd

from resp_metrics.cycles import cycles_from_comments


class TestCyclesFromCommentsEmpty:
    """Tests for empty/null input handling."""

    def test_none_comments_returns_empty(self):
        """Should return empty DataFrame for None input."""
        result = cycles_from_comments(None, block=1)
        assert result.empty
        assert list(result.columns) == ["n_cycle", "t_inspi", "t_expi", "t_next_inspi"]

    def test_empty_df_returns_empty(self, empty_comments_df):
        """Should return empty DataFrame for empty input."""
        result = cycles_from_comments(empty_comments_df, block=1)
        assert result.empty

    def test_wrong_block_returns_empty(self, sample_comments_df):
        """Should return empty DataFrame when block not found."""
        result = cycles_from_comments(sample_comments_df, block=99)
        assert result.empty


class TestCyclesFromCommentsBasic:
    """Tests for basic cycle detection functionality."""

    def test_detects_two_complete_cycles(self, sample_comments_df):
        """Should detect 2 complete cycles from 3 INSPI markers.
        
        Comments: INSPI(0), EXPI(1), INSPI(2), EXPI(3), INSPI(4), EXPI(5)
        Cycles: 
          - Cycle 1: INSPI(0) -> EXPI(1) -> INSPI(2)
          - Cycle 2: INSPI(2) -> EXPI(3) -> INSPI(4)
        Note: INSPI(4) -> EXPI(5) is incomplete (no next INSPI)
        """
        result = cycles_from_comments(sample_comments_df, block=1)
        
        assert len(result) == 2
        assert list(result.columns) == ["n_cycle", "t_inspi", "t_expi", "t_next_inspi"]
        
        # Cycle 1
        assert result.iloc[0]["n_cycle"] == 1
        assert result.iloc[0]["t_inspi"] == pytest.approx(0.0)
        assert result.iloc[0]["t_expi"] == pytest.approx(1.0)
        assert result.iloc[0]["t_next_inspi"] == pytest.approx(2.0)
        
        # Cycle 2
        assert result.iloc[1]["n_cycle"] == 2
        assert result.iloc[1]["t_inspi"] == pytest.approx(2.0)
        assert result.iloc[1]["t_expi"] == pytest.approx(3.0)
        assert result.iloc[1]["t_next_inspi"] == pytest.approx(4.0)

    def test_single_cycle(self):
        """Should detect single complete cycle."""
        comments = pd.DataFrame({
            "block": [1, 1, 1],
            "time_block": [0.0, 1.0, 2.0],
            "Comment": ["INSPI", "EXPI", "INSPI"]
        })
        result = cycles_from_comments(comments, block=1)
        
        assert len(result) == 1
        assert result.iloc[0]["n_cycle"] == 1
        assert result.iloc[0]["t_inspi"] == 0.0
        assert result.iloc[0]["t_expi"] == 1.0
        assert result.iloc[0]["t_next_inspi"] == 2.0


class TestCyclesFromCommentsBlockFiltering:
    """Tests for block filtering functionality."""

    def test_filters_by_block(self, comments_df_block2):
        """Should only return cycles from specified block."""
        result = cycles_from_comments(comments_df_block2, block=2)
        
        assert len(result) == 1
        assert result.iloc[0]["t_inspi"] == pytest.approx(10.0)
        assert result.iloc[0]["t_expi"] == pytest.approx(11.0)
        assert result.iloc[0]["t_next_inspi"] == pytest.approx(12.0)

    def test_mixed_blocks(self, sample_comments_df, comments_df_block2):
        """Should correctly filter when multiple blocks present."""
        combined = pd.concat([sample_comments_df, comments_df_block2], ignore_index=True)
        
        result_block1 = cycles_from_comments(combined, block=1)
        result_block2 = cycles_from_comments(combined, block=2)
        
        assert len(result_block1) == 2
        assert len(result_block2) == 1


class TestCyclesFromCommentsLabels:
    """Tests for label handling."""

    def test_case_insensitive_labels(self, comments_df_lowercase):
        """Should match labels case-insensitively."""
        result = cycles_from_comments(
            comments_df_lowercase, 
            block=1, 
            insp_label="INSPI", 
            expi_label="EXPI"
        )
        
        assert len(result) == 1
        assert result.iloc[0]["t_inspi"] == 0.0
        assert result.iloc[0]["t_expi"] == 1.0

    def test_custom_labels(self):
        """Should work with custom label names."""
        comments = pd.DataFrame({
            "block": [1, 1, 1, 1],
            "time_block": [0.0, 1.0, 2.0, 3.0],
            "Comment": ["IN", "OUT", "IN", "OUT"]
        })
        result = cycles_from_comments(
            comments, 
            block=1, 
            insp_label="IN", 
            expi_label="OUT"
        )
        
        assert len(result) == 1
        assert result.iloc[0]["t_inspi"] == 0.0
        assert result.iloc[0]["t_expi"] == 1.0

    def test_labels_with_whitespace(self):
        """Should handle labels with leading/trailing whitespace."""
        comments = pd.DataFrame({
            "block": [1, 1, 1, 1],
            "time_block": [0.0, 1.0, 2.0, 3.0],
            "Comment": ["  INSPI  ", " EXPI ", "INSPI", "EXPI"]
        })
        result = cycles_from_comments(comments, block=1)
        
        assert len(result) == 1


class TestCyclesFromCommentsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_expi_skips_cycle(self, comments_df_missing_expi):
        """Should skip INSPI without following EXPI."""
        result = cycles_from_comments(comments_df_missing_expi, block=1)
        
        # Only first cycle is complete (INSPI at 0, EXPI at 1, next INSPI at 2)
        # Second INSPI at 2 has no EXPI before next INSPI at 4
        assert len(result) == 1
        assert result.iloc[0]["t_inspi"] == 0.0
        assert result.iloc[0]["t_expi"] == 1.0

    def test_no_inspi_returns_empty(self):
        """Should return empty if no INSPI markers."""
        comments = pd.DataFrame({
            "block": [1, 1],
            "time_block": [0.0, 1.0],
            "Comment": ["EXPI", "EXPI"]
        })
        result = cycles_from_comments(comments, block=1)
        assert result.empty

    def test_no_expi_returns_empty(self):
        """Should return empty if no EXPI markers."""
        comments = pd.DataFrame({
            "block": [1, 1],
            "time_block": [0.0, 1.0],
            "Comment": ["INSPI", "INSPI"]
        })
        result = cycles_from_comments(comments, block=1)
        assert result.empty

    def test_only_one_inspi_returns_empty(self):
        """Should return empty with single INSPI (needs next INSPI for cycle)."""
        comments = pd.DataFrame({
            "block": [1, 1],
            "time_block": [0.0, 1.0],
            "Comment": ["INSPI", "EXPI"]
        })
        result = cycles_from_comments(comments, block=1)
        # Single INSPI cannot form a complete cycle (needs t_next_inspi)
        assert result.empty

    def test_expi_before_inspi_ignored(self):
        """Should ignore EXPI that comes before first INSPI."""
        comments = pd.DataFrame({
            "block": [1, 1, 1, 1, 1],
            "time_block": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Comment": ["EXPI", "INSPI", "EXPI", "INSPI", "EXPI"]
        })
        result = cycles_from_comments(comments, block=1)
        
        assert len(result) == 1
        assert result.iloc[0]["t_inspi"] == 1.0  # First valid INSPI
        assert result.iloc[0]["t_expi"] == 2.0
        assert result.iloc[0]["t_next_inspi"] == 3.0
