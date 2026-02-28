"""Tests for data cleaning module."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import DataCleaner


class TestDataCleaner:
    def test_clean_box_scores_basic(self, sample_box_scores_df, cleaning_config):
        """Test basic cleaning pipeline runs without error."""
        cleaner = DataCleaner(cleaning_config)
        result = cleaner.clean_box_scores(sample_box_scores_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_deduplicate(self, sample_box_scores_df, cleaning_config):
        """Test that duplicate records are removed."""
        # Add a duplicate row
        df = pd.concat([sample_box_scores_df, sample_box_scores_df.iloc[:1]], ignore_index=True)
        cleaner = DataCleaner(cleaning_config)
        result = cleaner._deduplicate(df)

        assert len(result) == len(sample_box_scores_df)

    def test_remove_low_minute_games(self, cleaning_config):
        """Test that games with low minutes are removed."""
        df = pd.DataFrame({
            "PLAYER_ID": [1, 1, 1],
            "MIN": [30.0, 3.0, 25.0],
        })
        cleaner = DataCleaner(cleaning_config)
        result = cleaner._remove_low_minute_games(df)

        assert len(result) == 2
        assert all(result["MIN"] >= 5)

    def test_parse_minutes_string_format(self, cleaning_config):
        """Test parsing of MM:SS minute format."""
        df = pd.DataFrame({"MIN": ["32:15", "25:30", "0:45"]})
        cleaner = DataCleaner(cleaning_config)
        result = cleaner._parse_minutes(df)

        assert result["MIN"].iloc[0] == pytest.approx(32.25, abs=0.01)
        assert result["MIN"].iloc[1] == pytest.approx(25.5, abs=0.01)

    def test_parse_minutes_numeric_format(self, cleaning_config):
        """Test that numeric minutes are handled correctly."""
        df = pd.DataFrame({"MIN": [30.0, 25.5, 0.0]})
        cleaner = DataCleaner(cleaning_config)
        result = cleaner._parse_minutes(df)

        assert result["MIN"].iloc[0] == 30.0

    def test_flag_outliers(self, sample_box_scores_df, cleaning_config):
        """Test outlier flagging."""
        cleaner = DataCleaner(cleaning_config)
        result = cleaner.flag_outliers(sample_box_scores_df)

        assert "IS_OUTLIER" in result.columns
        assert result["IS_OUTLIER"].dtype == bool

    def test_handle_trades_split_season(self, cleaning_config):
        """Test trade handling with split_season strategy."""
        df = pd.DataFrame({
            "PLAYER_ID": [1, 1, 1, 2, 2],
            "TEAM_ID": [10, 10, 20, 30, 30],
            "GAME_ID": ["g1", "g2", "g3", "g4", "g5"],
            "SEASON": [2023, 2023, 2023, 2023, 2023],
        })
        cleaner = DataCleaner(cleaning_config)
        result = cleaner.handle_trades(df, "split_season")

        assert "IS_TRADED_SEASON" in result.columns
        # Player 1 played for two teams
        player1 = result[result["PLAYER_ID"] == 1]
        assert player1["IS_TRADED_SEASON"].all()
        # Player 2 stayed with one team
        player2 = result[result["PLAYER_ID"] == 2]
        assert not player2["IS_TRADED_SEASON"].any()

    def test_handle_trades_exclude(self, cleaning_config):
        """Test trade handling with exclude strategy."""
        df = pd.DataFrame({
            "PLAYER_ID": [1, 1, 1, 2, 2],
            "TEAM_ID": [10, 10, 20, 30, 30],
            "GAME_ID": ["g1", "g2", "g3", "g4", "g5"],
            "SEASON": [2023, 2023, 2023, 2023, 2023],
        })
        cleaner = DataCleaner(cleaning_config)
        result = cleaner.handle_trades(df, "exclude")

        # Player 1's games should be excluded (traded)
        assert len(result) == 2
        assert all(result["PLAYER_ID"] == 2)

    def test_validate_schema_passes(self, sample_box_scores_df, cleaning_config):
        """Test schema validation with valid data."""
        cleaner = DataCleaner(cleaning_config)
        assert cleaner.validate_schema(sample_box_scores_df)

    def test_validate_schema_fails(self, cleaning_config):
        """Test schema validation with missing columns."""
        df = pd.DataFrame({"PTS": [20], "REB": [5]})
        cleaner = DataCleaner(cleaning_config)
        assert not cleaner.validate_schema(df)

    def test_sort_chronologically(self, sample_box_scores_df, cleaning_config):
        """Test that output is sorted by player and date."""
        cleaner = DataCleaner(cleaning_config)
        result = cleaner._sort_chronologically(sample_box_scores_df)

        for _, group in result.groupby("PLAYER_ID"):
            dates = group["GAME_DATE"].values
            assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))
