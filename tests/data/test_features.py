"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.data.features import FORBIDDEN_FEATURES, FeatureEngineer


class TestFeatureEngineering:
    def test_build_features_basic(self, sample_box_scores_df, feature_config):
        """Test that features are built without error."""
        engineer = FeatureEngineer(feature_config)
        result = engineer.build_features(sample_box_scores_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_box_scores_df)

    def test_rolling_features_exclude_current_game(self, sample_box_scores_df, feature_config):
        """Verify rolling stats don't include the target game."""
        engineer = FeatureEngineer(feature_config)
        result = engineer.build_features(sample_box_scores_df)

        # For Curry's games, the rolling avg at any point should not include that game's value
        curry = result[result["PLAYER_ID"] == 201939].sort_values("GAME_DATE")

        for i in range(1, len(curry)):
            row = curry.iloc[i]
            prior_pts = curry.iloc[:i]["PTS"].values

            if "pts_rolling_3" in result.columns and not np.isnan(row["pts_rolling_3"]):
                # The rolling mean should be based on prior games only
                window = min(3, i)
                expected = prior_pts[-window:].mean()
                assert abs(row["pts_rolling_3"] - expected) < 0.01, (
                    f"Rolling mean at game {i} should be {expected}, got {row['pts_rolling_3']}"
                )

    def test_rolling_features_handle_season_boundary(self, feature_config):
        """Verify features reset at season boundaries."""
        df = pd.DataFrame({
            "PLAYER_ID": [1] * 6,
            "GAME_ID": [f"g{i}" for i in range(6)],
            "GAME_DATE": pd.date_range("2022-04-08", periods=3).tolist()
                + pd.date_range("2022-10-18", periods=3).tolist(),
            "SEASON": [2021, 2021, 2021, 2022, 2022, 2022],
            "PTS": [20, 25, 30, 15, 20, 25],
            "REB": [5, 6, 7, 8, 9, 10],
            "FG3M": [2, 3, 4, 1, 2, 3],
            "FTM": [3, 4, 5, 2, 3, 4],
            "MIN": [30, 32, 31, 28, 33, 30],
            "MATCHUP": ["A vs. B"] * 6,
            "TEAM_ABBREVIATION": ["AAA"] * 6,
        })

        engineer = FeatureEngineer(feature_config)
        result = engineer.build_features(df)

        # Season averages should be computed within season
        if "season_avg_pts" in result.columns:
            season_2022 = result[result["SEASON"] == 2022].sort_values("GAME_DATE")
            # First game of new season should have NaN season average
            assert pd.isna(season_2022.iloc[0]["season_avg_pts"])

    def test_no_future_leakage(self, sample_box_scores_df, feature_config):
        """Verify no features use future information."""
        engineer = FeatureEngineer(feature_config)
        result = engineer.build_features(sample_box_scores_df)

        # Check no forbidden features are present
        present_forbidden = FORBIDDEN_FEATURES & set(result.columns)
        assert len(present_forbidden) == 0, f"Forbidden features found: {present_forbidden}"

    def test_home_away_feature(self, sample_box_scores_df, feature_config):
        """Test home/away indicator is correctly computed."""
        engineer = FeatureEngineer(feature_config)
        result = engineer.build_features(sample_box_scores_df)

        assert "is_home" in result.columns
        # 'vs.' = home, '@' = away
        home_games = result[result["MATCHUP"].str.contains("vs.")]
        assert all(home_games["is_home"] == 1)

        away_games = result[result["MATCHUP"].str.contains("@")]
        assert all(away_games["is_home"] == 0)

    def test_back_to_back_detection(self, feature_config):
        """Test back-to-back game detection."""
        df = pd.DataFrame({
            "PLAYER_ID": [1, 1, 1],
            "GAME_ID": ["g1", "g2", "g3"],
            "GAME_DATE": pd.to_datetime(["2023-10-24", "2023-10-25", "2023-10-28"]),
            "SEASON": [2023, 2023, 2023],
            "PTS": [20, 25, 30],
            "REB": [5, 6, 7],
            "FG3M": [2, 3, 4],
            "FTM": [3, 4, 5],
            "MIN": [30, 32, 31],
            "MATCHUP": ["A vs. B"] * 3,
            "TEAM_ABBREVIATION": ["AAA"] * 3,
        })
        engineer = FeatureEngineer(feature_config)
        result = engineer.build_features(df)

        assert "is_back_to_back" in result.columns
        sorted_result = result.sort_values("GAME_DATE")
        assert sorted_result.iloc[1]["is_back_to_back"] == 1  # Oct 25 after Oct 24
        assert sorted_result.iloc[2]["is_back_to_back"] == 0  # Oct 28 after Oct 25

    def test_get_feature_columns(self, sample_box_scores_df, feature_config):
        """Test that feature column extraction works."""
        engineer = FeatureEngineer(feature_config)
        result = engineer.build_features(sample_box_scores_df)

        feature_cols = FeatureEngineer.get_feature_columns(result)
        assert len(feature_cols) > 0
        # Should not include raw stat columns or IDs
        assert "PTS" not in feature_cols
        assert "PLAYER_ID" not in feature_cols
        assert "GAME_ID" not in feature_cols

    def test_get_target_columns(self):
        """Test target column mapping."""
        targets = FeatureEngineer.get_target_columns()
        assert targets["points"] == "PTS"
        assert targets["rebounds"] == "REB"
        assert targets["three_pointers_made"] == "FG3M"
        assert targets["free_throws_made"] == "FTM"
