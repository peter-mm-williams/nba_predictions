"""Integration tests for the full pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import DataCleaner
from src.data.features import FeatureEngineer
from src.data.sequences import NBASequenceDataset, SequenceBuilder
from src.evaluation.metrics import compute_all_metrics


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.integration
class TestFullPipeline:
    def test_end_to_end_small_sample(self):
        """Run full pipeline on small synthetic dataset."""
        # Load fixture data
        with open(FIXTURES_DIR / "sample_box_scores.json") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # Stage 1: Clean
        cleaner = DataCleaner({
            "min_minutes_played": 5,
            "min_games_for_player": 2,
            "handle_trades": "split_season",
        })
        cleaned = cleaner.clean_box_scores(df)
        assert len(cleaned) > 0

        # Stage 2: Features
        engineer = FeatureEngineer({
            "rolling_windows": [3],
            "sequence_length": 3,
        })
        featured = engineer.build_features(cleaned)
        feature_cols = FeatureEngineer.get_feature_columns(featured)
        assert len(feature_cols) > 0

        # Stage 3: Sequences
        builder = SequenceBuilder({"sequence_length": 3, "min_games_for_sequence": 2})
        sequences = builder.build_sequences(featured, feature_cols)
        assert len(sequences) > 0

        # Stage 4: Dataset
        dataset = NBASequenceDataset(sequences, "points")
        assert len(dataset) > 0
        item = dataset[0]
        assert item["target"].dtype == torch.long

    def test_temporal_split_no_leakage(self):
        """Verify test set has no data from training period."""
        rng = np.random.default_rng(42)

        # Create multi-season data
        dates_2022 = pd.date_range("2022-10-18", periods=20)
        dates_2023 = pd.date_range("2023-10-24", periods=20)

        df = pd.DataFrame({
            "PLAYER_ID": [1] * 40,
            "GAME_ID": [f"g{i}" for i in range(40)],
            "GAME_DATE": list(dates_2022) + list(dates_2023),
            "SEASON": [2022] * 20 + [2023] * 20,
            "PTS": rng.integers(10, 35, size=40),
            "REB": rng.integers(2, 12, size=40),
            "FG3M": rng.integers(0, 8, size=40),
            "FTM": rng.integers(0, 10, size=40),
            "MIN": rng.integers(20, 38, size=40).astype(float),
            "MATCHUP": ["A vs. B"] * 40,
            "TEAM_ID": [100] * 40,
            "TEAM_ABBREVIATION": ["AAA"] * 40,
        })

        # Split
        train_end = 2022
        train_df = df[df["SEASON"] <= train_end]
        test_df = df[df["SEASON"] > train_end]

        # Verify no overlap
        train_dates = set(train_df["GAME_DATE"])
        test_dates = set(test_df["GAME_DATE"])
        assert len(train_dates & test_dates) == 0

        # Verify test games are all after training period
        assert train_df["GAME_DATE"].max() < test_df["GAME_DATE"].min()

    def test_feature_temporal_integrity(self):
        """Verify features only use past data."""
        rng = np.random.default_rng(42)
        n_games = 15

        df = pd.DataFrame({
            "PLAYER_ID": [1] * n_games,
            "GAME_ID": [f"g{i}" for i in range(n_games)],
            "GAME_DATE": pd.date_range("2023-10-24", periods=n_games),
            "SEASON": [2023] * n_games,
            "PTS": rng.integers(10, 35, size=n_games),
            "REB": rng.integers(2, 12, size=n_games),
            "FG3M": rng.integers(0, 8, size=n_games),
            "FTM": rng.integers(0, 10, size=n_games),
            "MIN": rng.integers(20, 38, size=n_games).astype(float),
            "MATCHUP": ["A vs. B"] * n_games,
            "TEAM_ABBREVIATION": ["AAA"] * n_games,
        })

        engineer = FeatureEngineer({"rolling_windows": [3, 5], "sequence_length": 5})
        featured = engineer.build_features(df)

        # For each game, rolling features should use only prior games
        for idx in range(1, len(featured)):
            row = featured.iloc[idx]
            prior_pts = featured.iloc[:idx]["PTS"].values

            if "pts_rolling_3" in featured.columns and not pd.isna(row["pts_rolling_3"]):
                window = min(3, idx)
                expected = prior_pts[-window:].mean()
                assert abs(row["pts_rolling_3"] - expected) < 0.01


# Need torch import at module level for the integration test
import torch
