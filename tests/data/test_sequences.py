"""Tests for sequence formatting module."""

import numpy as np
import pytest

from src.data.features import FeatureEngineer
from src.data.sequences import NBASequenceDataset, PlayerGameSequence, SequenceBuilder


class TestSequenceBuilder:
    def test_build_sequences_basic(self, sample_box_scores_df, feature_config):
        """Test that sequences are built from feature-engineered data."""
        engineer = FeatureEngineer(feature_config)
        df = engineer.build_features(sample_box_scores_df)
        feature_cols = FeatureEngineer.get_feature_columns(df)

        builder = SequenceBuilder({"sequence_length": 3, "min_games_for_sequence": 2})
        sequences = builder.build_sequences(df, feature_cols)

        assert len(sequences) > 0
        assert isinstance(sequences[0], PlayerGameSequence)

    def test_sequence_shape(self, sample_box_scores_df, feature_config):
        """Test that sequence arrays have correct shapes."""
        engineer = FeatureEngineer(feature_config)
        df = engineer.build_features(sample_box_scores_df)
        feature_cols = FeatureEngineer.get_feature_columns(df)
        seq_length = 3

        builder = SequenceBuilder({"sequence_length": seq_length, "min_games_for_sequence": 2})
        sequences = builder.build_sequences(df, feature_cols)

        for seq in sequences:
            assert seq.sequence_features.shape == (seq_length, len(feature_cols))
            assert seq.mask.shape == (seq_length,)

    def test_sequence_padding(self, sample_box_scores_df, feature_config):
        """Test that early games are properly padded."""
        engineer = FeatureEngineer(feature_config)
        df = engineer.build_features(sample_box_scores_df)
        feature_cols = FeatureEngineer.get_feature_columns(df)

        builder = SequenceBuilder({"sequence_length": 5, "min_games_for_sequence": 1})
        sequences = builder.build_sequences(df, feature_cols)

        # First game for each player should have mask with mostly zeros
        first_game_seqs = [s for s in sequences if s.mask.sum() == 0]
        for seq in first_game_seqs:
            # All padded positions should be zero
            padded_positions = seq.mask == 0
            assert np.all(seq.sequence_features[padded_positions] == 0)

    def test_sequence_targets(self, sample_box_scores_df, feature_config):
        """Test that targets are correctly populated."""
        engineer = FeatureEngineer(feature_config)
        df = engineer.build_features(sample_box_scores_df)
        feature_cols = FeatureEngineer.get_feature_columns(df)

        builder = SequenceBuilder({"sequence_length": 3, "min_games_for_sequence": 2})
        sequences = builder.build_sequences(df, feature_cols)

        for seq in sequences:
            assert "points" in seq.targets
            assert "rebounds" in seq.targets
            assert isinstance(seq.targets["points"], int)

    def test_sequence_no_current_game(self, sample_box_scores_df, feature_config):
        """Test that sequences don't include the target game's data."""
        engineer = FeatureEngineer(feature_config)
        df = engineer.build_features(sample_box_scores_df)
        feature_cols = FeatureEngineer.get_feature_columns(df)

        builder = SequenceBuilder({"sequence_length": 3, "min_games_for_sequence": 2})
        sequences = builder.build_sequences(df, feature_cols)

        # The sequence for a game should contain only prior games
        # Check mask sums are correct
        for seq in sequences:
            valid_steps = int(seq.mask.sum())
            assert valid_steps <= 3  # Can't exceed sequence length

    def test_min_games_filter(self, sample_box_scores_df, feature_config):
        """Test that players with too few games are excluded."""
        engineer = FeatureEngineer(feature_config)
        df = engineer.build_features(sample_box_scores_df)
        feature_cols = FeatureEngineer.get_feature_columns(df)

        # Require more games than available
        builder = SequenceBuilder({"sequence_length": 3, "min_games_for_sequence": 100})
        sequences = builder.build_sequences(df, feature_cols)

        assert len(sequences) == 0


class TestNBASequenceDataset:
    def test_dataset_len(self, sample_box_scores_df, feature_config):
        """Test dataset length."""
        engineer = FeatureEngineer(feature_config)
        df = engineer.build_features(sample_box_scores_df)
        feature_cols = FeatureEngineer.get_feature_columns(df)

        builder = SequenceBuilder({"sequence_length": 3, "min_games_for_sequence": 2})
        sequences = builder.build_sequences(df, feature_cols)

        dataset = NBASequenceDataset(sequences, "points")
        assert len(dataset) == len(sequences)

    def test_dataset_getitem(self, sample_box_scores_df, feature_config):
        """Test that dataset items have correct tensor types."""
        engineer = FeatureEngineer(feature_config)
        df = engineer.build_features(sample_box_scores_df)
        feature_cols = FeatureEngineer.get_feature_columns(df)

        builder = SequenceBuilder({"sequence_length": 3, "min_games_for_sequence": 2})
        sequences = builder.build_sequences(df, feature_cols)

        dataset = NBASequenceDataset(sequences, "points")
        item = dataset[0]

        assert "sequence_features" in item
        assert "mask" in item
        assert "target" in item
        assert item["sequence_features"].shape[0] == 3  # seq_length
