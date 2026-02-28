"""Sequence formatting for sequential (LSTM/Transformer) models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.features import FeatureEngineer
from src.utils.logging import get_logger

logger = get_logger("data.sequences")


@dataclass
class PlayerGameSequence:
    """A single training/inference example for sequential models."""

    player_id: str
    game_id: str
    team_id: str
    sequence_features: np.ndarray  # Shape: (seq_len, num_features)
    static_features: np.ndarray  # Shape: (num_static_features,)
    targets: dict[str, int]  # {"points": 24, "rebounds": 8, ...}
    mask: np.ndarray  # Shape: (seq_len,) - 1 for valid, 0 for padded


class SequenceBuilder:
    """Builds game sequences from feature-engineered data."""

    def __init__(self, config: dict):
        """Initialize the sequence builder.

        Args:
            config: Features section from configuration.
        """
        self.seq_length = config.get("sequence_length", 10)
        self.min_games_for_inclusion = config.get("min_games_for_sequence", 3)

    def build_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        static_cols: list[str] | None = None,
    ) -> list[PlayerGameSequence]:
        """Build sequences from feature-engineered DataFrame.

        For each game, creates a sequence of the previous `seq_length` games
        for that player. Games at the start of a career/season are padded.

        Args:
            df: Feature-engineered DataFrame sorted by (PLAYER_ID, GAME_DATE).
            feature_cols: Columns to include in sequential features.
            static_cols: Columns to include as static (non-sequential) features.

        Returns:
            List of PlayerGameSequence instances.
        """
        if static_cols is None:
            static_cols = []

        target_map = FeatureEngineer.get_target_columns()
        sequences = []

        for player_id, player_df in df.groupby("PLAYER_ID"):
            player_df = player_df.sort_values("GAME_DATE").reset_index(drop=True)

            if len(player_df) < self.min_games_for_inclusion:
                continue

            feature_matrix = player_df[feature_cols].values.astype(np.float32)
            static_matrix = (
                player_df[static_cols].values.astype(np.float32)
                if static_cols
                else np.zeros((len(player_df), 0), dtype=np.float32)
            )

            for i in range(len(player_df)):
                # Build sequence from previous games (not including current)
                start = max(0, i - self.seq_length)
                seq = feature_matrix[start:i]
                actual_len = len(seq)

                # Pad if necessary
                if actual_len < self.seq_length:
                    pad_len = self.seq_length - actual_len
                    padding = np.zeros(
                        (pad_len, feature_matrix.shape[1]), dtype=np.float32
                    )
                    seq = np.vstack([padding, seq]) if actual_len > 0 else padding
                    mask = np.array(
                        [0] * pad_len + [1] * actual_len, dtype=np.float32
                    )
                else:
                    mask = np.ones(self.seq_length, dtype=np.float32)

                # Static features for the current game context
                static = static_matrix[i]

                # Targets
                row = player_df.iloc[i]
                targets = {}
                for stat_name, col_name in target_map.items():
                    if col_name in player_df.columns:
                        val = row[col_name]
                        targets[stat_name] = int(val) if not pd.isna(val) else 0

                sequences.append(
                    PlayerGameSequence(
                        player_id=str(player_id),
                        game_id=str(row.get("GAME_ID", "")),
                        team_id=str(row.get("TEAM_ID", "")),
                        sequence_features=seq,
                        static_features=static,
                        targets=targets,
                        mask=mask,
                    )
                )

        logger.info("Built %d sequences for %d players.", len(sequences), df["PLAYER_ID"].nunique())
        return sequences

    def save_sequences(self, sequences: list[PlayerGameSequence], path: str) -> None:
        """Save sequences to disk as a compressed numpy archive."""
        seq_features = np.array([s.sequence_features for s in sequences])
        static_features = np.array([s.static_features for s in sequences])
        masks = np.array([s.mask for s in sequences])

        # Save targets as separate arrays per stat
        target_keys = list(sequences[0].targets.keys()) if sequences else []
        target_arrays = {}
        for key in target_keys:
            target_arrays[f"target_{key}"] = np.array(
                [s.targets.get(key, 0) for s in sequences]
            )

        # Save metadata
        player_ids = np.array([s.player_id for s in sequences])
        game_ids = np.array([s.game_id for s in sequences])

        np.savez_compressed(
            path,
            seq_features=seq_features,
            static_features=static_features,
            masks=masks,
            player_ids=player_ids,
            game_ids=game_ids,
            **target_arrays,
        )
        logger.info("Saved %d sequences to %s", len(sequences), path)


class NBASequenceDataset(Dataset):
    """PyTorch Dataset for NBA player game sequences."""

    def __init__(self, sequences: list[PlayerGameSequence], target_stat: str = "points"):
        """Initialize the dataset.

        Args:
            sequences: List of PlayerGameSequence instances.
            target_stat: Which stat to use as the target.
        """
        self.sequences = sequences
        self.target_stat = target_stat

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        return {
            "sequence_features": torch.from_numpy(seq.sequence_features),
            "static_features": torch.from_numpy(seq.static_features),
            "mask": torch.from_numpy(seq.mask),
            "target": torch.tensor(
                seq.targets.get(self.target_stat, 0), dtype=torch.long
            ),
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Custom collate function for DataLoader."""
        return {
            "sequence_features": torch.stack([b["sequence_features"] for b in batch]),
            "static_features": torch.stack([b["static_features"] for b in batch]),
            "mask": torch.stack([b["mask"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]),
        }
