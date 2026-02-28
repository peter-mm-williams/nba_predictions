"""LSTM-based sequential model for NBA stat prediction."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseStatPredictor
from src.models.sequential.distributions import (
    DistributionHead,
    NegativeBinomialDistribution,
)
from src.utils.logging import get_logger

logger = get_logger("models.lstm")


class LSTMStatPredictor(BaseStatPredictor, nn.Module):
    """LSTM-based model with player/team embeddings for stat prediction."""

    def __init__(self, config: dict):
        """Initialize the LSTM model.

        Args:
            config: Model configuration dict containing:
                - num_players: Number of unique players for embedding.
                - num_teams: Number of unique teams for embedding.
                - num_seq_features: Number of sequential features per timestep.
                - num_static_features: Number of static features.
                - embedding_dim: Dimension for player/team embeddings.
                - hidden_dim: LSTM hidden dimension.
                - num_layers: Number of LSTM layers.
                - dropout: Dropout rate.
                - bidirectional: Whether to use bidirectional LSTM.
                - max_value: Maximum stat value for distribution output.
                - target_stats: List of target stat names.
        """
        nn.Module.__init__(self)

        self.config = config
        num_players = config.get("num_players", 5000)
        num_teams = config.get("num_teams", 30)
        embed_dim = config.get("embedding_dim", 64)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.2)
        bidirectional = config.get("bidirectional", False)
        num_seq_features = config.get("num_seq_features", 20)
        num_static_features = config.get("num_static_features", 5)
        max_value = config.get("max_value", 60)
        target_stats = config.get(
            "target_stats", ["points", "rebounds", "three_pointers_made", "free_throws_made"]
        )

        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # Embeddings
        self.player_embedding = nn.Embedding(num_players, embed_dim, padding_idx=0)
        self.team_embedding = nn.Embedding(num_teams, embed_dim, padding_idx=0)

        # Sequential feature projection
        self.feature_projection = nn.Linear(num_seq_features, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Static feature projection
        static_input_dim = num_static_features + 2 * embed_dim
        self.static_projection = nn.Linear(static_input_dim, hidden_dim)

        # Combine LSTM output and static features
        combined_dim = hidden_dim * self.num_directions + hidden_dim

        # Distribution heads per target stat
        self.distribution_heads = nn.ModuleDict(
            {stat: DistributionHead(combined_dim, max_value) for stat in target_stats}
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence_features: torch.Tensor,
        static_features: torch.Tensor,
        mask: torch.Tensor,
        player_ids: torch.Tensor | None = None,
        team_ids: torch.Tensor | None = None,
    ) -> dict[str, NegativeBinomialDistribution]:
        """Forward pass.

        Args:
            sequence_features: (batch, seq_len, num_seq_features)
            static_features: (batch, num_static_features)
            mask: (batch, seq_len) - 1 for valid timesteps
            player_ids: (batch,) optional player IDs for embedding
            team_ids: (batch,) optional team IDs for embedding

        Returns:
            Dict mapping stat names to NegativeBinomialDistribution instances.
        """
        batch_size = sequence_features.shape[0]

        # Project sequential features
        seq_proj = self.feature_projection(sequence_features)  # (B, T, H)
        seq_proj = self.dropout(seq_proj)

        # Pack padded sequences for efficient LSTM processing
        lengths = mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, (hidden, _) = self.lstm(packed)

        # Use last hidden state from each direction
        if self.num_directions == 2:
            # Concatenate forward and backward final hidden states
            seq_repr = torch.cat(
                [hidden[-2], hidden[-1]], dim=-1
            )  # (B, 2*H)
        else:
            seq_repr = hidden[-1]  # (B, H)

        # Static features + embeddings
        static_parts = [static_features]
        if player_ids is not None:
            static_parts.append(self.player_embedding(player_ids))
        else:
            static_parts.append(torch.zeros(batch_size, self.player_embedding.embedding_dim, device=sequence_features.device))
        if team_ids is not None:
            static_parts.append(self.team_embedding(team_ids))
        else:
            static_parts.append(torch.zeros(batch_size, self.team_embedding.embedding_dim, device=sequence_features.device))

        static_combined = torch.cat(static_parts, dim=-1)
        static_repr = self.static_projection(static_combined)  # (B, H)
        static_repr = self.dropout(static_repr)

        # Combine sequential and static representations
        combined = torch.cat([seq_repr, static_repr], dim=-1)  # (B, combined_dim)

        # Predict distributions for each target stat
        distributions = {}
        for stat_name, head in self.distribution_heads.items():
            distributions[stat_name] = head(combined)

        return distributions

    def fit(self, train_data, val_data, config: dict) -> "LSTMStatPredictor":
        """Train the model (delegates to Trainer class)."""
        # Training is handled by src/training/trainer.py
        raise NotImplementedError("Use src.training.trainer.Trainer for training.")

    def predict_distribution(self, X: dict[str, torch.Tensor]) -> dict[str, NegativeBinomialDistribution]:
        """Return distributions for all target stats."""
        self.eval()
        with torch.no_grad():
            return self.forward(**X)

    def predict_proba_over(self, X: dict[str, torch.Tensor], threshold: int) -> dict[str, np.ndarray]:
        """Return P(stat > threshold) for each stat."""
        self.eval()
        with torch.no_grad():
            distributions = self.forward(**X)
            result = {}
            for stat, dist in distributions.items():
                cdf_val = dist.cdf(
                    torch.tensor(threshold, device=self.device_param, dtype=torch.float32)
                )
                result[stat] = (1.0 - cdf_val).cpu().numpy()
            return result

    @property
    def device_param(self) -> torch.device:
        """Get the device of model parameters."""
        return next(self.parameters()).device

    def save(self, path: str) -> None:
        """Save model state and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "config": self.config},
            path,
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LSTMStatPredictor":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Model loaded from %s", path)
        return model
