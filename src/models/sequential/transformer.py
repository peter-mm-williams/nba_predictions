"""Transformer encoder model for NBA stat prediction."""

import math
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

logger = get_logger("models.transformer")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerStatPredictor(BaseStatPredictor, nn.Module):
    """Transformer encoder model for NBA stat prediction."""

    def __init__(self, config: dict):
        """Initialize the Transformer model.

        Args:
            config: Model configuration dict.
        """
        nn.Module.__init__(self)

        self.config = config
        num_players = config.get("num_players", 5000)
        num_teams = config.get("num_teams", 30)
        embed_dim = config.get("embedding_dim", 64)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 2)
        num_heads = config.get("num_heads", 4)
        ff_dim = config.get("feedforward_dim", 256)
        dropout = config.get("dropout", 0.2)
        num_seq_features = config.get("num_seq_features", 20)
        num_static_features = config.get("num_static_features", 5)
        max_value = config.get("max_value", 60)
        seq_length = config.get("sequence_length", 10)
        target_stats = config.get(
            "target_stats", ["points", "rebounds", "three_pointers_made", "free_throws_made"]
        )

        self.hidden_dim = hidden_dim

        # Embeddings
        self.player_embedding = nn.Embedding(num_players, embed_dim, padding_idx=0)
        self.team_embedding = nn.Embedding(num_teams, embed_dim, padding_idx=0)

        # Input projection
        self.feature_projection = nn.Linear(num_seq_features, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=seq_length + 10, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Static feature projection
        static_input_dim = num_static_features + 2 * embed_dim
        self.static_projection = nn.Linear(static_input_dim, hidden_dim)

        # Combine transformer output and static features
        combined_dim = hidden_dim + hidden_dim

        # Distribution heads
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
            mask: (batch, seq_len) - 1 for valid, 0 for padded
            player_ids: (batch,) optional
            team_ids: (batch,) optional

        Returns:
            Dict mapping stat names to distributions.
        """
        batch_size = sequence_features.shape[0]

        # Project and encode
        seq_proj = self.feature_projection(sequence_features)  # (B, T, H)
        seq_proj = self.positional_encoding(seq_proj)

        # Create attention mask (True = ignore)
        src_key_padding_mask = mask == 0  # (B, T)

        # Transformer encoding
        transformer_out = self.transformer_encoder(
            seq_proj, src_key_padding_mask=src_key_padding_mask
        )  # (B, T, H)

        # Pool: use mean of non-padded positions
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        masked_out = transformer_out * mask_expanded
        seq_repr = masked_out.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (B, H)

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
        static_repr = self.static_projection(static_combined)
        static_repr = self.dropout(static_repr)

        # Combine
        combined = torch.cat([seq_repr, static_repr], dim=-1)

        # Predict distributions
        distributions = {}
        for stat_name, head in self.distribution_heads.items():
            distributions[stat_name] = head(combined)

        return distributions

    def fit(self, train_data, val_data, config: dict) -> "TransformerStatPredictor":
        """Train the model (delegates to Trainer class)."""
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
    def load(cls, path: str) -> "TransformerStatPredictor":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Model loaded from %s", path)
        return model
