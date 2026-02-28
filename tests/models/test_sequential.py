"""Tests for sequential models (LSTM, Transformer)."""

import tempfile

import numpy as np
import pytest
import torch

from src.models.sequential.distributions import (
    DiscretizedHead,
    DistributionHead,
    NegativeBinomialDistribution,
)
from src.models.sequential.lstm import LSTMStatPredictor
from src.models.sequential.transformer import TransformerStatPredictor


class TestNegativeBinomialDistribution:
    def test_log_prob_valid(self):
        """Test that log probabilities are valid (negative, finite)."""
        mu = torch.tensor([10.0, 20.0, 5.0])
        alpha = torch.tensor([0.5, 0.3, 1.0])
        dist = NegativeBinomialDistribution(mu, alpha)

        x = torch.tensor([8.0, 22.0, 3.0])
        log_p = dist.log_prob(x)

        assert torch.all(torch.isfinite(log_p))
        assert torch.all(log_p <= 0)

    def test_mean(self):
        """Test distribution mean."""
        mu = torch.tensor([10.0, 20.0])
        alpha = torch.tensor([0.5, 0.3])
        dist = NegativeBinomialDistribution(mu, alpha)

        assert torch.allclose(dist.mean(), mu)

    def test_pmf_range_sums_to_one(self):
        """Test that PMF over full range sums to approximately 1."""
        mu = torch.tensor([15.0])
        alpha = torch.tensor([0.5])
        dist = NegativeBinomialDistribution(mu, alpha)

        pmf = dist.pmf_range(100)
        assert pmf.shape == (1, 101)
        assert torch.isclose(pmf.sum(dim=-1), torch.tensor([1.0]), atol=0.01)

    def test_cdf_at_large_value(self):
        """Test that CDF approaches 1 for large values."""
        mu = torch.tensor([10.0])
        alpha = torch.tensor([0.5])
        dist = NegativeBinomialDistribution(mu, alpha)

        cdf_val = dist.cdf(torch.tensor([100.0]))
        assert cdf_val.item() > 0.99


class TestDistributionHead:
    def test_forward_shape(self):
        """Test output distribution parameter shapes."""
        head = DistributionHead(input_dim=64, max_value=60)
        x = torch.randn(8, 64)
        dist = head(x)

        assert dist.mu.shape == (8,)
        assert dist.alpha.shape == (8,)
        assert torch.all(dist.mu > 0)
        assert torch.all(dist.alpha > 0)

    def test_predict_proba(self):
        """Test probability prediction output."""
        head = DistributionHead(input_dim=32)
        x = torch.randn(4, 32)
        probs = head.predict_proba(x, threshold=20)

        assert probs.shape == (4,)
        # Allow small floating point errors
        assert torch.all(probs >= -1e-6) and torch.all(probs <= 1.0 + 1e-6)


class TestDiscretizedHead:
    def test_forward_shape(self):
        """Test discretized output shape."""
        head = DiscretizedHead(input_dim=64, num_bins=61)
        x = torch.randn(8, 64)
        log_probs = head(x)

        assert log_probs.shape == (8, 61)
        # Log-softmax should sum to 0 in log space (probs sum to 1)
        probs = torch.exp(log_probs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


class TestLSTMModel:
    def test_forward_pass_shape(self, sequential_model_config):
        """Verify output shapes are correct."""
        model = LSTMStatPredictor(sequential_model_config)
        batch_size = 4
        seq_len = 5

        sequence_features = torch.randn(batch_size, seq_len, 10)
        static_features = torch.randn(batch_size, 3)
        mask = torch.ones(batch_size, seq_len)

        distributions = model(sequence_features, static_features, mask)

        assert "points" in distributions
        assert "rebounds" in distributions
        assert distributions["points"].mu.shape == (batch_size,)

    def test_distribution_valid(self, sequential_model_config):
        """Verify output distributions have positive parameters."""
        model = LSTMStatPredictor(sequential_model_config)

        sequence_features = torch.randn(4, 5, 10)
        static_features = torch.randn(4, 3)
        mask = torch.ones(4, 5)

        distributions = model(sequence_features, static_features, mask)

        for stat, dist in distributions.items():
            assert torch.all(dist.mu > 0), f"{stat} mu should be positive"
            assert torch.all(dist.alpha > 0), f"{stat} alpha should be positive"

    def test_gradient_flow(self, sequential_model_config):
        """Verify gradients flow through all parameters."""
        model = LSTMStatPredictor(sequential_model_config)

        sequence_features = torch.randn(4, 5, 10)
        static_features = torch.randn(4, 3)
        mask = torch.ones(4, 5)
        player_ids = torch.tensor([1, 2, 3, 4])
        team_ids = torch.tensor([1, 2, 1, 3])
        target = torch.tensor([20.0, 15.0, 25.0, 10.0])

        distributions = model(sequence_features, static_features, mask, player_ids, team_ids)
        # Use all distribution heads in the loss so all parameters get gradients
        loss = sum(-d.log_prob(target).mean() for d in distributions.values())
        loss.backward()

        # All parameters should have gradients (except padding_idx=0 embeddings)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_save_load(self, sequential_model_config):
        """Test model serialization."""
        model = LSTMStatPredictor(sequential_model_config)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            model.save(f.name)
            loaded = LSTMStatPredictor.load(f.name)

            # Forward pass should produce same results
            torch.manual_seed(42)
            x = torch.randn(2, 5, 10)
            s = torch.randn(2, 3)
            m = torch.ones(2, 5)

            model.eval()
            loaded.eval()
            with torch.no_grad():
                orig = model(x, s, m)
                load = loaded(x, s, m)

            assert torch.allclose(orig["points"].mu, load["points"].mu)

    def test_masking(self, sequential_model_config):
        """Test that masking works (different masks should give different results)."""
        model = LSTMStatPredictor(sequential_model_config)
        model.eval()

        x = torch.randn(2, 5, 10)
        s = torch.randn(2, 3)

        # Full mask vs partial mask
        full_mask = torch.ones(2, 5)
        partial_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)

        with torch.no_grad():
            dist_full = model(x, s, full_mask)
            dist_partial = model(x, s, partial_mask)

        # Results should differ
        assert not torch.allclose(dist_full["points"].mu, dist_partial["points"].mu)


class TestTransformerModel:
    def test_forward_pass_shape(self, sequential_model_config):
        """Verify output shapes are correct."""
        model = TransformerStatPredictor(sequential_model_config)
        batch_size = 4
        seq_len = 5

        sequence_features = torch.randn(batch_size, seq_len, 10)
        static_features = torch.randn(batch_size, 3)
        mask = torch.ones(batch_size, seq_len)

        distributions = model(sequence_features, static_features, mask)

        assert "points" in distributions
        assert distributions["points"].mu.shape == (batch_size,)

    def test_distribution_valid(self, sequential_model_config):
        """Verify output distributions have positive parameters."""
        model = TransformerStatPredictor(sequential_model_config)

        distributions = model(
            torch.randn(4, 5, 10),
            torch.randn(4, 3),
            torch.ones(4, 5),
        )

        for stat, dist in distributions.items():
            assert torch.all(dist.mu > 0)
            assert torch.all(dist.alpha > 0)

    def test_gradient_flow(self, sequential_model_config):
        """Verify gradients flow through all parameters."""
        model = TransformerStatPredictor(sequential_model_config)

        target = torch.tensor([20.0, 15.0, 25.0, 10.0])
        player_ids = torch.tensor([1, 2, 3, 4])
        team_ids = torch.tensor([1, 2, 1, 3])
        distributions = model(
            torch.randn(4, 5, 10),
            torch.randn(4, 3),
            torch.ones(4, 5),
            player_ids,
            team_ids,
        )
        # Use all distribution heads so all parameters get gradients
        loss = sum(-d.log_prob(target).mean() for d in distributions.values())
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
