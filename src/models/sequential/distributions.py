"""Output distribution heads for neural network models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeBinomialDistribution:
    """Negative Binomial distribution parameterized by mu (mean) and alpha (dispersion).

    PMF: P(X=k) = Gamma(k+r) / (Gamma(k+1)*Gamma(r)) * p^r * (1-p)^k
    where r = 1/alpha, p = 1/(1 + mu*alpha)
    """

    def __init__(self, mu: torch.Tensor, alpha: torch.Tensor):
        """Initialize with mean and dispersion parameters.

        Args:
            mu: Mean parameter (positive).
            alpha: Dispersion parameter (positive). Larger = more variance.
        """
        self.mu = mu
        self.alpha = alpha
        # Convert to r, p parameterization for PyTorch
        self.r = 1.0 / alpha
        self.p = 1.0 / (1.0 + mu * alpha)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of observed values.

        Args:
            x: Observed count values.

        Returns:
            Log probabilities.
        """
        r = self.r
        p = self.p
        x = x.float()
        return (
            torch.lgamma(x + r)
            - torch.lgamma(x + 1)
            - torch.lgamma(r)
            + r * torch.log(p)
            + x * torch.log(1 - p + 1e-8)
        )

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute CDF up to value x (inclusive).

        Uses the regularized incomplete beta function relationship.
        For practical purposes, computes PMF up to x and sums.

        Args:
            x: Values at which to evaluate CDF.

        Returns:
            CDF values.
        """
        # Compute CDF by summing PMF values
        max_val = int(x.max().item()) + 1
        k_values = torch.arange(max_val, device=x.device, dtype=x.dtype)

        # Expand dimensions for broadcasting
        # mu, alpha: (batch,) -> (batch, 1)
        # k_values: (max_val,) -> (1, max_val)
        mu = self.mu.unsqueeze(-1)
        alpha = self.alpha.unsqueeze(-1)
        r = 1.0 / alpha
        p = 1.0 / (1.0 + mu * alpha)
        k = k_values.unsqueeze(0)

        log_pmf = (
            torch.lgamma(k + r)
            - torch.lgamma(k + 1)
            - torch.lgamma(r)
            + r * torch.log(p)
            + k * torch.log(1 - p + 1e-8)
        )
        pmf = torch.exp(log_pmf)

        # Create mask for k <= x
        x_expanded = x.unsqueeze(-1)
        mask = k <= x_expanded

        return (pmf * mask).sum(dim=-1)

    def mean(self) -> torch.Tensor:
        """Return the distribution mean."""
        return self.mu

    def variance(self) -> torch.Tensor:
        """Return the distribution variance."""
        return self.mu + self.alpha * self.mu**2

    def pmf_range(self, max_value: int) -> torch.Tensor:
        """Compute PMF for values 0 through max_value.

        Args:
            max_value: Maximum value (inclusive).

        Returns:
            Tensor of shape (batch_size, max_value + 1).
        """
        k_values = torch.arange(
            max_value + 1, device=self.mu.device, dtype=self.mu.dtype
        )
        mu = self.mu.unsqueeze(-1)
        alpha = self.alpha.unsqueeze(-1)
        r = 1.0 / alpha
        p = 1.0 / (1.0 + mu * alpha)
        k = k_values.unsqueeze(0)

        log_pmf = (
            torch.lgamma(k + r)
            - torch.lgamma(k + 1)
            - torch.lgamma(r)
            + r * torch.log(p)
            + k * torch.log(1 - p + 1e-8)
        )
        return torch.exp(log_pmf)


class DistributionHead(nn.Module):
    """Neural network head that outputs Negative Binomial distribution parameters."""

    def __init__(self, input_dim: int, max_value: int = 60):
        """Initialize the distribution head.

        Args:
            input_dim: Dimension of the input representation.
            max_value: Maximum stat value to model.
        """
        super().__init__()
        self.mu_layer = nn.Linear(input_dim, 1)
        self.alpha_layer = nn.Linear(input_dim, 1)
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> NegativeBinomialDistribution:
        """Compute distribution parameters from input representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            NegativeBinomialDistribution instance.
        """
        mu = F.softplus(self.mu_layer(x)).squeeze(-1)
        alpha = F.softplus(self.alpha_layer(x)).squeeze(-1) + 1e-4
        return NegativeBinomialDistribution(mu=mu, alpha=alpha)

    def predict_proba(self, x: torch.Tensor, threshold: int) -> torch.Tensor:
        """Compute P(stat > threshold).

        Args:
            x: Input tensor.
            threshold: Threshold value.

        Returns:
            Tensor of probabilities.
        """
        dist = self.forward(x)
        return 1.0 - dist.cdf(torch.tensor(threshold, device=x.device, dtype=x.dtype))


class DiscretizedHead(nn.Module):
    """Neural network head that outputs a categorical distribution over bins."""

    def __init__(self, input_dim: int, num_bins: int = 61):
        """Initialize the discretized head.

        Args:
            input_dim: Dimension of the input representation.
            num_bins: Number of output bins (0 to max_value).
        """
        super().__init__()
        self.projection = nn.Linear(input_dim, num_bins)
        self.num_bins = num_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities over bins.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Log-probabilities of shape (batch_size, num_bins).
        """
        return F.log_softmax(self.projection(x), dim=-1)

    def predict_proba(self, x: torch.Tensor, threshold: int) -> torch.Tensor:
        """Compute P(stat > threshold).

        Args:
            x: Input tensor.
            threshold: Threshold value.

        Returns:
            Tensor of probabilities.
        """
        log_probs = self.forward(x)
        probs = torch.exp(log_probs)
        # Sum probabilities for bins > threshold
        if threshold + 1 < self.num_bins:
            return probs[:, threshold + 1 :].sum(dim=-1)
        return torch.zeros(probs.shape[0], device=x.device)
