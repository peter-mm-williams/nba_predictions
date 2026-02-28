"""Abstract base class for all prediction models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseStatPredictor(ABC):
    """Abstract base for all NBA stat prediction models.

    All models must output probability distributions over counting stats,
    not just point predictions.
    """

    @abstractmethod
    def fit(self, train_data: Any, val_data: Any, config: dict) -> "BaseStatPredictor":
        """Train the model.

        Args:
            train_data: Training dataset (format depends on model type).
            val_data: Validation dataset.
            config: Training configuration.

        Returns:
            self, for method chaining.
        """

    @abstractmethod
    def predict_distribution(self, X: Any) -> Any:
        """Return full probability distribution over stat values.

        Args:
            X: Input features (format depends on model type).

        Returns:
            Distribution object or array of probabilities.
        """

    @abstractmethod
    def predict_proba_over(self, X: Any, threshold: int) -> np.ndarray:
        """Return P(stat > threshold) for each sample.

        Args:
            X: Input features.
            threshold: The threshold value.

        Returns:
            Array of probabilities, shape (n_samples,).
        """

    def predict_mean(self, X: Any) -> np.ndarray:
        """Return the mean (expected value) of the predicted distribution.

        Default implementation uses predict_pmf if available.

        Args:
            X: Input features.

        Returns:
            Array of predicted means, shape (n_samples,).
        """
        pmf = self.predict_pmf(X)
        values = np.arange(pmf.shape[1])
        return (pmf * values[None, :]).sum(axis=1)

    def predict_pmf(self, X: Any, max_value: int = 60) -> np.ndarray:
        """Return probability mass function over [0, max_value].

        Args:
            X: Input features.
            max_value: Maximum stat value to model.

        Returns:
            Array of shape (n_samples, max_value + 1).
        """
        raise NotImplementedError("Subclass must implement predict_pmf or predict_mean")

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save the model.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseStatPredictor":
        """Load model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            Loaded model instance.
        """
