"""Logistic regression model for NBA stat prediction.

Uses a multi-class approach: discretizes the stat into bins and fits
a logistic regression model to predict bin probabilities.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseStatPredictor
from src.utils.logging import get_logger

logger = get_logger("models.logistic")


class LogisticStatPredictor(BaseStatPredictor):
    """Logistic regression model that outputs discretized probability distributions."""

    def __init__(self, config: dict):
        """Initialize the logistic regression model.

        Args:
            config: Model configuration from config.yaml.
        """
        self.config = config
        logistic_config = config.get("logistic", {})
        dist_config = config.get("distribution", {})

        self.max_value = dist_config.get("max_value", 60)
        self.num_bins = dist_config.get("num_bins", 61)
        self.solver = logistic_config.get("solver", "lbfgs")
        self.max_iter = logistic_config.get("max_iter", 1000)
        self.class_weight = logistic_config.get("class_weight", "balanced")

        self.model: LogisticRegression | None = None
        self.scaler = StandardScaler()
        self._classes: np.ndarray | None = None

    def fit(self, train_data: tuple, val_data: tuple, config: dict) -> "LogisticStatPredictor":
        """Train the logistic regression model.

        Args:
            train_data: (X_train, y_train) where y_train is integer stat values.
            val_data: (X_val, y_val) for validation.
            config: Training configuration (unused for logistic regression).

        Returns:
            self
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Clip targets to max_value
        y_train = np.clip(y_train, 0, self.max_value)
        y_val = np.clip(y_val, 0, self.max_value)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        logger.info(
            "Training logistic regression: %d samples, %d features, %d classes",
            X_train_scaled.shape[0],
            X_train_scaled.shape[1],
            len(np.unique(y_train)),
        )

        self.model = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            n_jobs=-1,
        )
        self.model.fit(X_train_scaled, y_train)
        self._classes = self.model.classes_

        # Evaluate on validation set
        X_val_scaled = self.scaler.transform(X_val)
        val_accuracy = self.model.score(X_val_scaled, y_val)
        logger.info("Validation accuracy: %.4f", val_accuracy)

        return self

    def predict_distribution(self, X: np.ndarray) -> np.ndarray:
        """Return probability distribution over stat values.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, num_bins) with probabilities.
        """
        return self.predict_pmf(X)

    def predict_pmf(self, X: np.ndarray, max_value: int | None = None) -> np.ndarray:
        """Return PMF over [0, max_value].

        Args:
            X: Feature matrix.
            max_value: Override max value (defaults to config).

        Returns:
            Array of shape (n_samples, max_value + 1).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if max_value is None:
            max_value = self.max_value

        X_scaled = self.scaler.transform(X)
        raw_probs = self.model.predict_proba(X_scaled)

        # Map class probabilities to full range [0, max_value]
        full_probs = np.zeros((X.shape[0], max_value + 1))
        for i, cls in enumerate(self._classes):
            if 0 <= cls <= max_value:
                full_probs[:, int(cls)] = raw_probs[:, i]

        # Normalize
        row_sums = full_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        full_probs = full_probs / row_sums

        return full_probs

    def predict_proba_over(self, X: np.ndarray, threshold: int) -> np.ndarray:
        """Return P(stat > threshold).

        Args:
            X: Feature matrix.
            threshold: Threshold value.

        Returns:
            Array of probabilities, shape (n_samples,).
        """
        pmf = self.predict_pmf(X)
        if threshold + 1 < pmf.shape[1]:
            return pmf[:, threshold + 1 :].sum(axis=1)
        return np.zeros(X.shape[0])

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "config": self.config,
                    "classes": self._classes,
                },
                f,
            )
        logger.info("Logistic model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LogisticStatPredictor":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        predictor = cls(data["config"])
        predictor.model = data["model"]
        predictor.scaler = data["scaler"]
        predictor._classes = data["classes"]
        logger.info("Logistic model loaded from %s", path)
        return predictor
