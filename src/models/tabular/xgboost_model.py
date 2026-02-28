"""XGBoost model for NBA stat prediction.

Uses XGBoost to predict parameters of a Negative Binomial distribution
or a multi-class classification over stat value bins.
"""

import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.models.base import BaseStatPredictor
from src.utils.logging import get_logger

logger = get_logger("models.xgboost")


class XGBoostStatPredictor(BaseStatPredictor):
    """XGBoost model that outputs discretized probability distributions."""

    def __init__(self, config: dict):
        """Initialize the XGBoost model.

        Args:
            config: Model configuration from config.yaml.
        """
        self.config = config
        xgb_config = config.get("xgboost", {})
        dist_config = config.get("distribution", {})

        self.max_value = dist_config.get("max_value", 60)
        self.num_bins = dist_config.get("num_bins", 61)

        self.n_estimators = xgb_config.get("n_estimators", 500)
        self.max_depth = xgb_config.get("max_depth", 6)
        self.learning_rate = xgb_config.get("learning_rate", 0.05)
        self.early_stopping_rounds = xgb_config.get("early_stopping_rounds", 50)

        self.model: xgb.XGBClassifier | None = None
        self._classes: np.ndarray | None = None

    def fit(self, train_data: tuple, val_data: tuple, config: dict) -> "XGBoostStatPredictor":
        """Train the XGBoost model.

        Args:
            train_data: (X_train, y_train).
            val_data: (X_val, y_val).
            config: Training configuration.

        Returns:
            self
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Clip targets
        y_train = np.clip(y_train, 0, self.max_value)
        y_val = np.clip(y_val, 0, self.max_value)

        logger.info(
            "Training XGBoost: %d samples, %d features",
            X_train.shape[0],
            X_train.shape[1],
        )

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric="mlogloss",
            early_stopping_rounds=self.early_stopping_rounds,
            tree_method="hist",
            n_jobs=-1,
            random_state=config.get("random_seed", 42),
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=config.get("verbose", False),
        )

        self._classes = self.model.classes_

        # Log best iteration
        best_iter = self.model.best_iteration
        logger.info("Training complete. Best iteration: %d", best_iter)

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
            max_value: Override max value.

        Returns:
            Array of shape (n_samples, max_value + 1).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if max_value is None:
            max_value = self.max_value

        raw_probs = self.model.predict_proba(X)

        # Map class probabilities to full range [0, max_value]
        full_probs = np.zeros((X.shape[0], max_value + 1))
        for i, cls in enumerate(self._classes):
            cls_int = int(cls)
            if 0 <= cls_int <= max_value:
                full_probs[:, cls_int] = raw_probs[:, i]

        # Normalize rows
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

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Return expected value of predicted distribution."""
        pmf = self.predict_pmf(X)
        values = np.arange(pmf.shape[1])
        return (pmf * values[None, :]).sum(axis=1)

    def feature_importance(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """Get feature importance scores.

        Args:
            feature_names: Optional list of feature names.

        Returns:
            Dict mapping feature names to importance scores.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        importance = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importance))]

        return dict(
            sorted(
                zip(feature_names, importance, strict=False),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "config": self.config, "classes": self._classes},
                f,
            )
        logger.info("XGBoost model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostStatPredictor":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        predictor = cls(data["config"])
        predictor.model = data["model"]
        predictor._classes = data["classes"]
        logger.info("XGBoost model loaded from %s", path)
        return predictor
