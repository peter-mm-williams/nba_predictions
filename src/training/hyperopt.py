"""Hyperparameter optimization using Optuna."""

from typing import Any, Callable

import optuna

from src.utils.logging import get_logger

logger = get_logger("training.hyperopt")


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization."""

    def __init__(self, config: dict):
        """Initialize the optimizer.

        Args:
            config: Hyperopt configuration section.
        """
        self.n_trials = config.get("n_trials", 50)
        self.timeout_hours = config.get("timeout_hours", 4)
        self.direction = config.get("direction", "minimize")

    def optimize(
        self,
        objective_fn: Callable[[optuna.Trial], float],
        study_name: str = "nba_prop_predictor",
    ) -> dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            objective_fn: Function that takes an Optuna trial and returns the metric.
            study_name: Name for the Optuna study.

        Returns:
            Best hyperparameters as a dict.
        """
        logger.info(
            "Starting hyperparameter optimization: %d trials, %dh timeout",
            self.n_trials,
            self.timeout_hours,
        )

        study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
        )

        study.optimize(
            objective_fn,
            n_trials=self.n_trials,
            timeout=self.timeout_hours * 3600,
        )

        logger.info("Best trial value: %.4f", study.best_trial.value)
        logger.info("Best parameters: %s", study.best_trial.params)

        return study.best_trial.params

    @staticmethod
    def suggest_sequential_params(trial: optuna.Trial) -> dict:
        """Suggest hyperparameters for sequential models.

        Args:
            trial: Optuna trial object.

        Returns:
            Dict of suggested hyperparameters.
        """
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
        }

    @staticmethod
    def suggest_xgboost_params(trial: optuna.Trial) -> dict:
        """Suggest hyperparameters for XGBoost.

        Args:
            trial: Optuna trial object.

        Returns:
            Dict of suggested hyperparameters.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
