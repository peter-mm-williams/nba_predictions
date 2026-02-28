"""Pytest fixtures for NBA prop predictor tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_box_scores_df():
    """Load sample box scores as a DataFrame."""
    with open(FIXTURES_DIR / "sample_box_scores.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


@pytest.fixture
def cleaning_config():
    """Return a cleaning configuration dict."""
    return {
        "min_minutes_played": 5,
        "min_games_for_player": 2,
        "handle_trades": "split_season",
        "exclude_playoff_games": False,
    }


@pytest.fixture
def feature_config():
    """Return a feature engineering configuration dict."""
    return {
        "rolling_windows": [3, 5],
        "target_stats": ["points", "rebounds", "three_pointers_made", "free_throws_made"],
        "sequence_length": 5,
    }


@pytest.fixture
def model_config():
    """Return a model configuration dict."""
    return {
        "distribution": {
            "type": "negative_binomial",
            "max_value": 60,
            "num_bins": 61,
        },
        "logistic": {
            "solver": "lbfgs",
            "max_iter": 100,
            "class_weight": "balanced",
        },
        "xgboost": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "early_stopping_rounds": 5,
            "eval_metric": "mlogloss",
        },
    }


@pytest.fixture
def sequential_model_config():
    """Return a sequential model configuration dict."""
    return {
        "num_players": 100,
        "num_teams": 30,
        "embedding_dim": 16,
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "bidirectional": False,
        "num_seq_features": 10,
        "num_static_features": 3,
        "max_value": 60,
        "target_stats": ["points", "rebounds"],
        "sequence_length": 5,
        "num_heads": 2,
        "feedforward_dim": 64,
    }


@pytest.fixture
def sample_pmf():
    """Return a sample PMF array for evaluation tests."""
    rng = np.random.default_rng(42)
    n_samples = 50
    num_bins = 61

    # Generate somewhat realistic PMFs (peaked around 15-25 for points)
    pmf = np.zeros((n_samples, num_bins))
    for i in range(n_samples):
        center = rng.integers(10, 30)
        spread = rng.uniform(3, 8)
        x = np.arange(num_bins)
        weights = np.exp(-0.5 * ((x - center) / spread) ** 2)
        pmf[i] = weights / weights.sum()

    return pmf


@pytest.fixture
def sample_targets():
    """Return sample target values matching sample_pmf."""
    rng = np.random.default_rng(42)
    return rng.integers(5, 40, size=50)
