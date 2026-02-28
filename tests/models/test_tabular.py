"""Tests for tabular models (Logistic Regression, XGBoost)."""

import tempfile

import numpy as np
import pytest

from src.models.tabular.logistic import LogisticStatPredictor
from src.models.tabular.xgboost_model import XGBoostStatPredictor


@pytest.fixture
def sample_tabular_data():
    """Generate sample training/validation data with consistent classes."""
    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    n_features = 15

    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    y_train = rng.integers(0, 35, size=n_train).astype(np.int64)
    X_val = rng.standard_normal((n_val, n_features)).astype(np.float32)
    # Ensure val classes are subset of train classes
    train_classes = np.unique(y_train)
    y_val = rng.choice(train_classes, size=n_val).astype(np.int64)

    return (X_train, y_train), (X_val, y_val)


class TestLogisticStatPredictor:
    def test_fit_and_predict(self, sample_tabular_data, model_config):
        """Test that logistic model trains and produces valid PMF."""
        train_data, val_data = sample_tabular_data
        model = LogisticStatPredictor(model_config)
        model.fit(train_data, val_data, {})

        X_test = val_data[0]
        pmf = model.predict_pmf(X_test)

        assert pmf.shape == (len(X_test), 61)
        # PMF should sum to ~1 for each sample
        assert np.allclose(pmf.sum(axis=1), 1.0, atol=0.01)
        # All probabilities should be non-negative
        assert np.all(pmf >= 0)

    def test_predict_proba_over(self, sample_tabular_data, model_config):
        """Test over/under probability prediction."""
        train_data, val_data = sample_tabular_data
        model = LogisticStatPredictor(model_config)
        model.fit(train_data, val_data, {})

        probs = model.predict_proba_over(val_data[0], threshold=20)
        assert probs.shape == (len(val_data[0]),)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_mean(self, sample_tabular_data, model_config):
        """Test point prediction (expected value)."""
        train_data, val_data = sample_tabular_data
        model = LogisticStatPredictor(model_config)
        model.fit(train_data, val_data, {})

        means = model.predict_mean(val_data[0])
        assert means.shape == (len(val_data[0]),)
        assert np.all(means >= 0)

    def test_save_load(self, sample_tabular_data, model_config):
        """Test model serialization."""
        train_data, val_data = sample_tabular_data
        model = LogisticStatPredictor(model_config)
        model.fit(train_data, val_data, {})

        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            model.save(f.name)
            loaded = LogisticStatPredictor.load(f.name)

            # Predictions should match
            pmf_orig = model.predict_pmf(val_data[0])
            pmf_loaded = loaded.predict_pmf(val_data[0])
            assert np.allclose(pmf_orig, pmf_loaded)

    def test_not_trained_raises(self, model_config):
        """Test that prediction before training raises error."""
        model = LogisticStatPredictor(model_config)
        with pytest.raises(RuntimeError, match="Model not trained"):
            model.predict_pmf(np.zeros((5, 10)))


class TestXGBoostStatPredictor:
    def test_fit_and_predict(self, sample_tabular_data, model_config):
        """Test that XGBoost model trains and produces valid PMF."""
        train_data, val_data = sample_tabular_data
        model = XGBoostStatPredictor(model_config)
        model.fit(train_data, val_data, {"random_seed": 42, "verbose": False})

        pmf = model.predict_pmf(val_data[0])
        assert pmf.shape[0] == len(val_data[0])
        assert pmf.shape[1] == 61
        # PMF should sum to ~1
        assert np.allclose(pmf.sum(axis=1), 1.0, atol=0.01)

    def test_predict_proba_over(self, sample_tabular_data, model_config):
        """Test over/under probability prediction."""
        train_data, val_data = sample_tabular_data
        model = XGBoostStatPredictor(model_config)
        model.fit(train_data, val_data, {"random_seed": 42, "verbose": False})

        probs = model.predict_proba_over(val_data[0], threshold=20)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_feature_importance(self, sample_tabular_data, model_config):
        """Test feature importance extraction."""
        train_data, val_data = sample_tabular_data
        model = XGBoostStatPredictor(model_config)
        model.fit(train_data, val_data, {"random_seed": 42, "verbose": False})

        feature_names = [f"feature_{i}" for i in range(train_data[0].shape[1])]
        importance = model.feature_importance(feature_names)
        assert len(importance) == len(feature_names)
        assert all(v >= 0 for v in importance.values())

    def test_save_load(self, sample_tabular_data, model_config):
        """Test model serialization."""
        train_data, val_data = sample_tabular_data
        model = XGBoostStatPredictor(model_config)
        model.fit(train_data, val_data, {"random_seed": 42, "verbose": False})

        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            model.save(f.name)
            loaded = XGBoostStatPredictor.load(f.name)

            pmf_orig = model.predict_pmf(val_data[0])
            pmf_loaded = loaded.predict_pmf(val_data[0])
            assert np.allclose(pmf_orig, pmf_loaded)
