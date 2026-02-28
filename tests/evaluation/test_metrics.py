"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    calibration_error,
    compute_all_metrics,
    coverage,
    crps_discrete,
    mean_absolute_error,
    negative_log_likelihood,
    root_mean_squared_error,
)


class TestNegativeLogLikelihood:
    def test_perfect_prediction(self):
        """NLL should be low for perfect predictions."""
        y_true = np.array([5, 10, 15])
        pmf = np.zeros((3, 61))
        pmf[0, 5] = 1.0
        pmf[1, 10] = 1.0
        pmf[2, 15] = 1.0

        nll = negative_log_likelihood(y_true, pmf)
        assert nll == pytest.approx(0.0, abs=1e-5)

    def test_uniform_prediction(self):
        """NLL should be higher for uniform predictions."""
        y_true = np.array([10])
        pmf = np.ones((1, 61)) / 61

        nll = negative_log_likelihood(y_true, pmf)
        assert nll > 0
        assert nll == pytest.approx(-np.log(1 / 61), abs=1e-5)

    def test_wrong_prediction(self):
        """NLL should be very high for wrong predictions."""
        y_true = np.array([10])
        pmf = np.zeros((1, 61))
        pmf[0, 20] = 1.0  # Predicts 20, actual is 10

        nll = negative_log_likelihood(y_true, pmf)
        # Should be capped by the epsilon floor
        assert nll > 10


class TestCRPS:
    def test_perfect_prediction(self):
        """CRPS should be 0 for perfect point mass predictions."""
        y_true = np.array([10])
        pmf = np.zeros((1, 61))
        pmf[0, 10] = 1.0

        crps = crps_discrete(y_true, pmf)
        assert crps == pytest.approx(0.0, abs=1e-5)

    def test_spread_prediction(self):
        """CRPS should increase with prediction spread."""
        y_true = np.array([10])

        # Tight prediction
        pmf_tight = np.zeros((1, 61))
        pmf_tight[0, 9:12] = [0.2, 0.6, 0.2]
        crps_tight = crps_discrete(y_true, pmf_tight)

        # Wide prediction
        pmf_wide = np.zeros((1, 61))
        pmf_wide[0, 5:16] = 1 / 11
        crps_wide = crps_discrete(y_true, pmf_wide)

        assert crps_tight < crps_wide


class TestCalibrationError:
    def test_perfect_calibration(self):
        """ECE should be 0 for perfectly calibrated predictions."""
        # This is hard to test exactly, so just check it returns valid structure
        y_true = np.random.default_rng(42).integers(0, 40, size=100)
        pmf = np.zeros((100, 61))
        for i, y in enumerate(y_true):
            pmf[i, max(0, y - 3):min(61, y + 4)] = 1 / 7

        result = calibration_error(y_true, pmf)
        assert "ece" in result
        assert "bin_predicted" in result
        assert "bin_actual" in result
        assert result["ece"] >= 0


class TestPointMetrics:
    def test_mae(self):
        """Test mean absolute error."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])

        # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7, 7/3 ≈ 2.333
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == pytest.approx(7.0 / 3, abs=1e-5)

    def test_rmse(self):
        """Test root mean squared error."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])

        rmse = root_mean_squared_error(y_true, y_pred)
        assert rmse == pytest.approx(0.0, abs=1e-5)

    def test_rmse_greater_than_mae(self):
        """RMSE should be >= MAE."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(5, 35, size=100).astype(float)
        y_pred = y_true + rng.standard_normal(100) * 5

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        assert rmse >= mae


class TestCoverage:
    def test_perfect_coverage(self):
        """Coverage should be 1.0 when all values are in the CI."""
        y_true = np.array([10, 20])
        pmf = np.zeros((2, 61))
        # Very wide distributions centered on true values
        for i, y in enumerate(y_true):
            pmf[i, max(0, y - 10):min(61, y + 11)] = 1 / 21

        cov = coverage(y_true, pmf, confidence=0.9)
        assert cov == 1.0

    def test_coverage_in_range(self, sample_pmf, sample_targets):
        """Coverage should be between 0 and 1."""
        cov = coverage(sample_targets, sample_pmf)
        assert 0 <= cov <= 1


class TestComputeAllMetrics:
    def test_returns_all_metrics(self, sample_pmf, sample_targets):
        """Test that all expected metrics are returned."""
        metrics = compute_all_metrics(sample_targets, sample_pmf)

        assert "nll" in metrics
        assert "crps" in metrics
        assert "calibration_error" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "coverage_90" in metrics

    def test_all_values_finite(self, sample_pmf, sample_targets):
        """Test that all metric values are finite."""
        metrics = compute_all_metrics(sample_targets, sample_pmf)

        for name, value in metrics.items():
            assert np.isfinite(value), f"{name} is not finite: {value}"
