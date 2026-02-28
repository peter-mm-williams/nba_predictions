"""Scoring functions for evaluating distribution predictions."""

import numpy as np

from src.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


def negative_log_likelihood(y_true: np.ndarray, pmf: np.ndarray) -> float:
    """Compute mean negative log-likelihood.

    Args:
        y_true: True stat values, shape (n_samples,).
        pmf: Predicted PMF, shape (n_samples, num_bins).

    Returns:
        Mean NLL (lower is better).
    """
    n_samples = len(y_true)
    nll = 0.0
    for i in range(n_samples):
        y = int(y_true[i])
        if 0 <= y < pmf.shape[1]:
            prob = max(pmf[i, y], 1e-10)
        else:
            prob = 1e-10
        nll -= np.log(prob)
    return nll / n_samples


def crps_discrete(y_true: np.ndarray, pmf: np.ndarray) -> float:
    """Compute mean Continuous Ranked Probability Score for discrete distributions.

    CRPS = sum_k (F(k) - 1(y <= k))^2

    Args:
        y_true: True stat values, shape (n_samples,).
        pmf: Predicted PMF, shape (n_samples, num_bins).

    Returns:
        Mean CRPS (lower is better).
    """
    n_samples, num_bins = pmf.shape
    # Compute CDF
    cdf = np.cumsum(pmf, axis=1)

    total_crps = 0.0
    for i in range(n_samples):
        y = int(y_true[i])
        indicator = np.zeros(num_bins)
        indicator[y:] = 1.0  # 1(y <= k) for each k
        total_crps += np.sum((cdf[i] - indicator) ** 2)

    return total_crps / n_samples


def calibration_error(
    y_true: np.ndarray,
    pmf: np.ndarray,
    thresholds: list[float] | None = None,
    num_bins: int = 10,
) -> dict:
    """Compute calibration error across probability thresholds.

    For each predicted probability level p, checks if the actual frequency
    of the event matches p.

    Args:
        y_true: True stat values, shape (n_samples,).
        pmf: Predicted PMF, shape (n_samples, num_bins).
        thresholds: Stat thresholds to evaluate (e.g., [10, 15, 20, 25]).
        num_bins: Number of calibration bins.

    Returns:
        Dict with calibration metrics.
    """
    if thresholds is None:
        thresholds = [5, 10, 15, 20, 25, 30]

    all_predicted = []
    all_actual = []

    for threshold in thresholds:
        if threshold + 1 < pmf.shape[1]:
            predicted_over = pmf[:, threshold + 1:].sum(axis=1)
        else:
            predicted_over = np.zeros(len(y_true))
        actual_over = (y_true > threshold).astype(float)
        all_predicted.extend(predicted_over)
        all_actual.extend(actual_over)

    all_predicted = np.array(all_predicted)
    all_actual = np.array(all_actual)

    # Bin by predicted probability
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_predicted = []
    bin_actual = []
    bin_counts = []

    for j in range(num_bins):
        low, high = bin_edges[j], bin_edges[j + 1]
        in_bin = (all_predicted >= low) & (all_predicted < high)
        if in_bin.sum() > 0:
            bin_predicted.append(all_predicted[in_bin].mean())
            bin_actual.append(all_actual[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_predicted.append((low + high) / 2)
            bin_actual.append(0.0)
            bin_counts.append(0)

    bin_predicted = np.array(bin_predicted)
    bin_actual = np.array(bin_actual)
    bin_counts = np.array(bin_counts)

    # Expected Calibration Error (weighted by bin count)
    total = bin_counts.sum()
    ece = 0.0
    if total > 0:
        ece = np.sum(bin_counts * np.abs(bin_predicted - bin_actual)) / total

    return {
        "ece": float(ece),
        "bin_predicted": bin_predicted.tolist(),
        "bin_actual": bin_actual.tolist(),
        "bin_counts": bin_counts.tolist(),
    }


def mean_absolute_error(y_true: np.ndarray, predicted_mean: np.ndarray) -> float:
    """Compute mean absolute error of point predictions.

    Args:
        y_true: True stat values.
        predicted_mean: Predicted mean values.

    Returns:
        MAE.
    """
    return float(np.mean(np.abs(y_true - predicted_mean)))


def root_mean_squared_error(y_true: np.ndarray, predicted_mean: np.ndarray) -> float:
    """Compute root mean squared error.

    Args:
        y_true: True stat values.
        predicted_mean: Predicted mean values.

    Returns:
        RMSE.
    """
    return float(np.sqrt(np.mean((y_true - predicted_mean) ** 2)))


def coverage(
    y_true: np.ndarray, pmf: np.ndarray, confidence: float = 0.9
) -> float:
    """Compute coverage: fraction of true values within the predicted CI.

    Args:
        y_true: True stat values.
        pmf: Predicted PMF.
        confidence: Confidence level (e.g., 0.9 for 90% CI).

    Returns:
        Coverage fraction.
    """
    alpha = (1 - confidence) / 2
    n_samples = len(y_true)
    covered = 0

    for i in range(n_samples):
        cdf = np.cumsum(pmf[i])
        lower = np.searchsorted(cdf, alpha)
        upper = np.searchsorted(cdf, 1 - alpha)
        y = int(y_true[i])
        if lower <= y <= upper:
            covered += 1

    return covered / n_samples


def compute_all_metrics(
    y_true: np.ndarray,
    pmf: np.ndarray,
    predicted_mean: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_true: True stat values.
        pmf: Predicted PMF, shape (n_samples, num_bins).
        predicted_mean: Optional predicted means. If None, computed from PMF.

    Returns:
        Dict of metric name -> value.
    """
    if predicted_mean is None:
        values = np.arange(pmf.shape[1])
        predicted_mean = (pmf * values[None, :]).sum(axis=1)

    cal = calibration_error(y_true, pmf)

    return {
        "nll": negative_log_likelihood(y_true, pmf),
        "crps": crps_discrete(y_true, pmf),
        "calibration_error": cal["ece"],
        "mae": mean_absolute_error(y_true, predicted_mean),
        "rmse": root_mean_squared_error(y_true, predicted_mean),
        "coverage_90": coverage(y_true, pmf, 0.90),
    }
