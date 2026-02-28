"""Visualization utilities for model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.io import ensure_dir
from src.utils.logging import get_logger

logger = get_logger("evaluation.plots")


def plot_calibration(
    calibration_data: dict,
    title: str = "Calibration Plot",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a reliability diagram.

    Args:
        calibration_data: Dict from CalibrationAnalyzer with bin_midpoints, bin_frequencies.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    midpoints = calibration_data["bin_midpoints"]
    frequencies = calibration_data["bin_frequencies"]
    counts = calibration_data["bin_counts"]
    ece = calibration_data.get("ece", 0)

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(midpoints, frequencies, width=0.08, alpha=0.7, label="Model")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(f"{title} (ECE: {ece:.4f})")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predictions
    ax2.bar(midpoints, counts, width=0.08, alpha=0.7, color="gray")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Calibration plot saved to %s", save_path)

    return fig


def plot_distribution_comparison(
    predicted_pmf: np.ndarray,
    actual_value: int,
    max_display: int = 50,
    title: str = "Predicted Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a predicted PMF with the actual value marked.

    Args:
        predicted_pmf: PMF array, shape (num_bins,).
        actual_value: The true stat value.
        max_display: Maximum x-axis value to show.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    display_range = min(max_display + 1, len(predicted_pmf))
    x = np.arange(display_range)
    ax.bar(x, predicted_pmf[:display_range], alpha=0.7, color="steelblue", label="Predicted PMF")
    ax.axvline(actual_value, color="red", linestyle="--", linewidth=2, label=f"Actual: {actual_value}")

    # Mark the predicted mean
    values = np.arange(len(predicted_pmf))
    pred_mean = (predicted_pmf * values).sum()
    ax.axvline(pred_mean, color="orange", linestyle=":", linewidth=2, label=f"Pred Mean: {pred_mean:.1f}")

    ax.set_xlabel("Stat Value")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    metrics: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot comparison of metrics across models.

    Args:
        metrics: Dict mapping model name -> metric name -> value.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    model_names = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_names, strict=False):
        values = [metrics[m][metric] for m in model_names]
        bars = ax.bar(model_names, values, alpha=0.8)
        ax.set_title(metric.upper())
        ax.set_ylabel("Value")
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Model comparison plot saved to %s", save_path)

    return fig


def plot_slice_analysis(
    slice_results: dict[str, dict],
    metric: str = "ece",
    title: str = "Calibration by Slice",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot calibration metrics across analysis slices.

    Args:
        slice_results: Dict from CalibrationAnalyzer.analyze_by_slice.
        metric: Which metric to plot from each slice result.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    slices = list(slice_results.keys())
    values = [slice_results[s].get(metric, 0) for s in slices]

    fig, ax = plt.subplots(figsize=(max(8, len(slices) * 1.5), 5))
    bars = ax.bar(slices, values, alpha=0.8, color="teal")
    ax.set_title(title)
    ax.set_ylabel(metric.upper())
    ax.set_xticklabels(slices, rotation=45, ha="right")

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        history: Dict with 'train_loss' and 'val_loss' lists.
        title: Plot title.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (NLL)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
