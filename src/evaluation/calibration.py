"""Calibration analysis for distribution predictions."""

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("evaluation.calibration")

# Analysis slice definitions
ANALYSIS_SLICES = {
    "position": ["PG", "SG", "SF", "PF", "C"],
    "usage_tier": ["high", "medium", "low"],
    "home_away": ["home", "away"],
    "rest_days": ["back_to_back", "1_day", "2_plus_days"],
    "opponent_strength": ["top_10_defense", "middle", "bottom_10_defense"],
    "minutes_tier": ["starter", "rotation", "bench"],
}


class CalibrationAnalyzer:
    """Analyze calibration of probability predictions across slices."""

    def __init__(self, num_bins: int = 10):
        """Initialize the analyzer.

        Args:
            num_bins: Number of bins for calibration plots.
        """
        self.num_bins = num_bins

    def reliability_diagram_data(
        self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray
    ) -> dict:
        """Compute data for a reliability diagram.

        Args:
            predicted_probs: Predicted probabilities for the event.
            actual_outcomes: Binary outcomes (0 or 1).

        Returns:
            Dict with bin_midpoints, bin_frequencies, bin_counts, ece.
        """
        bin_edges = np.linspace(0, 1, self.num_bins + 1)
        bin_midpoints = []
        bin_frequencies = []
        bin_counts = []

        for i in range(self.num_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            in_bin = (predicted_probs >= low) & (predicted_probs < high)
            count = in_bin.sum()
            bin_counts.append(int(count))

            if count > 0:
                bin_midpoints.append(float(predicted_probs[in_bin].mean()))
                bin_frequencies.append(float(actual_outcomes[in_bin].mean()))
            else:
                bin_midpoints.append(float((low + high) / 2))
                bin_frequencies.append(0.0)

        total = sum(bin_counts)
        ece = 0.0
        if total > 0:
            ece = sum(
                c * abs(p - f)
                for c, p, f in zip(bin_counts, bin_midpoints, bin_frequencies, strict=False)
            ) / total

        return {
            "bin_midpoints": bin_midpoints,
            "bin_frequencies": bin_frequencies,
            "bin_counts": bin_counts,
            "ece": float(ece),
        }

    def analyze_by_slice(
        self,
        df: pd.DataFrame,
        predicted_col: str,
        actual_col: str,
        slice_col: str,
    ) -> dict[str, dict]:
        """Compute calibration metrics grouped by a slice column.

        Args:
            df: DataFrame containing predictions and metadata.
            predicted_col: Column name for predicted probabilities.
            actual_col: Column name for actual binary outcomes.
            slice_col: Column name to group by.

        Returns:
            Dict mapping slice values to calibration dicts.
        """
        results = {}
        for slice_val, group in df.groupby(slice_col):
            if len(group) < 10:
                continue
            results[str(slice_val)] = self.reliability_diagram_data(
                group[predicted_col].values,
                group[actual_col].values,
            )
        return results

    def full_slice_analysis(
        self,
        df: pd.DataFrame,
        predicted_col: str,
        actual_col: str,
        slice_columns: list[str],
    ) -> dict[str, dict[str, dict]]:
        """Run calibration analysis across all slice dimensions.

        Args:
            df: DataFrame with predictions and metadata.
            predicted_col: Predicted probability column.
            actual_col: Actual outcome column.
            slice_columns: List of column names to slice by.

        Returns:
            Nested dict: slice_dim -> slice_value -> calibration_data.
        """
        results = {}
        for col in slice_columns:
            if col in df.columns:
                results[col] = self.analyze_by_slice(
                    df, predicted_col, actual_col, col
                )
            else:
                logger.warning("Slice column '%s' not found in DataFrame.", col)
        return results
