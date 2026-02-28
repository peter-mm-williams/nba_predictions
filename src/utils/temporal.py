"""Time-series utilities for NBA data processing."""

from datetime import datetime

import numpy as np
import pandas as pd


def nba_season_string(year: int) -> str:
    """Convert a season start year to NBA API season string.

    Example: 2023 -> '2023-24'
    """
    next_year = str(year + 1)[-2:]
    return f"{year}-{next_year}"


def parse_game_date(date_str: str) -> datetime:
    """Parse various NBA game date formats to datetime."""
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date: {date_str}")


def calculate_days_rest(game_dates: pd.Series) -> pd.Series:
    """Calculate days of rest between consecutive games.

    Args:
        game_dates: Series of game dates, sorted chronologically.

    Returns:
        Series of rest days (NaN for first game).
    """
    dates = pd.to_datetime(game_dates)
    return dates.diff().dt.days


def detect_back_to_back(game_dates: pd.Series) -> pd.Series:
    """Detect back-to-back games (played on consecutive days).

    Returns:
        Boolean series indicating back-to-back games.
    """
    rest_days = calculate_days_rest(game_dates)
    return rest_days == 1


def compute_rolling_stat(
    values: pd.Series,
    window: int,
    min_periods: int = 1,
    stat: str = "mean",
) -> pd.Series:
    """Compute rolling statistic, shifted to exclude current game.

    The shift ensures no data leakage - the rolling stat for game N
    uses only games 1..N-1.

    Args:
        values: Series of stat values, ordered chronologically.
        window: Number of prior games to include.
        min_periods: Minimum observations for a valid result.
        stat: Aggregation function ('mean', 'std', 'sum', 'median').

    Returns:
        Shifted rolling statistic series.
    """
    roller = values.shift(1).rolling(window=window, min_periods=min_periods)
    agg_func = getattr(roller, stat)
    return agg_func()


def compute_trend_slope(values: pd.Series, window: int) -> pd.Series:
    """Compute the slope of a linear trend over a rolling window.

    Uses ordinary least squares on the shifted (lag-1) values.

    Args:
        values: Series of stat values.
        window: Window size for the trend.

    Returns:
        Series of trend slopes (positive = increasing).
    """
    shifted = values.shift(1)

    def _slope(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            return np.nan
        x = np.arange(len(arr))
        coeffs = np.polyfit(x, arr, 1)
        return coeffs[0]

    return shifted.rolling(window=window, min_periods=2).apply(_slope, raw=True)


def season_game_number(game_dates: pd.Series, season_col: pd.Series) -> pd.Series:
    """Compute the game number within each season for a player.

    Args:
        game_dates: Series of game dates.
        season_col: Series indicating the season for each game.

    Returns:
        Series of 1-indexed game numbers within each season.
    """
    df = pd.DataFrame({"date": game_dates, "season": season_col})
    return df.groupby("season").cumcount() + 1
