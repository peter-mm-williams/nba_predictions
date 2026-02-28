"""Feature engineering for NBA player prop prediction.

Critical constraint: All features must be computable using only data available
T-30 minutes before game time (no in-game or post-game data leakage).
"""

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.temporal import (
    calculate_days_rest,
    compute_rolling_stat,
    compute_trend_slope,
    detect_back_to_back,
    season_game_number,
)

logger = get_logger("data.features")


# Feature safety registry
SAFE_FEATURES = {
    # Rolling statistics (prior games only)
    "pts_rolling_3", "pts_rolling_5", "pts_rolling_10", "pts_rolling_20",
    "reb_rolling_3", "reb_rolling_5", "reb_rolling_10", "reb_rolling_20",
    "fg3m_rolling_3", "fg3m_rolling_5", "fg3m_rolling_10", "fg3m_rolling_20",
    "ftm_rolling_3", "ftm_rolling_5", "ftm_rolling_10", "ftm_rolling_20",
    "pts_std_10", "reb_std_10", "fg3m_std_10", "ftm_std_10",
    "pts_trend_5", "reb_trend_5", "fg3m_trend_5", "ftm_trend_5",
    # Rest and schedule
    "days_rest", "is_back_to_back", "is_home", "season_game_num",
    # Opponent (season-to-date stats only)
    "opp_def_rating_std", "opp_pace_std", "opp_pts_allowed_position",
    # Player baseline (prior data only)
    "season_avg_pts", "season_avg_reb", "season_avg_fg3m", "season_avg_ftm",
    "career_avg_pts", "career_avg_reb", "career_avg_fg3m", "career_avg_ftm",
    "usage_rate_season", "minutes_avg_10",
    # Team context (prior data only)
    "team_pace_std", "team_off_rating_std", "key_teammate_out",
}

FORBIDDEN_FEATURES = {
    "actual_minutes_played",
    "game_outcome",
    "in_game_stats",
    "post_game_adjustments",
}

# Stat columns and their feature prefixes
STAT_MAP = {
    "PTS": "pts",
    "REB": "reb",
    "FG3M": "fg3m",
    "FTM": "ftm",
}


class FeatureEngineer:
    """Builds features from cleaned box score data."""

    def __init__(self, config: dict):
        """Initialize the feature engineer.

        Args:
            config: Features section from configuration.
        """
        self.rolling_windows = config.get("rolling_windows", [3, 5, 10, 20])
        self.sequence_length = config.get("sequence_length", 10)
        self.target_stats = config.get(
            "target_stats",
            ["points", "rebounds", "three_pointers_made", "free_throws_made"],
        )

    def build_features(self, df: pd.DataFrame, team_stats: pd.DataFrame | None = None) -> pd.DataFrame:
        """Build all features from cleaned box score data.

        Args:
            df: Cleaned box score DataFrame, sorted by (PLAYER_ID, GAME_DATE).
            team_stats: Optional team-level stats for opponent/team features.

        Returns:
            DataFrame with all engineered features appended.
        """
        logger.info("Building features for %d records...", len(df))

        df = df.copy()
        df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

        df = self._build_rolling_features(df)
        df = self._build_trend_features(df)
        df = self._build_schedule_features(df)
        df = self._build_season_averages(df)
        df = self._build_career_averages(df)
        df = self._build_minutes_features(df)

        if team_stats is not None:
            df = self._build_opponent_features(df, team_stats)
            df = self._build_team_features(df, team_stats)

        df = self._build_home_away(df)

        self._validate_no_leakage(df)

        logger.info("Feature engineering complete. %d features built.", len(self.get_feature_columns(df)))
        return df

    def _build_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build rolling mean and std features for target stats."""
        for stat_col, prefix in STAT_MAP.items():
            if stat_col not in df.columns:
                continue
            for window in self.rolling_windows:
                col_name = f"{prefix}_rolling_{window}"
                df[col_name] = df.groupby("PLAYER_ID")[stat_col].transform(
                    lambda s: compute_rolling_stat(s, window, stat="mean")
                )

            # Rolling std with window=10
            std_col = f"{prefix}_std_10"
            df[std_col] = df.groupby("PLAYER_ID")[stat_col].transform(
                lambda s: compute_rolling_stat(s, 10, stat="std")
            )

        return df

    def _build_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build trend (slope) features for target stats."""
        for stat_col, prefix in STAT_MAP.items():
            if stat_col not in df.columns:
                continue
            trend_col = f"{prefix}_trend_5"
            df[trend_col] = df.groupby("PLAYER_ID")[stat_col].transform(
                lambda s: compute_trend_slope(s, 5)
            )
        return df

    def _build_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build rest days, back-to-back, and season game number features."""
        if "GAME_DATE" not in df.columns:
            return df

        df["days_rest"] = df.groupby("PLAYER_ID")["GAME_DATE"].transform(
            calculate_days_rest
        )
        df["is_back_to_back"] = df.groupby("PLAYER_ID")["GAME_DATE"].transform(
            detect_back_to_back
        ).astype(int)

        if "SEASON" in df.columns:
            df["season_game_num"] = 0
            for player_id, group in df.groupby("PLAYER_ID"):
                nums = season_game_number(group["GAME_DATE"], group["SEASON"])
                df.loc[group.index, "season_game_num"] = nums.values
        return df

    def _build_season_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build expanding season-to-date averages (excluding current game)."""
        if "SEASON" not in df.columns:
            return df

        for stat_col, prefix in STAT_MAP.items():
            if stat_col not in df.columns:
                continue
            col_name = f"season_avg_{prefix}"
            # Shift to exclude current game, then expanding mean within season
            df[col_name] = (
                df.groupby(["PLAYER_ID", "SEASON"])[stat_col]
                .transform(lambda s: s.shift(1).expanding().mean())
            )
        return df

    def _build_career_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build expanding career averages (excluding current game)."""
        for stat_col, prefix in STAT_MAP.items():
            if stat_col not in df.columns:
                continue
            col_name = f"career_avg_{prefix}"
            df[col_name] = (
                df.groupby("PLAYER_ID")[stat_col]
                .transform(lambda s: s.shift(1).expanding().mean())
            )
        return df

    def _build_minutes_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build minutes-related features."""
        if "MIN" not in df.columns:
            return df

        df["minutes_avg_10"] = df.groupby("PLAYER_ID")["MIN"].transform(
            lambda s: compute_rolling_stat(s, 10, stat="mean")
        )
        return df

    def _build_home_away(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build home/away indicator from matchup string."""
        if "MATCHUP" in df.columns:
            # NBA API: 'vs.' = home, '@' = away
            df["is_home"] = df["MATCHUP"].str.contains("vs.", na=False).astype(int)
        else:
            df["is_home"] = 0
        return df

    def _build_opponent_features(
        self, df: pd.DataFrame, team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Build opponent-based features using season-to-date team stats.

        Args:
            df: Player game log DataFrame.
            team_stats: Team-level stats DataFrame with defensive ratings, pace, etc.
        """
        if "MATCHUP" not in df.columns:
            return df

        # Extract opponent team abbreviation from matchup
        def _extract_opponent(matchup: str) -> str | None:
            if pd.isna(matchup):
                return None
            if " vs. " in matchup:
                return matchup.split(" vs. ")[1].strip()
            if " @ " in matchup:
                return matchup.split(" @ ")[1].strip()
            return None

        df["OPP_TEAM"] = df["MATCHUP"].apply(_extract_opponent)

        # Merge opponent stats if available
        if "DEF_RATING" in team_stats.columns and "TEAM_ABBREVIATION" in team_stats.columns:
            opp_stats = team_stats[["TEAM_ABBREVIATION", "SEASON", "DEF_RATING", "PACE"]].copy()
            opp_stats = opp_stats.rename(columns={
                "TEAM_ABBREVIATION": "OPP_TEAM",
                "DEF_RATING": "opp_def_rating_std",
                "PACE": "opp_pace_std",
            })
            df = df.merge(opp_stats, on=["OPP_TEAM", "SEASON"], how="left")

        return df

    def _build_team_features(
        self, df: pd.DataFrame, team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Build team context features."""
        if "TEAM_ABBREVIATION" not in df.columns:
            return df

        if "PACE" in team_stats.columns and "OFF_RATING" in team_stats.columns:
            own_stats = team_stats[
                ["TEAM_ABBREVIATION", "SEASON", "PACE", "OFF_RATING"]
            ].copy()
            own_stats = own_stats.rename(columns={
                "PACE": "team_pace_std",
                "OFF_RATING": "team_off_rating_std",
            })
            df = df.merge(
                own_stats,
                on=["TEAM_ABBREVIATION", "SEASON"],
                how="left",
            )

        return df

    def _validate_no_leakage(self, df: pd.DataFrame) -> None:
        """Validate that no forbidden features are present."""
        present_forbidden = FORBIDDEN_FEATURES & set(df.columns)
        if present_forbidden:
            raise ValueError(
                f"Data leakage detected! Forbidden features present: {present_forbidden}"
            )

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        """Get the list of engineered feature column names."""
        non_feature_cols = {
            "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
            "TEAM_NAME", "GAME_ID", "GAME_DATE", "SEASON", "SEASON_YEAR",
            "SEASON_ID", "MATCHUP", "WL",
            "PTS", "REB", "AST", "FG3M", "FTM", "FGA", "FGM",
            "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
            "OREB", "DREB", "STL", "BLK", "TOV", "BLKA", "PFD",
            "PF", "PLUS_MINUS", "MIN", "IS_OUTLIER", "IS_TRADED_SEASON",
            "OPP_TEAM", "VIDEO_AVAILABLE",
            "NBA_FANTASY_PTS", "DD2", "TD3",
        }
        # Also exclude any _RANK columns from PlayerGameLogs/TeamGameLogs
        return [
            col for col in df.columns
            if col not in non_feature_cols and not col.endswith("_RANK")
        ]

    @staticmethod
    def get_target_columns() -> dict[str, str]:
        """Return mapping of target stat names to DataFrame column names."""
        return {
            "points": "PTS",
            "rebounds": "REB",
            "three_pointers_made": "FG3M",
            "free_throws_made": "FTM",
        }
