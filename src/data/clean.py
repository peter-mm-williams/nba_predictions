"""Data cleaning pipeline for NBA box score data."""

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("data.clean")

# Expected columns after cleaning
REQUIRED_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "FG3M",
    "FTM",
    "FGA",
    "FGM",
    "FG3A",
    "FTA",
    "OREB",
    "DREB",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
]


class DataCleaner:
    """Cleans and validates NBA box score data."""

    def __init__(self, config: dict):
        """Initialize the cleaner.

        Args:
            config: Cleaning configuration from config.yaml.
        """
        self.min_minutes = config.get("min_minutes_played", 5)
        self.min_games = config.get("min_games_for_player", 20)
        self.trade_strategy = config.get("handle_trades", "split_season")
        self.exclude_playoffs = config.get("exclude_playoff_games", False)

    def clean_box_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline on box score data.

        Args:
            df: Raw box score DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Starting data cleaning on %d records...", len(df))

        df = self._standardize_columns(df)
        df = self._parse_dates(df)
        df = self._parse_minutes(df)
        df = self._deduplicate(df)
        df = self._remove_low_minute_games(df)
        df = self._filter_minimum_games(df)
        df = self.handle_trades(df, self.trade_strategy)
        df = self.flag_outliers(df)
        df = self._sort_chronologically(df)

        logger.info("Cleaning complete. %d records remaining.", len(df))
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to uppercase."""
        df = df.copy()
        df.columns = df.columns.str.upper().str.strip()

        # Standardize common column name variations
        rename_map = {
            "PLAYER_ID": "PLAYER_ID",
            "PLAYER_NAME": "PLAYER_NAME",
            "MATCHUP": "MATCHUP",
            "WL": "WL",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse game dates to datetime."""
        df = df.copy()
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    def _parse_minutes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse minutes played, handling 'MM:SS' format."""
        df = df.copy()
        if "MIN" in df.columns:
            def _parse_min(val):
                if pd.isna(val) or val == "" or val is None:
                    return 0.0
                if isinstance(val, (int, float)):
                    return float(val)
                val = str(val)
                if ":" in val:
                    parts = val.split(":")
                    return float(parts[0]) + float(parts[1]) / 60
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            df["MIN"] = df["MIN"].apply(_parse_min)
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records by (PLAYER_ID, GAME_ID)."""
        before = len(df)
        if "PLAYER_ID" in df.columns and "GAME_ID" in df.columns:
            df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"], keep="first")
        dupes_removed = before - len(df)
        if dupes_removed > 0:
            logger.info("Removed %d duplicate records.", dupes_removed)
        return df

    def _remove_low_minute_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove games where player played fewer than minimum minutes."""
        before = len(df)
        if "MIN" in df.columns:
            df = df[df["MIN"] >= self.min_minutes].copy()
        removed = before - len(df)
        if removed > 0:
            logger.info(
                "Removed %d low-minute games (<%d min).", removed, self.min_minutes
            )
        return df

    def _filter_minimum_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude players with fewer than minimum career games."""
        if "PLAYER_ID" not in df.columns:
            return df
        game_counts = df.groupby("PLAYER_ID")["GAME_ID"].nunique()
        valid_players = game_counts[game_counts >= self.min_games].index
        before = df["PLAYER_ID"].nunique()
        df = df[df["PLAYER_ID"].isin(valid_players)].copy()
        after = df["PLAYER_ID"].nunique()
        if before != after:
            logger.info(
                "Filtered players: %d -> %d (min %d games).",
                before,
                after,
                self.min_games,
            )
        return df

    def handle_trades(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle mid-season trades.

        Args:
            df: Box score DataFrame.
            strategy: One of 'split_season', 'exclude', 'aggregate'.

        Returns:
            DataFrame with trade handling applied.
        """
        if "PLAYER_ID" not in df.columns or "TEAM_ID" not in df.columns:
            return df
        if "SEASON" not in df.columns:
            return df

        if strategy == "split_season":
            # Keep records as-is; each game is already associated with the correct team.
            # Add a trade flag for players who played for multiple teams in a season.
            season_teams = df.groupby(["PLAYER_ID", "SEASON"])["TEAM_ID"].nunique()
            traded = season_teams[season_teams > 1].reset_index()
            traded_keys = set(
                zip(traded["PLAYER_ID"], traded["SEASON"], strict=False)
            )
            df = df.copy()
            df["IS_TRADED_SEASON"] = df.apply(
                lambda r: (r.get("PLAYER_ID"), r.get("SEASON")) in traded_keys,
                axis=1,
            )
            traded_count = len(traded)
            if traded_count > 0:
                logger.info("Flagged %d player-seasons with mid-season trades.", traded_count)
        elif strategy == "exclude":
            # Exclude games from seasons where a player was traded
            season_teams = df.groupby(["PLAYER_ID", "SEASON"])["TEAM_ID"].nunique()
            traded = season_teams[season_teams > 1].reset_index()
            traded_keys = set(
                zip(traded["PLAYER_ID"], traded["SEASON"], strict=False)
            )
            before = len(df)
            df = df[
                ~df.apply(
                    lambda r: (r.get("PLAYER_ID"), r.get("SEASON")) in traded_keys,
                    axis=1,
                )
            ].copy()
            logger.info("Excluded %d records from traded player-seasons.", before - len(df))
        # 'aggregate' strategy: keep all records as-is (team context handled in features)

        return df

    def flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag games with stats >3 standard deviations from player mean.

        Adds an 'IS_OUTLIER' boolean column.
        """
        df = df.copy()
        stat_cols = ["PTS", "REB", "AST", "FG3M", "FTM"]
        available_stats = [c for c in stat_cols if c in df.columns]

        if not available_stats or "PLAYER_ID" not in df.columns:
            df["IS_OUTLIER"] = False
            return df

        # Calculate player means and stds
        player_stats = df.groupby("PLAYER_ID")[available_stats].agg(["mean", "std"])
        player_stats.columns = [
            f"{col}_{stat}" for col, stat in player_stats.columns
        ]
        player_stats = player_stats.reset_index()

        df = df.merge(player_stats, on="PLAYER_ID", how="left")

        # Flag if any stat is >3σ from player mean
        outlier_mask = pd.Series(False, index=df.index)
        for col in available_stats:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            if mean_col in df.columns and std_col in df.columns:
                deviation = (df[col] - df[mean_col]).abs()
                threshold = 3 * df[std_col].fillna(0)
                # Only flag if std > 0 (player has variance)
                outlier_mask = outlier_mask | (
                    (deviation > threshold) & (df[std_col] > 0)
                )

        df["IS_OUTLIER"] = outlier_mask

        # Drop temporary stat columns
        temp_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
        df = df.drop(columns=temp_cols)

        outlier_count = df["IS_OUTLIER"].sum()
        if outlier_count > 0:
            logger.info("Flagged %d outlier game records.", outlier_count)

        return df

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by player and game date."""
        sort_cols = []
        if "PLAYER_ID" in df.columns:
            sort_cols.append("PLAYER_ID")
        if "GAME_DATE" in df.columns:
            sort_cols.append("GAME_DATE")
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check that required columns are present.

        Args:
            df: DataFrame to validate.

        Returns:
            True if all required columns are present.
        """
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            logger.warning("Missing required columns: %s", missing)
            return False
        return True
