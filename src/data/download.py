"""Data download clients for NBA API and Basketball Reference."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    PlayerGameLog,
    PlayerGameLogs,
    TeamGameLog,
    TeamGameLogs,
)
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams

from src.utils.io import ensure_dir, save_parquet
from src.utils.logging import get_logger
from src.utils.temporal import nba_season_string

logger = get_logger("data.download")


class NBADataDownloader:
    """Downloads NBA data from the nba_api package."""

    def __init__(self, config: dict):
        """Initialize the downloader.

        Args:
            config: Data source configuration dict with rate_limit, retry, timeout settings.
        """
        api_config = config.get("sources", {}).get("nba_api", {})
        self.rate_limit = api_config.get("rate_limit_seconds", 0.6)
        self.retry_attempts = api_config.get("retry_attempts", 3)
        self.timeout = api_config.get("timeout_seconds", 30)
        self.data_dir = Path(config.get("data_dir", "./data/raw"))

    def _rate_limit_pause(self) -> None:
        """Pause to respect API rate limits."""
        time.sleep(self.rate_limit)

    def _api_call_with_retry(self, endpoint_cls, **kwargs) -> pd.DataFrame:
        """Make an API call with retry logic.

        Args:
            endpoint_cls: nba_api endpoint class.
            **kwargs: Arguments to pass to the endpoint.

        Returns:
            DataFrame from the API response.

        Raises:
            Exception: If all retry attempts fail.
        """
        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._rate_limit_pause()
                endpoint = endpoint_cls(**kwargs, timeout=self.timeout)
                return endpoint.get_data_frames()[0]
            except Exception as e:
                last_error = e
                wait_time = 2**attempt
                logger.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt,
                    self.retry_attempts,
                    str(e),
                    wait_time,
                )
                time.sleep(wait_time)
        raise RuntimeError(
            f"API call failed after {self.retry_attempts} attempts: {last_error}"
        )

    def download_player_game_logs(self, season: int) -> pd.DataFrame:
        """Download player-level game logs for a season via PlayerGameLogs.

        Uses PlayerGameLogs (plural) to fetch all player game logs for the
        season in a single call. This endpoint returns PLAYER_ID, PLAYER_NAME,
        TEAM_ID, FG3M, and all other box score columns needed downstream.

        Args:
            season: Season start year (e.g., 2023 for 2023-24 season).

        Returns:
            DataFrame with player game logs.
        """
        season_str = nba_season_string(season)
        logger.info("Downloading player game logs for %s...", season_str)

        df = self._api_call_with_retry(
            PlayerGameLogs,
            season_nullable=season_str,
            season_type_nullable="Regular Season",
        )
        df["SEASON"] = season
        return df

    def download_team_game_logs(self, season: int) -> pd.DataFrame:
        """Download team-level game logs for a season via TeamGameLogs.

        Uses TeamGameLogs (plural) to fetch all team game logs for the
        season in a single call. This endpoint returns TEAM_ID, FG3M,
        and all other box score columns needed downstream.

        Args:
            season: Season start year.

        Returns:
            DataFrame with team game logs.
        """
        season_str = nba_season_string(season)
        logger.info("Downloading team game logs for %s...", season_str)

        df = self._api_call_with_retry(
            TeamGameLogs,
            season_nullable=season_str,
            season_type_nullable="Regular Season",
        )
        df["SEASON"] = season
        return df

    def download_single_player_game_log(
        self, player_id: int, season: int
    ) -> pd.DataFrame:
        """Download game logs for a single player in a season.

        Args:
            player_id: NBA player ID.
            season: Season start year.

        Returns:
            DataFrame with player game logs.
        """
        season_str = nba_season_string(season)
        logger.info(
            "Downloading game log for player %d, season %s...", player_id, season_str
        )

        df = self._api_call_with_retry(
            PlayerGameLog,
            player_id=player_id,
            season=season_str,
            season_type_all_star="Regular Season",
        )
        df["SEASON"] = season
        return df

    def download_single_team_game_log(
        self, team_id: int, season: int
    ) -> pd.DataFrame:
        """Download game logs for a single team in a season.

        Args:
            team_id: NBA team ID.
            season: Season start year.

        Returns:
            DataFrame with team game logs.
        """
        season_str = nba_season_string(season)
        logger.info(
            "Downloading game log for team %d, season %s...", team_id, season_str
        )

        df = self._api_call_with_retry(
            TeamGameLog,
            team_id=team_id,
            season=season_str,
            season_type_all_star="Regular Season",
        )
        df["SEASON"] = season
        return df

    def download_league_game_logs(self, season: int) -> pd.DataFrame:
        """Download league-wide game logs for a season.

        Alias for download_player_game_logs. Kept for backwards compatibility.

        Args:
            season: Season start year.

        Returns:
            DataFrame with league game logs.
        """
        return self.download_player_game_logs(season)

    # Columns that must be present in downloaded player game logs.
    # Used to detect stale files from older endpoints.
    REQUIRED_PLAYER_COLUMNS = {"PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "PTS", "FG3M"}

    def download_all_seasons(self, start: int, end: int, force: bool = False) -> None:
        """Download all data for a range of seasons and save to disk.

        Args:
            start: First season start year.
            end: Last season start year (inclusive).
            force: If True, re-download even if files exist.
        """
        box_scores_dir = ensure_dir(self.data_dir / "box_scores")

        for season in range(start, end + 1):
            season_str = nba_season_string(season)
            output_path = box_scores_dir / f"player_game_logs_{season}.parquet"

            if output_path.exists() and not force:
                # Validate that cached file has required columns
                try:
                    existing = pd.read_parquet(output_path, columns=["PLAYER_ID"])
                except Exception:
                    logger.warning(
                        "Season %s file is missing required columns. Re-downloading...",
                        season_str,
                    )
                    output_path.unlink()
                else:
                    logger.info("Season %s already downloaded, skipping.", season_str)
                    continue

            try:
                df = self.download_player_game_logs(season)
                save_parquet(df, output_path)
                logger.info(
                    "Saved %d records for season %s to %s",
                    len(df),
                    season_str,
                    output_path,
                )
            except Exception as e:
                logger.error("Failed to download season %s: %s", season_str, e)

            # Also download team logs
            team_path = box_scores_dir / f"team_game_logs_{season}.parquet"
            if not team_path.exists() or force:
                try:
                    team_df = self.download_team_game_logs(season)
                    save_parquet(team_df, team_path)
                except Exception as e:
                    logger.error("Failed to download team logs for %s: %s", season_str, e)

    @staticmethod
    def get_all_players() -> pd.DataFrame:
        """Get a DataFrame of all NBA players."""
        return pd.DataFrame(nba_players.get_players())

    @staticmethod
    def get_all_teams() -> pd.DataFrame:
        """Get a DataFrame of all NBA teams."""
        return pd.DataFrame(nba_teams.get_teams())
