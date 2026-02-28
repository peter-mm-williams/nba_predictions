"""Data download clients for NBA API and Basketball Reference."""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    LeagueGameLog,
    PlayerGameLog,
    TeamGameLog,
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
        """Download player-level game logs for a season.

        Args:
            season: Season start year (e.g., 2023 for 2023-24 season).

        Returns:
            DataFrame with player game logs.
        """
        season_str = nba_season_string(season)
        logger.info("Downloading player game logs for %s...", season_str)

        df = self._api_call_with_retry(
            PlayerGameLog,
            player_id_nullable="",
            season_nullable=season_str,
            season_type_nullable="Regular Season",
        )
        df["SEASON"] = season
        return df

    def download_team_game_logs(self, season: int) -> pd.DataFrame:
        """Download team-level game logs for a season.

        Args:
            season: Season start year.

        Returns:
            DataFrame with team game logs.
        """
        season_str = nba_season_string(season)
        logger.info("Downloading team game logs for %s...", season_str)

        df = self._api_call_with_retry(
            TeamGameLog,
            season_nullable=season_str,
            season_type_nullable="Regular Season",
        )
        df["SEASON"] = season
        return df

    def download_league_game_logs(self, season: int) -> pd.DataFrame:
        """Download league-wide game logs for a season.

        Args:
            season: Season start year.

        Returns:
            DataFrame with league game logs.
        """
        season_str = nba_season_string(season)
        logger.info("Downloading league game logs for %s...", season_str)

        df = self._api_call_with_retry(
            LeagueGameLog,
            season=season_str,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
        )
        df["SEASON"] = season
        return df

    def download_all_seasons(self, start: int, end: int) -> None:
        """Download all data for a range of seasons and save to disk.

        Args:
            start: First season start year.
            end: Last season start year (inclusive).
        """
        box_scores_dir = ensure_dir(self.data_dir / "box_scores")

        for season in range(start, end + 1):
            season_str = nba_season_string(season)
            output_path = box_scores_dir / f"player_game_logs_{season}.parquet"

            if output_path.exists():
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
            try:
                team_df = self.download_team_game_logs(season)
                team_path = box_scores_dir / f"team_game_logs_{season}.parquet"
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
