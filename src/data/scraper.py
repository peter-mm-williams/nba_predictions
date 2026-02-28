"""Basketball Reference scraper for supplementary data."""

import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.io import ensure_dir, save_parquet
from src.utils.logging import get_logger

logger = get_logger("data.scraper")

BASE_URL = "https://www.basketball-reference.com"


class BasketballReferenceScraper:
    """Scraper for Basketball Reference data."""

    def __init__(self, config: dict):
        """Initialize the scraper.

        Args:
            config: Basketball Reference source configuration.
        """
        br_config = config.get("sources", {}).get("basketball_reference", {})
        self.enabled = br_config.get("enabled", True)
        self.rate_limit = br_config.get("rate_limit_seconds", 3.0)
        self.data_dir = Path(config.get("data_dir", "./data/raw/supplementary"))
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "NBA Prop Predictor Research (educational use)"}
        )

    def _rate_limit_pause(self) -> None:
        """Pause to respect rate limits."""
        time.sleep(self.rate_limit)

    def _fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch and parse a page.

        Args:
            url: Full URL to fetch.

        Returns:
            BeautifulSoup object of the page.

        Raises:
            requests.HTTPError: If the request fails.
        """
        self._rate_limit_pause()
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def scrape_team_ratings(self, season: int) -> pd.DataFrame:
        """Scrape team offensive/defensive ratings for a season.

        Args:
            season: Season start year (e.g., 2023 for 2023-24).

        Returns:
            DataFrame with team ratings.
        """
        if not self.enabled:
            logger.info("Basketball Reference scraping disabled.")
            return pd.DataFrame()

        url = f"{BASE_URL}/leagues/NBA_{season + 1}_ratings.html"
        logger.info("Scraping team ratings from %s", url)

        soup = self._fetch_page(url)
        table = soup.find("table", {"id": "ratings"})
        if table is None:
            logger.warning("Could not find ratings table for season %d", season)
            return pd.DataFrame()

        df = pd.read_html(str(table))[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(col).strip("_") for col in df.columns]
        df["SEASON"] = season
        return df

    def scrape_player_advanced_stats(self, season: int) -> pd.DataFrame:
        """Scrape advanced player stats for a season.

        Args:
            season: Season start year.

        Returns:
            DataFrame with advanced player stats.
        """
        if not self.enabled:
            return pd.DataFrame()

        url = f"{BASE_URL}/leagues/NBA_{season + 1}_advanced.html"
        logger.info("Scraping advanced player stats from %s", url)

        soup = self._fetch_page(url)
        table = soup.find("table", {"id": "advanced_stats"})
        if table is None:
            logger.warning("Could not find advanced stats table for season %d", season)
            return pd.DataFrame()

        df = pd.read_html(str(table))[0]
        # Remove header rows that appear within the data
        if "Rk" in df.columns:
            df = df[df["Rk"] != "Rk"]
        df["SEASON"] = season
        return df

    def scrape_all_seasons(self, start: int, end: int) -> None:
        """Scrape supplementary data for a range of seasons.

        Args:
            start: First season start year.
            end: Last season start year (inclusive).
        """
        output_dir = ensure_dir(self.data_dir)

        for season in range(start, end + 1):
            # Team ratings
            try:
                ratings = self.scrape_team_ratings(season)
                if not ratings.empty:
                    save_parquet(ratings, output_dir / f"team_ratings_{season}.parquet")
            except Exception as e:
                logger.error("Failed to scrape team ratings for %d: %s", season, e)

            # Advanced stats
            try:
                advanced = self.scrape_player_advanced_stats(season)
                if not advanced.empty:
                    save_parquet(advanced, output_dir / f"advanced_stats_{season}.parquet")
            except Exception as e:
                logger.error("Failed to scrape advanced stats for %d: %s", season, e)
