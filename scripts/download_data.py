"""CLI for downloading NBA data.

Usage:
    python scripts/download_data.py --config config/config.yaml
    python scripts/download_data.py --config config/config.yaml --seasons 2022 2024
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Config
from src.data.download import NBADataDownloader
from src.data.scraper import BasketballReferenceScraper
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Download NBA data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Path to override configuration file",
    )
    parser.add_argument(
        "--seasons",
        nargs=2,
        type=int,
        default=None,
        help="Start and end season years (e.g., 2022 2024)",
    )
    parser.add_argument(
        "--skip-bbref",
        action="store_true",
        help="Skip Basketball Reference scraping",
    )
    args = parser.parse_args()

    config = Config.load(args.config, args.override)
    setup_logging(config.get("logging"))

    data_config = config["data"]
    data_config["data_dir"] = config.get("paths.data_dir", "./data/raw")

    start = args.seasons[0] if args.seasons else data_config["seasons"]["start"]
    end = args.seasons[1] if args.seasons else data_config["seasons"]["end"]

    # Download from NBA API
    downloader = NBADataDownloader(data_config)
    downloader.download_all_seasons(start, end)

    # Scrape supplementary data
    if not args.skip_bbref and data_config.get("sources", {}).get("basketball_reference", {}).get("enabled", True):
        scraper = BasketballReferenceScraper(data_config)
        scraper.scrape_all_seasons(start, end)


if __name__ == "__main__":
    main()
