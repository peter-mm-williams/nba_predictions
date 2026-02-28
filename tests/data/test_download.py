"""Tests for data download module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.download import NBADataDownloader


@pytest.fixture
def downloader():
    config = {
        "data_dir": "/tmp/test_data",
        "sources": {
            "nba_api": {
                "rate_limit_seconds": 0.0,
                "retry_attempts": 2,
                "timeout_seconds": 5,
            }
        },
    }
    return NBADataDownloader(config)


class TestNBADataDownloader:
    def test_init(self, downloader):
        assert downloader.rate_limit == 0.0
        assert downloader.retry_attempts == 2
        assert downloader.timeout == 5

    @patch("src.data.download.PlayerGameLog")
    def test_download_player_game_logs(self, mock_endpoint, downloader):
        """Test that player game logs are fetched and returned correctly."""
        mock_df = pd.DataFrame({
            "PLAYER_ID": [1, 2],
            "PTS": [20, 30],
        })
        mock_instance = MagicMock()
        mock_instance.get_data_frames.return_value = [mock_df]
        mock_endpoint.return_value = mock_instance

        result = downloader.download_player_game_logs(2023)

        assert isinstance(result, pd.DataFrame)
        assert "SEASON" in result.columns
        assert result["SEASON"].iloc[0] == 2023
        assert len(result) == 2

    @patch("src.data.download.TeamGameLog")
    def test_download_team_game_logs(self, mock_endpoint, downloader):
        """Test team game log download."""
        mock_df = pd.DataFrame({
            "TEAM_ID": [1, 2],
            "W": [1, 0],
        })
        mock_instance = MagicMock()
        mock_instance.get_data_frames.return_value = [mock_df]
        mock_endpoint.return_value = mock_instance

        result = downloader.download_team_game_logs(2023)
        assert isinstance(result, pd.DataFrame)
        assert "SEASON" in result.columns

    @patch("src.data.download.PlayerGameLog")
    def test_retry_logic(self, mock_endpoint, downloader):
        """Test that API calls are retried on failure."""
        mock_endpoint.side_effect = [
            ConnectionError("Timeout"),
            ConnectionError("Timeout"),
        ]

        with pytest.raises(RuntimeError, match="API call failed after 2 attempts"):
            downloader.download_player_game_logs(2023)

    @patch("src.data.download.PlayerGameLog")
    def test_retry_success_on_second_attempt(self, mock_endpoint, downloader):
        """Test that retry succeeds after initial failure."""
        mock_df = pd.DataFrame({"PLAYER_ID": [1], "PTS": [20]})
        mock_instance = MagicMock()
        mock_instance.get_data_frames.return_value = [mock_df]

        # First call fails, second succeeds
        mock_endpoint.side_effect = [ConnectionError("Timeout"), mock_instance]

        result = downloader.download_player_game_logs(2023)
        assert len(result) == 1

    def test_get_all_players(self):
        """Test that static player list is returned."""
        result = NBADataDownloader.get_all_players()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_get_all_teams(self):
        """Test that static team list is returned."""
        result = NBADataDownloader.get_all_teams()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30
