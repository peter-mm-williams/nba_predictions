"""Logging configuration for the NBA prop predictor."""

import logging
import sys
from pathlib import Path


def setup_logging(config: dict | None = None) -> logging.Logger:
    """Set up logging based on configuration.

    Args:
        config: Logging section from the config. If None, uses sensible defaults.

    Returns:
        Root logger for the application.
    """
    if config is None:
        config = {}

    level = getattr(logging, config.get("level", "INFO").upper(), logging.INFO)
    fmt = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_logging = config.get("file_logging", False)

    root_logger = logging.getLogger("nba_prop_predictor")
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if file_logging:
        log_dir = Path(config.get("log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "nba_prop_predictor.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the application namespace."""
    return logging.getLogger(f"nba_prop_predictor.{name}")
