"""File I/O helpers for the NBA prop predictor."""

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to parquet format, creating parent dirs as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from parquet format."""
    return pd.read_parquet(path)


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data to JSON, creating parent dirs as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict | list:
    """Load data from JSON."""
    with open(path) as f:
        return json.load(f)
