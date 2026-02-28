"""Configuration loader with YAML + environment variable support."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


ENV_VAR_PATTERN = re.compile(r"\$\{(\w+):([^}]*)\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR:default} patterns in config values."""
    if isinstance(value, str):
        match = ENV_VAR_PATTERN.fullmatch(value)
        if match:
            env_var, default = match.group(1), match.group(2)
            resolved = os.environ.get(env_var, default)
            # Try to cast to int/float if the default looks numeric
            if default.isdigit():
                try:
                    return int(resolved)
                except ValueError:
                    pass
            try:
                return float(resolved)
            except ValueError:
                pass
            return resolved
        # Handle partial env var patterns within strings
        def _replace(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2))
        return ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Configuration manager with YAML loading and environment variable resolution."""

    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(cls, config_path: str, override_path: str | None = None) -> "Config":
        """Load configuration from YAML file with optional environment override.

        Args:
            config_path: Path to the main config YAML file.
            override_path: Optional path to an override YAML file (e.g., config.dev.yaml).

        Returns:
            Config instance with resolved values.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        if override_path:
            override_path = Path(override_path)
            if override_path.exists():
                with open(override_path) as f:
                    overrides = yaml.safe_load(f) or {}
                data = _deep_merge(data, overrides)

        data = _resolve_env_vars(data)
        return cls(data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value using dot-separated key path.

        Example:
            config.get("training.batch_size")
            config.get("models.sequential.hidden_dim")
        """
        keys = dotted_key.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Dict-style access to top-level config sections."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def data(self) -> dict:
        """Return the full configuration dictionary."""
        return self._data

    def to_flat_dict(self, prefix: str = "") -> dict[str, Any]:
        """Flatten nested config into dot-separated keys."""

        def _flatten(d: dict, parent_key: str) -> dict:
            items: dict[str, Any] = {}
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(_flatten(v, new_key))
                else:
                    items[new_key] = v
            return items

        return _flatten(self._data, prefix)
