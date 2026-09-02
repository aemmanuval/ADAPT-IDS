"""Hierarchical YAML configuration with environment variable overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"
_cached_config: dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_configs(*configs: dict) -> dict:
    """Left-fold merge: later dicts override earlier ones."""
    result: dict = {}
    for cfg in configs:
        result = _deep_merge(result, cfg)
    return result


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration from YAML files with optional overrides.

    Resolution order:
    1. configs/default.yaml
    2. *config_path* (if given)
    3. *overrides* dict
    4. Selected environment-variable overrides
    """
    global _cached_config

    with open(_DEFAULT_CONFIG) as fh:
        base = yaml.safe_load(fh) or {}

    if config_path is not None:
        path = Path(config_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        with open(path) as fh:
            overlay = yaml.safe_load(fh) or {}
        base = _deep_merge(base, overlay)

    if overrides:
        base = _deep_merge(base, overrides)

    env_level = os.environ.get("ADAPT_IDS_LOG_LEVEL")
    if env_level:
        base.setdefault("logging", {})["level"] = env_level

    env_seed = os.environ.get("ADAPT_IDS_SEED")
    if env_seed:
        base.setdefault("experiment", {})["random_seed"] = int(env_seed)

    _cached_config = base
    return base


def get_config() -> dict[str, Any]:
    """Return the most recently loaded config, loading defaults if needed."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def get_project_root() -> Path:
    return _PROJECT_ROOT
