"""Temporal (chronological) train/test splitting for IDS evaluation.

Preserves time ordering — critical for drift research.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from adaptive_ids.utils.logging import get_logger

logger = get_logger("evaluation.temporal")


def temporal_split(
    df: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.20,
) -> dict[str, pd.DataFrame]:
    """Split dataset chronologically by timestamp.

    Guarantees: max(train_ts) < min(val_ts) < min(test_ts).
    """
    if timestamp_column not in df.columns:
        logger.warning(
            "Timestamp column '%s' not found — falling back to row-order split",
            timestamp_column,
        )
        return _row_order_split(df, train_fraction, validation_fraction, test_fraction)

    df = df.copy()
    df["_ts_parsed"] = pd.to_datetime(df[timestamp_column], errors="coerce")
    n_bad = df["_ts_parsed"].isna().sum()
    if n_bad > 0:
        logger.warning("Dropped %d rows with unparseable timestamps", n_bad)
        df = df.dropna(subset=["_ts_parsed"])

    df = df.sort_values("_ts_parsed").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))

    splits = {
        "train": df.iloc[:train_end].drop(columns=["_ts_parsed"]),
        "validation": df.iloc[train_end:val_end].drop(columns=["_ts_parsed"]),
        "test": df.iloc[val_end:].drop(columns=["_ts_parsed"]),
    }

    for name, split_df in splits.items():
        logger.info("  %s: %d rows", name, len(split_df))

    _validate_temporal_integrity(splits, timestamp_column)
    return splits


def _row_order_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> dict[str, pd.DataFrame]:
    n = len(df)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))
    return {
        "train": df.iloc[:t_end].copy(),
        "validation": df.iloc[t_end:v_end].copy(),
        "test": df.iloc[v_end:].copy(),
    }


def _validate_temporal_integrity(
    splits: dict[str, pd.DataFrame],
    ts_col: str,
) -> None:
    """Assert that train timestamps strictly precede test timestamps."""
    if ts_col not in splits["train"].columns:
        return

    train_ts = pd.to_datetime(splits["train"][ts_col], errors="coerce").dropna()
    test_ts = pd.to_datetime(splits["test"][ts_col], errors="coerce").dropna()

    if len(train_ts) == 0 or len(test_ts) == 0:
        return

    train_max = train_ts.max()
    test_min = test_ts.min()

    if train_max > test_min:
        logger.error(
            "TEMPORAL LEAKAGE: train max (%s) > test min (%s)!",
            train_max, test_min,
        )
        raise ValueError(
            f"Temporal leakage detected: max training timestamp ({train_max}) "
            f"is after min test timestamp ({test_min})"
        )
    logger.info("Temporal integrity OK: train_max=%s < test_min=%s", train_max, test_min)


def random_split(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
    test_fraction: float = 0.20,
    random_seed: int = 42,
    *,
    stratify_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Conventional random split for baseline comparison."""
    from sklearn.model_selection import train_test_split

    strat = df[stratify_column] if stratify_column and stratify_column in df.columns else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=strat,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def get_temporal_info(
    df: pd.DataFrame,
    timestamp_column: str = "Timestamp",
) -> dict[str, Any]:
    """Summarise the temporal range and distribution of the dataset."""
    if timestamp_column not in df.columns:
        return {"available": False}

    ts = pd.to_datetime(df[timestamp_column], errors="coerce").dropna()
    if len(ts) == 0:
        return {"available": False}

    return {
        "available": True,
        "min": str(ts.min()),
        "max": str(ts.max()),
        "duration": str(ts.max() - ts.min()),
        "n_valid": int(len(ts)),
        "n_invalid": int(df[timestamp_column].shape[0] - len(ts)),
        "unique_dates": sorted(ts.dt.date.unique().astype(str).tolist()),
    }
