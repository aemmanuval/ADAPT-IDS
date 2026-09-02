"""Feature selection and importance analysis for CIC-IDS2017 flow features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from adaptive_ids.utils.logging import get_logger

logger = get_logger("features.selection")

RECOMMENDED_FLOW_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def select_available_features(
    df: pd.DataFrame,
    candidates: list[str] | None = None,
) -> list[str]:
    """Return the subset of *candidates* that actually exist in *df*."""
    if candidates is None:
        candidates = RECOMMENDED_FLOW_FEATURES
    available = [c for c in candidates if c in df.columns]
    logger.info("Selected %d / %d candidate features", len(available), len(candidates))
    return available


def compute_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-feature descriptive statistics for the given columns."""
    subset = df[feature_cols].select_dtypes(include="number")
    stats = subset.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    stats["missing"] = subset.isnull().sum()
    stats["n_unique"] = subset.nunique()
    stats["skewness"] = subset.skew()
    stats["kurtosis"] = subset.kurtosis()
    return stats


def compare_window_distributions(
    window_a: pd.DataFrame,
    window_b: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Side-by-side mean/std comparison of two temporal windows."""
    rows = []
    for col in feature_cols:
        if col not in window_a.columns or col not in window_b.columns:
            continue
        a, b = window_a[col], window_b[col]
        rows.append({
            "feature": col,
            "mean_a": a.mean(), "std_a": a.std(),
            "mean_b": b.mean(), "std_b": b.std(),
            "mean_shift": b.mean() - a.mean(),
            "relative_shift": (
                abs(b.mean() - a.mean()) / (abs(a.mean()) + 1e-10)
            ),
        })
    return pd.DataFrame(rows)
