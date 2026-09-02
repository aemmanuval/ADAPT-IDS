"""Publication-quality plots for IDS research experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from adaptive_ids.utils.logging import get_logger

logger = get_logger("visualization")

FIGURE_DIR = Path("results/figures")
DPI = 150
STYLE = "seaborn-v0_8-whitegrid"


def _setup() -> None:
    plt.style.use(STYLE)
    sns.set_palette("colorblind")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(
    labels: np.ndarray | pd.Series,
    title: str = "Class Distribution",
    filename: str = "class_distribution.png",
) -> Path:
    _setup()
    counts = pd.Series(labels).value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot.barh(ax=ax, color=sns.color_palette("colorblind", len(counts)))
    ax.set_xlabel("Count")
    ax.set_title(title)
    for i, v in enumerate(counts):
        pct = v / counts.sum() * 100
        ax.text(v + counts.max() * 0.01, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=9)
    plt.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_confusion_matrix(
    cm: np.ndarray | list,
    labels: list[str],
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
) -> Path:
    _setup()
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_performance_over_time(
    windowed_metrics: list[dict[str, Any]],
    metric_name: str = "f1",
    drift_positions: list[int] | None = None,
    title: str = "Performance Over Time",
    filename: str = "performance_over_time.png",
) -> Path:
    _setup()
    windows = [m["window_id"] for m in windowed_metrics]
    values = [m.get(metric_name, 0) for m in windowed_metrics]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(windows, values, "o-", linewidth=1.5, markersize=3, label=metric_name.upper())
    ax.set_xlabel("Window")
    ax.set_ylabel(metric_name.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    if drift_positions:
        for pos in drift_positions:
            win_idx = None
            for m in windowed_metrics:
                if m["window_start"] <= pos < m["window_end"]:
                    win_idx = m["window_id"]
                    break
            if win_idx is not None:
                ax.axvline(x=win_idx, color="red", linestyle="--", alpha=0.7, label="Drift")

        handles, lbls = ax.get_legend_handles_labels()
        unique = dict(zip(lbls, handles))
        ax.legend(unique.values(), unique.keys())
    else:
        ax.legend()

    plt.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_error_rate_with_drift(
    error_rates: list[float],
    drift_positions: list[int],
    window_size: int = 100,
    title: str = "Error Rate & Drift Events",
    filename: str = "error_rate_drift.png",
) -> Path:
    """Rolling error rate with vertical lines at drift detections."""
    _setup()
    fig, ax = plt.subplots(figsize=(14, 5))

    if len(error_rates) > window_size:
        rolling = pd.Series(error_rates).rolling(window=window_size, min_periods=1).mean()
    else:
        rolling = pd.Series(error_rates)

    ax.plot(range(len(rolling)), rolling, linewidth=1, alpha=0.8, label=f"Rolling error (w={window_size})")

    for pos in drift_positions:
        ax.axvline(x=pos, color="red", linestyle="--", alpha=0.6)

    if drift_positions:
        ax.axvline(x=drift_positions[0], color="red", linestyle="--", alpha=0.6, label="Drift detected")

    ax.set_xlabel("Stream Position")
    ax.set_ylabel("Error Rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str],
    max_features: int = 12,
    filename: str = "feature_distributions.png",
) -> Path:
    _setup()
    feats = [f for f in features if f in df.columns][:max_features]
    n = len(feats)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, feat in enumerate(feats):
        ax = axes[i]
        data = df[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) > 50_000:
            data = data.sample(50_000, random_state=42)
        ax.hist(data, bins=50, edgecolor="none", alpha=0.7)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=12)
    plt.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_traffic_over_time(
    timestamps: pd.Series,
    title: str = "Traffic Volume Over Time",
    filename: str = "traffic_over_time.png",
) -> Path:
    _setup()
    ts = pd.to_datetime(timestamps, errors="coerce").dropna()
    counts = ts.dt.floor("h").value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(counts.index, counts.values, width=0.03, alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow Count")
    ax.set_title(title)
    fig.autofmt_xdate()
    plt.tight_layout()

    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path
