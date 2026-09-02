"""IDS-specific evaluation metrics.

Prioritises recall, F1, FPR, FNR over raw accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    matthews_corrcoef, accuracy_score,
)

from adaptive_ids.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_label: str = "ATTACK",
    average: str = "binary",
) -> dict[str, Any]:
    """Compute IDS evaluation metrics.

    For binary: positive_label determines the positive class.
    For multiclass: macro/weighted averages are computed.
    """
    labels = sorted(set(y_true) | set(y_pred))
    is_binary = len(labels) <= 2

    if is_binary and positive_label in labels:
        pos_label = positive_label
        avg = "binary"
    elif is_binary:
        pos_label = labels[-1]
        avg = "binary"
    else:
        pos_label = positive_label
        avg = "macro"

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, pos_label=pos_label, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if not is_binary:
        metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_labels"] = labels

    if is_binary and len(cm) == 2:
        neg_label = [l for l in labels if l != pos_label][0]
        neg_idx = labels.index(neg_label)
        pos_idx = labels.index(pos_label)
        tp = int(cm[pos_idx, pos_idx])
        fn = int(cm[pos_idx, neg_idx])
        fp = int(cm[neg_idx, pos_idx])
        tn = int(cm[neg_idx, neg_idx])
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        metrics["tp"] = tp
        metrics["fp"] = fp
        metrics["fn"] = fn
        metrics["tn"] = tn
    else:
        metrics["fpr"] = None
        metrics["fnr"] = None

    metrics["support"] = int(len(y_true))
    metrics["class_report"] = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0,
    )

    return metrics


def compute_windowed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 5000,
    *,
    positive_label: str = "ATTACK",
) -> list[dict[str, Any]]:
    """Compute metrics over non-overlapping windows for performance-over-time."""
    n = len(y_true)
    results = []

    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        wt = y_true[start:end]
        wp = y_pred[start:end]

        if len(wt) == 0:
            continue

        m = compute_metrics(wt, wp, positive_label=positive_label)
        m["window_start"] = start
        m["window_end"] = end
        m["window_id"] = len(results)
        results.append(m)

    return results


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Metrics saved to %s", path)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("Confusion matrix saved to %s", path)


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    *,
    index: np.ndarray | None = None,
) -> None:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if index is not None:
        df["index"] = index
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Predictions saved to %s", path)
