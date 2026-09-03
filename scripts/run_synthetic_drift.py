#!/usr/bin/env python3
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
"""Phase 4: Controlled synthetic drift experiments.

Tests how each drift type affects IDS performance and how well
different adaptation strategies handle them.

Drift types:
  - Sudden: abrupt distribution change at a point
  - Gradual: progressive change over a range
  - Incremental: continuous slow shift across the stream
  - Recurring: drift that appears and disappears cyclically
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.evaluation.temporal import temporal_split
from adaptive_ids.evaluation.metrics import compute_metrics, compute_windowed_metrics
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.streaming.stream import CSVStream
from adaptive_ids.drift.detectors import create_detector
from adaptive_ids.drift.synthetic import SyntheticDriftGenerator
from adaptive_ids.adaptation.strategies import (
    StaticStrategy, DriftTriggeredStrategy, AdaptiveModelManager,
)
from adaptive_ids.visualization.plots import plot_performance_over_time
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def run_drift_scenario(
    name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list[str],
    config: dict,
    seed: int,
) -> dict:
    """Run static + adaptive on one drift scenario."""
    algo = "lightgbm"
    model_params = config["models"].get(algo, {})
    drift_cfg = config["drift"]
    results = {}

    for strategy_name, strategy in [("static", StaticStrategy()), ("adaptive", DriftTriggeredStrategy(cooldown=2000))]:
        model = BaselineIDS(algo, params=model_params, random_seed=seed)
        model.fit(X_train, y_train)
        detector = create_detector(drift_cfg)

        manager = AdaptiveModelManager(
            algorithm=algo, model_params=model_params,
            strategy=strategy, detector=detector,
            window_size=20000, random_seed=seed,
        )
        manager.set_initial_model(model)

        stream = CSVStream(X_test, y_test, feature_names=feature_names)
        preds, errors = [], []

        for event in stream:
            y_pred, is_correct, _ = manager.process_sample(
                event["features"], event["label"], event["index"]
            )
            preds.append(y_pred)
            errors.append(0 if is_correct else 1)

        metrics = compute_metrics(y_test, np.array(preds), positive_label="ATTACK")
        windowed = compute_windowed_metrics(y_test, np.array(preds), window_size=5000, positive_label="ATTACK")

        results[strategy_name] = {
            "f1": metrics["f1"],
            "recall": metrics["recall"],
            "fnr": metrics.get("fnr", 0),
            "mcc": metrics["mcc"],
            "n_retrains": manager.n_retrains,
            "windowed": windowed,
        }

    return results


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("synthetic_drift")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    proc_dir = root / config["dataset"]["processed_dir"]
    results_dir = root / "results" / "synthetic"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "cleaned.parquet"
    if not data_path.exists():
        logger.error("Run preprocess_dataset.py first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  —  Synthetic Drift Experiments (Phase 4)")
    print("=" * 60 + "\n")

    df = pd.read_parquet(data_path)
    ts_col = config["dataset"]["timestamp_column"]
    eval_cfg = config["evaluation"]

    splits = temporal_split(
        df, timestamp_column=ts_col,
        train_fraction=eval_cfg["train_fraction"],
        validation_fraction=eval_cfg["validation_fraction"],
        test_fraction=eval_cfg["test_fraction"],
    )

    pipeline = PreprocessingPipeline(config)
    _, X_train, y_train = pipeline.fit_transform(splits["train"])
    _, X_test_orig, y_test_orig = pipeline.transform(splits["test"])

    print(f"Train: {X_train.shape[0]:,} | Test: {X_test_orig.shape[0]:,} | Features: {X_train.shape[1]}")

    gen = SyntheticDriftGenerator(seed=seed)

    drift_scenarios = {
        "no_drift": lambda X, y: (X.copy(), y.copy()),
        "sudden": lambda X, y: gen.sudden_drift(X, y, position=0.5, magnitude=1.5),
        "gradual": lambda X, y: gen.gradual_drift(X, y, start=0.3, end=0.7, magnitude=1.5),
        "incremental": lambda X, y: gen.incremental_drift(X, y, magnitude=0.8),
        "recurring": lambda X, y: gen.recurring_drift(X, y, cycle_length=0.2, magnitude=1.5),
    }

    all_results = {}

    for drift_name, drift_fn in drift_scenarios.items():
        print(f"\n--- Drift type: {drift_name} ---")
        X_test, y_test = drift_fn(X_test_orig, y_test_orig)

        results = run_drift_scenario(
            drift_name, X_train, y_train, X_test, y_test,
            pipeline.feature_columns, config, seed,
        )
        all_results[drift_name] = results

        static_f1 = results["static"]["f1"]
        adaptive_f1 = results["adaptive"]["f1"]
        retrains = results["adaptive"]["n_retrains"]
        print(f"  Static  F1={static_f1:.4f}")
        print(f"  Adaptive F1={adaptive_f1:.4f} (retrains={retrains})")

        for strat in ["static", "adaptive"]:
            plot_performance_over_time(
                results[strat]["windowed"],
                metric_name="f1",
                title=f"F1 — {drift_name} ({strat})",
                filename=f"f1_synthetic_{drift_name}_{strat}.png",
            )

    # Summary table
    print("\n" + "=" * 60)
    print("  SYNTHETIC DRIFT COMPARISON")
    print("=" * 60)
    print(f"{'Drift Type':<16} {'Static F1':>10} {'Adaptive F1':>12} {'Recovery':>10} {'Retrains':>10}")
    print("-" * 62)

    summary_rows = []
    for drift_name, results in all_results.items():
        s = results["static"]
        a = results["adaptive"]
        recovery = a["f1"] - s["f1"]
        row = {
            "drift_type": drift_name,
            "static_f1": s["f1"],
            "static_recall": s["recall"],
            "adaptive_f1": a["f1"],
            "adaptive_recall": a["recall"],
            "recovery": recovery,
            "retrains": a["n_retrains"],
        }
        summary_rows.append(row)
        print(f"{drift_name:<16} {s['f1']:>10.4f} {a['f1']:>12.4f} {recovery:>+10.4f} {a['n_retrains']:>10}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "synthetic_drift_summary.csv", index=False)

    with open(results_dir / "synthetic_drift_results.json", "w") as fh:
        serializable = {}
        for drift_name, results in all_results.items():
            serializable[drift_name] = {
                strat: {k: v for k, v in data.items() if k != "windowed"}
                for strat, data in results.items()
            }
        json.dump(serializable, fh, indent=2, default=str)

    # Combined comparison plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (drift_name, results) in enumerate(all_results.items()):
        if idx >= 5:
            break
        ax = axes[idx]
        for strat, color in [("static", "#d62728"), ("adaptive", "#1f77b4")]:
            wm = results[strat]["windowed"]
            windows = [w["window_id"] for w in wm]
            f1s = [w["f1"] for w in wm]
            ax.plot(windows, f1s, linewidth=1, label=strat, color=color, alpha=0.8)
        ax.set_title(drift_name, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Window")
        ax.set_ylabel("F1")
        ax.legend(fontsize=8)

    axes[5].set_visible(False)
    plt.suptitle("Synthetic Drift — Static vs Adaptive", fontsize=13)
    plt.tight_layout()
    fig_path = root / "results" / "figures" / "synthetic_drift_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults saved to {results_dir}")
    print(f"Comparison plot: {fig_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
