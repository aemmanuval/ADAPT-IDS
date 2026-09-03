#!/usr/bin/env python3
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
"""Phase 2-3: Compare adaptation strategies under temporal drift.

Runs the same temporal test stream through three strategies:
  A. Static (no retraining)
  B. Periodic retraining (every N samples)
  C. Drift-triggered retraining (retrain only when ADWIN fires)

Measures F1-over-time, total retrains, and adaptation cost for each.
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
from adaptive_ids.drift.detectors import create_detector, ADWINDetector
from adaptive_ids.drift.events import DriftEventLogger
from adaptive_ids.adaptation.strategies import (
    StaticStrategy, PeriodicStrategy, DriftTriggeredStrategy,
    AdaptiveModelManager,
)
from adaptive_ids.visualization.plots import plot_performance_over_time
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def run_strategy(
    strategy_name: str,
    manager: AdaptiveModelManager,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Stream test data through one strategy and collect results."""
    logger = get_logger(f"adapt.{strategy_name}")
    stream = CSVStream(X_test, y_test, feature_names=feature_names)

    all_preds: list[str] = []
    errors: list[int] = []
    retrain_positions: list[int] = []

    t0 = time.perf_counter()
    for event in tqdm(stream, total=len(stream), desc=f"  {strategy_name}", ncols=80):
        y_pred, is_correct, did_retrain = manager.process_sample(
            event["features"], event["label"], event["index"]
        )
        all_preds.append(y_pred)
        errors.append(0 if is_correct else 1)
        if did_retrain:
            retrain_positions.append(event["index"])

    elapsed = time.perf_counter() - t0

    overall = compute_metrics(y_test, np.array(all_preds), positive_label="ATTACK")
    windowed = compute_windowed_metrics(
        y_test, np.array(all_preds), window_size=5000, positive_label="ATTACK"
    )

    return {
        "strategy": strategy_name,
        "overall_metrics": overall,
        "windowed_metrics": windowed,
        "n_retrains": manager.n_retrains,
        "retrain_positions": retrain_positions,
        "retrain_log": manager.retrain_log,
        "total_retrain_time_s": round(manager.total_retrain_time, 2),
        "total_elapsed_s": round(elapsed, 2),
        "strategy_stats": manager.strategy.get_stats(),
    }


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("adaptation_experiment")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    proc_dir = root / config["dataset"]["processed_dir"]
    results_dir = root / "results" / "adaptation"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "cleaned.parquet"
    if not data_path.exists():
        logger.error("Run preprocess_dataset.py first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  —  Adaptation Strategy Comparison")
    print("=" * 60 + "\n")

    df = pd.read_parquet(data_path)
    ts_col = config["dataset"]["timestamp_column"]
    eval_cfg = config["evaluation"]

    splits = temporal_split(
        df,
        timestamp_column=ts_col,
        train_fraction=eval_cfg["train_fraction"],
        validation_fraction=eval_cfg["validation_fraction"],
        test_fraction=eval_cfg["test_fraction"],
    )

    pipeline = PreprocessingPipeline(config)
    _, X_train, y_train = pipeline.fit_transform(splits["train"])
    _, X_test, y_test = pipeline.transform(splits["test"])

    print(f"Train: {X_train.shape[0]:,} samples | Test: {X_test.shape[0]:,} samples")
    print(f"Features: {X_train.shape[1]}")

    algo = "lightgbm"
    model_params = config["models"].get(algo, {})
    drift_cfg = config["drift"]

    strategies = {
        "static": StaticStrategy(),
        "periodic_5000": PeriodicStrategy(period=5000),
        "periodic_10000": PeriodicStrategy(period=10000),
        "drift_triggered": DriftTriggeredStrategy(cooldown=2000),
    }

    all_results: dict[str, dict] = {}

    for name, strategy in strategies.items():
        print(f"\n--- Running: {name} ---")

        initial_model = BaselineIDS(algo, params=model_params, random_seed=seed)
        initial_model.fit(X_train, y_train)

        detector = create_detector(drift_cfg)

        manager = AdaptiveModelManager(
            algorithm=algo,
            model_params=model_params,
            strategy=strategy,
            detector=detector,
            window_size=20000,
            random_seed=seed,
        )
        manager.set_initial_model(initial_model)

        result = run_strategy(name, manager, X_test, y_test, pipeline.feature_columns)
        all_results[name] = result

        f1 = result["overall_metrics"]["f1"]
        recall = result["overall_metrics"]["recall"]
        fpr = result["overall_metrics"].get("fpr", "N/A")
        n_ret = result["n_retrains"]
        cost = result["total_retrain_time_s"]
        print(f"  F1={f1:.4f}  Recall={recall:.4f}  FPR={fpr}  Retrains={n_ret}  Cost={cost}s")

        strat_dir = results_dir / name
        strat_dir.mkdir(parents=True, exist_ok=True)
        with open(strat_dir / "results.json", "w") as fh:
            json.dump(result, fh, indent=2, default=str)

        plot_performance_over_time(
            result["windowed_metrics"],
            metric_name="f1",
            drift_positions=result["retrain_positions"],
            title=f"F1 Over Time — {name}",
            filename=f"f1_adapt_{name}.png",
        )

    # Comparison summary
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<25} {'F1':>8} {'Recall':>8} {'FNR':>8} {'Retrains':>10} {'Cost(s)':>10}")
    print("-" * 75)

    summary_rows = []
    for name, result in all_results.items():
        m = result["overall_metrics"]
        row = {
            "strategy": name,
            "f1": m["f1"],
            "recall": m["recall"],
            "precision": m["precision"],
            "fnr": m.get("fnr", 0),
            "fpr": m.get("fpr", 0),
            "mcc": m["mcc"],
            "accuracy": m["accuracy"],
            "n_retrains": result["n_retrains"],
            "retrain_time_s": result["total_retrain_time_s"],
            "total_time_s": result["total_elapsed_s"],
        }
        summary_rows.append(row)
        print(
            f"{name:<25} {m['f1']:>8.4f} {m['recall']:>8.4f} "
            f"{m.get('fnr', 0):>8.4f} {result['n_retrains']:>10} "
            f"{result['total_retrain_time_s']:>10.1f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "comparison_summary.csv", index=False)

    with open(results_dir / "comparison.json", "w") as fh:
        json.dump(
            {name: {
                "f1": r["overall_metrics"]["f1"],
                "recall": r["overall_metrics"]["recall"],
                "fnr": r["overall_metrics"].get("fnr", 0),
                "fpr": r["overall_metrics"].get("fpr", 0),
                "mcc": r["overall_metrics"]["mcc"],
                "n_retrains": r["n_retrains"],
                "retrain_time_s": r["total_retrain_time_s"],
            } for name, r in all_results.items()},
            fh, indent=2,
        )

    # Combined F1-over-time plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = {"static": "#d62728", "periodic_5000": "#ff7f0e", "periodic_10000": "#2ca02c", "drift_triggered": "#1f77b4"}
    for name, result in all_results.items():
        wm = result["windowed_metrics"]
        windows = [w["window_id"] for w in wm]
        f1s = [w["f1"] for w in wm]
        ax.plot(windows, f1s, linewidth=1.2, label=name, color=colors.get(name, None), alpha=0.85)

    ax.set_xlabel("Window")
    ax.set_ylabel("F1 Score")
    ax.set_title("Adaptation Strategy Comparison — F1 Over Time")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_path = root / "results" / "figures" / "adaptation_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults saved to {results_dir}")
    print(f"Comparison plot: {fig_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
