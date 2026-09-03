#!/usr/bin/env python3
"""Phase 4: Compare LSTM vs LightGBM on both random and temporal evaluation.

Answers: does deep learning improve detection under drift?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.evaluation.temporal import temporal_split, random_split
from adaptive_ids.evaluation.metrics import compute_metrics, compute_windowed_metrics, save_metrics
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.models.lstm_ids import LSTMClassifier
from adaptive_ids.visualization.plots import plot_confusion_matrix, plot_performance_over_time
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def evaluate_model(model, X_test, y_test, positive_label="ATTACK"):
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, positive_label=positive_label)
    windowed = compute_windowed_metrics(y_test, y_pred, window_size=5000, positive_label=positive_label)
    return metrics, windowed, y_pred


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("lstm_comparison")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    proc_dir = root / config["dataset"]["processed_dir"]
    results_dir = root / "results" / "lstm_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "cleaned.parquet"
    if not data_path.exists():
        logger.error("Run preprocess_dataset.py first.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  —  LSTM vs LightGBM Comparison")
    print("=" * 60 + "\n")

    df = pd.read_parquet(data_path)
    ts_col = config["dataset"]["timestamp_column"]
    label_col = config["dataset"]["label_column"]
    eval_cfg = config["evaluation"]

    # Random split
    random_splits = random_split(df, train_fraction=0.8, test_fraction=0.2,
                                  random_seed=seed, stratify_column=label_col)

    # Temporal split
    temporal_splits = temporal_split(
        df, timestamp_column=ts_col,
        train_fraction=eval_cfg["train_fraction"],
        validation_fraction=eval_cfg["validation_fraction"],
        test_fraction=eval_cfg["test_fraction"],
    )

    pipeline = PreprocessingPipeline(config)

    all_results = {}

    for split_name, splits in [("random", random_splits), ("temporal", temporal_splits)]:
        print(f"\n{'='*40}")
        print(f"  Split: {split_name}")
        print(f"{'='*40}")

        _, X_train, y_train = pipeline.fit_transform(splits["train"])
        _, X_test, y_test = pipeline.transform(splits["test"])

        print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Features: {X_train.shape[1]}")

        models = {
            "lightgbm": BaselineIDS("lightgbm", params=config["models"].get("lightgbm", {}), random_seed=seed),
            "lstm": LSTMClassifier(
                hidden_size=128, num_layers=2, dropout=0.3,
                learning_rate=0.001, epochs=15, batch_size=1024,
                random_seed=seed,
            ),
        }

        for model_name, model in models.items():
            print(f"\n--- {model_name} ({split_name} split) ---")
            model.fit(X_train, y_train)

            metrics, windowed, y_pred = evaluate_model(model, X_test, y_test)

            print(f"  F1:     {metrics['f1']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            if metrics.get("fnr") is not None:
                print(f"  FNR:    {metrics['fnr']:.4f}")
            print(f"  MCC:    {metrics['mcc']:.4f}")
            print(f"  Time:   {model.training_time:.1f}s")

            key = f"{model_name}_{split_name}"
            all_results[key] = {
                "model": model_name,
                "split": split_name,
                "f1": metrics["f1"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "fnr": metrics.get("fnr", 0),
                "fpr": metrics.get("fpr", 0),
                "mcc": metrics["mcc"],
                "training_time_s": round(model.training_time, 2),
            }

            model_dir = results_dir / key
            model_dir.mkdir(parents=True, exist_ok=True)
            save_metrics(metrics, model_dir / "metrics.json")

            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(cm, metrics["confusion_labels"],
                                  title=f"{model_name} — {split_name} split",
                                  filename=f"cm_{key}.png")

            plot_performance_over_time(
                windowed, metric_name="f1",
                title=f"F1 Over Time — {model_name} ({split_name})",
                filename=f"f1_time_{key}.png",
            )

    # Summary
    print("\n" + "=" * 60)
    print("  LSTM vs LightGBM COMPARISON")
    print("=" * 60)
    print(f"{'Model':<12} {'Split':<10} {'F1':>8} {'Recall':>8} {'FNR':>8} {'MCC':>8} {'Time(s)':>10}")
    print("-" * 70)

    for key, r in all_results.items():
        print(f"{r['model']:<12} {r['split']:<10} {r['f1']:>8.4f} {r['recall']:>8.4f} "
              f"{r['fnr']:>8.4f} {r['mcc']:>8.4f} {r['training_time_s']:>10.1f}")

    with open(results_dir / "comparison.json", "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    summary_df = pd.DataFrame(all_results.values())
    summary_df.to_csv(results_dir / "comparison_summary.csv", index=False)

    print(f"\nResults saved to {results_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
