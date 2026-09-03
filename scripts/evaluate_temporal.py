#!/usr/bin/env python3
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
"""Temporal (chronological) evaluation — train on earlier, test on later traffic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.evaluation.temporal import temporal_split, get_temporal_info
from adaptive_ids.evaluation.metrics import (
    compute_metrics, compute_windowed_metrics,
    save_metrics, save_confusion_matrix, save_predictions,
)
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.models.registry import ModelRegistry
from adaptive_ids.experiments.tracker import ExperimentTracker
from adaptive_ids.visualization.plots import (
    plot_confusion_matrix, plot_performance_over_time, plot_traffic_over_time,
)
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("temporal_eval")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    proc_dir = root / config["dataset"]["processed_dir"]
    results_dir = root / "results" / "temporal"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "cleaned.parquet"
    if not data_path.exists():
        logger.error("Run preprocess_dataset.py first.")
        sys.exit(1)

    logger.info("=== Temporal Evaluation ===")
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows", len(df))

    ts_col = config["dataset"]["timestamp_column"]
    label_col = config["dataset"]["label_column"]

    ts_info = get_temporal_info(df, ts_col)
    logger.info("Temporal info: %s", json.dumps(ts_info, indent=2))
    with open(results_dir / "temporal_info.json", "w") as fh:
        json.dump(ts_info, fh, indent=2)

    if ts_col in df.columns:
        plot_traffic_over_time(df[ts_col], filename="traffic_over_time.png")

    eval_cfg = config["evaluation"]
    splits = temporal_split(
        df,
        timestamp_column=ts_col,
        train_fraction=eval_cfg["train_fraction"],
        validation_fraction=eval_cfg["validation_fraction"],
        test_fraction=eval_cfg["test_fraction"],
    )

    logger.info("Train: %d | Validation: %d | Test: %d",
                len(splits["train"]), len(splits["validation"]), len(splits["test"]))

    pipeline = PreprocessingPipeline(config)
    _, X_train, y_train = pipeline.fit_transform(splits["train"])
    _, X_test, y_test = pipeline.transform(splits["test"])

    logger.info("Features: %d", X_train.shape[1])

    for algo in ["lightgbm", "random_forest"]:
        logger.info("\n--- Temporal eval: %s ---", algo)
        params = config["models"].get(algo, {})
        model = BaselineIDS(algo, params=params, random_seed=seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        positive = config["classification"].get("positive_label", "ATTACK")
        metrics = compute_metrics(y_test, y_pred, positive_label=positive)

        logger.info("  F1:     %.4f", metrics["f1"])
        logger.info("  Recall: %.4f", metrics["recall"])
        logger.info("  MCC:    %.4f", metrics["mcc"])
        if metrics.get("fpr") is not None:
            logger.info("  FPR:    %.4f", metrics["fpr"])

        algo_dir = results_dir / algo
        algo_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, algo_dir / "metrics.json")
        save_predictions(y_test, y_pred, algo_dir / "predictions.csv")
        save_confusion_matrix(y_test, y_pred, algo_dir / "confusion_matrix.csv")

        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm, metrics["confusion_labels"],
            title=f"Confusion Matrix — {algo} (Temporal Split)",
            filename=f"cm_{algo}_temporal.png",
        )

        window_size = eval_cfg.get("window_size", 5000)
        windowed = compute_windowed_metrics(
            y_test, y_pred, window_size=window_size, positive_label=positive,
        )
        with open(algo_dir / "windowed_metrics.json", "w") as fh:
            json.dump(windowed, fh, indent=2, default=str)

        plot_performance_over_time(
            windowed, metric_name="f1",
            title=f"F1 Over Time — {algo} (Temporal)",
            filename=f"f1_over_time_{algo}_temporal.png",
        )

        model_path = algo_dir / f"{algo}_temporal.joblib"
        model.save(model_path)

    tracker = ExperimentTracker(results_dir)
    tracker.update({
        "experiment_name": "temporal_evaluation",
        "seed": seed,
        "split": "temporal",
        "train_fraction": eval_cfg["train_fraction"],
        "test_fraction": eval_cfg["test_fraction"],
        "temporal_info": ts_info,
    })
    tracker.save(filename="temporal_experiment.json")

    logger.info("\n=== Temporal evaluation complete ===")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
