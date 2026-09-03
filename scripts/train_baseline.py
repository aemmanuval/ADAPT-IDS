#!/usr/bin/env python3
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
"""Train baseline IDS models (LightGBM, Random Forest) with random split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.evaluation.temporal import random_split
from adaptive_ids.evaluation.metrics import (
    compute_metrics, save_metrics, save_confusion_matrix, save_predictions,
)
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.models.registry import ModelRegistry
from adaptive_ids.experiments.tracker import ExperimentTracker
from adaptive_ids.visualization.plots import plot_confusion_matrix, plot_class_distribution
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("train_baseline")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    proc_dir = root / config["dataset"]["processed_dir"]
    results_dir = root / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "cleaned.parquet"
    if not data_path.exists():
        logger.error("Preprocessed data not found at %s. Run preprocess_dataset.py first.", data_path)
        sys.exit(1)

    logger.info("=== Baseline Model Training (Random Split) ===")
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    label_col = config["dataset"]["label_column"]
    plot_class_distribution(df[label_col], filename="class_distribution.png")

    splits = random_split(
        df,
        train_fraction=0.80,
        test_fraction=0.20,
        random_seed=seed,
        stratify_column=label_col,
    )

    pipeline = PreprocessingPipeline(config)
    _, X_train, y_train = pipeline.fit_transform(splits["train"])
    _, X_test, y_test = pipeline.transform(splits["test"])

    logger.info("Train: %d samples, Test: %d samples", X_train.shape[0], X_test.shape[0])
    logger.info("Features: %d", X_train.shape[1])

    registry = ModelRegistry(root / "results" / "models")

    for algo in ["lightgbm", "random_forest"]:
        logger.info("\n--- Training %s ---", algo)
        params = config["models"].get(algo, {})
        model = BaselineIDS(algo, params=params, random_seed=seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        positive = config["classification"].get("positive_label", "ATTACK")
        metrics = compute_metrics(y_test, y_pred, positive_label=positive)

        logger.info("  Precision: %.4f", metrics["precision"])
        logger.info("  Recall:    %.4f", metrics["recall"])
        logger.info("  F1:        %.4f", metrics["f1"])
        logger.info("  MCC:       %.4f", metrics["mcc"])
        if metrics.get("fpr") is not None:
            logger.info("  FPR:       %.4f", metrics["fpr"])
            logger.info("  FNR:       %.4f", metrics["fnr"])

        algo_dir = results_dir / algo
        algo_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, algo_dir / "metrics.json")
        save_predictions(y_test, y_pred, algo_dir / "predictions.csv")
        save_confusion_matrix(y_test, y_pred, algo_dir / "confusion_matrix.csv")

        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm, metrics["confusion_labels"],
            title=f"Confusion Matrix — {algo} (Random Split)",
            filename=f"cm_{algo}_random.png",
        )

        model_path = algo_dir / f"{algo}_baseline.joblib"
        model.save(model_path)
        registry.register(
            model_id=f"{algo}_baseline_v1",
            model_path=model_path,
            metadata={**model.metadata, "split": "random", "metrics": metrics},
        )

    tracker = ExperimentTracker(results_dir)
    tracker.update({
        "experiment_name": "baseline_random_split",
        "seed": seed,
        "split": "random",
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
    })
    tracker.save(filename="baseline_experiment.json")

    logger.info("\n=== Baseline training complete ===")


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
