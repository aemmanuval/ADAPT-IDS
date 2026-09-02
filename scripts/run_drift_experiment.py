#!/usr/bin/env python3
"""Run ADWIN drift-detection experiment on temporally ordered test traffic.

Pipeline:
  1. Load trained model + temporal test set
  2. Stream test samples in chronological order
  3. Generate predictions → compute per-sample error (0/1)
  4. Feed error signal into ADWIN
  5. Log drift events + produce performance-over-time plots
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
from adaptive_ids.evaluation.metrics import compute_metrics, compute_windowed_metrics, save_metrics
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.streaming.stream import CSVStream
from adaptive_ids.drift.detectors import create_detector
from adaptive_ids.drift.events import DriftEventLogger
from adaptive_ids.experiments.tracker import ExperimentTracker
from adaptive_ids.visualization.plots import (
    plot_performance_over_time, plot_error_rate_with_drift,
)
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("drift_experiment")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    proc_dir = root / config["dataset"]["processed_dir"]
    results_dir = root / "results" / "drift"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = proc_dir / "cleaned.parquet"
    if not data_path.exists():
        logger.error("Run preprocess_dataset.py first.")
        sys.exit(1)

    logger.info("=== ADWIN Drift Detection Experiment ===")
    df = pd.read_parquet(data_path)

    ts_col = config["dataset"]["timestamp_column"]
    label_col = config["dataset"]["label_column"]
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

    algo = "lightgbm"
    model_path = root / "results" / "temporal" / algo / f"{algo}_temporal.joblib"
    if model_path.exists():
        logger.info("Loading pre-trained model from %s", model_path)
        model = BaselineIDS.load(model_path)
    else:
        logger.info("No pre-trained model found — training fresh %s", algo)
        params = config["models"].get(algo, {})
        model = BaselineIDS(algo, params=params, random_seed=seed)
        model.fit(X_train, y_train)

    detector = create_detector(config["drift"])
    event_logger = DriftEventLogger(results_dir)

    logger.info("Streaming %d test samples through ADWIN...", X_test.shape[0])
    stream = CSVStream(X_test, y_test, feature_names=pipeline.feature_columns)

    errors: list[int] = []
    all_preds: list[str] = []
    drift_positions: list[int] = []
    t0 = time.perf_counter()

    for event in tqdm(stream, total=len(stream), desc="Streaming", ncols=80):
        x = event["features"]
        y_true = event["label"]
        idx = event["index"]

        y_pred = model.predict_single(x)
        all_preds.append(y_pred)

        error = 0 if y_pred == y_true else 1
        errors.append(error)

        detector.update(float(error))

        if detector.drift_detected():
            rolling_err = np.mean(errors[-500:]) if len(errors) >= 500 else np.mean(errors)
            event_logger.log_drift(
                detector_name=detector.name,
                stream_position=idx,
                model_version="baseline_v1",
                window_size=detector.get_state().get("window_size", 0),
                metric_context={
                    "rolling_error_500": round(float(rolling_err), 4),
                    "cumulative_error": round(float(np.mean(errors)), 4),
                },
                detector_state=detector.get_state(),
            )
            drift_positions.append(idx)

    elapsed = time.perf_counter() - t0
    logger.info("Streaming completed in %.1fs", elapsed)
    logger.info("Total drift events: %d", event_logger.count)

    event_logger.save_csv()
    event_logger.save_json()

    positive = config["classification"].get("positive_label", "ATTACK")
    overall_metrics = compute_metrics(y_test, np.array(all_preds), positive_label=positive)
    save_metrics(overall_metrics, results_dir / "overall_metrics.json")

    window_size = eval_cfg.get("window_size", 5000)
    windowed = compute_windowed_metrics(
        y_test, np.array(all_preds),
        window_size=window_size,
        positive_label=positive,
    )
    with open(results_dir / "windowed_metrics.json", "w") as fh:
        json.dump(windowed, fh, indent=2, default=str)

    plot_performance_over_time(
        windowed,
        metric_name="f1",
        drift_positions=drift_positions,
        title="F1 Over Time with ADWIN Drift Events",
        filename="f1_over_time_drift.png",
    )

    plot_error_rate_with_drift(
        [float(e) for e in errors],
        drift_positions,
        window_size=min(500, len(errors) // 10 + 1),
        title="Error Rate with ADWIN Drift Detections",
        filename="error_rate_drift.png",
    )

    tracker = ExperimentTracker(results_dir)
    tracker.update({
        "experiment_name": "adwin_drift_detection",
        "seed": seed,
        "model": algo,
        "detector": detector.name,
        "detector_config": config["drift"].get("adwin", {}),
        "n_test_samples": int(X_test.shape[0]),
        "n_drift_events": event_logger.count,
        "drift_positions": drift_positions,
        "overall_metrics": overall_metrics,
        "elapsed_seconds": round(elapsed, 2),
    })
    tracker.save(filename="drift_experiment.json")

    logger.info("\n=== Drift Experiment Summary ===")
    logger.info("Test samples:    %d", X_test.shape[0])
    logger.info("Overall F1:      %.4f", overall_metrics["f1"])
    logger.info("Overall Recall:  %.4f", overall_metrics["recall"])
    logger.info("Drift events:    %d", event_logger.count)
    for ev in event_logger.events:
        logger.info("  Drift #%d at position %d", ev.event_id, ev.stream_position)
    logger.info("Results saved to %s", results_dir)


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
