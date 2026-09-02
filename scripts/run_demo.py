#!/usr/bin/env python3
"""ADAPT-IDS First-Milestone Demo

Single command that demonstrates the full Phase 1 pipeline:
  1. Verify data availability
  2. Load & preprocess dataset
  3. Train baseline LightGBM model (temporal split)
  4. Stream test traffic in chronological order
  5. Run ADWIN drift detection
  6. Report results and generate plots

Usage:
    python scripts/run_demo.py                    # full dataset
    python scripts/run_demo.py --max-rows 50000   # quick demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.data.loader import load_dataset, inspect_dataset, discover_csv_files
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.evaluation.temporal import temporal_split, get_temporal_info
from adaptive_ids.evaluation.metrics import (
    compute_metrics, compute_windowed_metrics, save_metrics,
)
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.streaming.stream import CSVStream
from adaptive_ids.drift.detectors import create_detector
from adaptive_ids.drift.events import DriftEventLogger
from adaptive_ids.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_performance_over_time,
    plot_error_rate_with_drift,
    plot_traffic_over_time,
)
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAPT-IDS Demo")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit dataset rows")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides: dict = {}
    if args.max_rows:
        overrides.setdefault("dataset", {})["max_rows"] = args.max_rows
    if args.sample:
        overrides.setdefault("dataset", {})["sample_fraction"] = args.sample
    if args.seed:
        overrides.setdefault("experiment", {})["random_seed"] = args.seed

    config = load_config(args.config, overrides=overrides if overrides else None)
    setup_logging(config["logging"]["level"])
    logger = get_logger("demo")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    raw_dir = root / config["dataset"]["raw_dir"]

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  —  First Milestone Demo")
    print("  Adaptive Intrusion Detection Under Concept & Feature Drift")
    print("=" * 60 + "\n")

    # ── Step 1: Verify data ──────────────────────────────────────────
    print("[1/7] Verifying dataset...")
    try:
        csv_files = discover_csv_files(raw_dir)
        print(f"  Found {len(csv_files)} CSV file(s) in {raw_dir}")
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        print("\n  To run this demo, download CIC-IDS2017 CSVs into data/raw/")
        print("  See: data/README.md for instructions")
        sys.exit(1)

    # ── Step 2: Load & preprocess ────────────────────────────────────
    print("[2/7] Loading and preprocessing...")
    df = load_dataset(
        raw_dir,
        max_rows=config["dataset"].get("max_rows"),
        sample_fraction=config["dataset"].get("sample_fraction", 1.0),
        random_seed=seed,
    )
    profile = inspect_dataset(df)
    print(f"  Rows: {profile['rows']:,}  |  Columns: {profile['columns']}")
    if profile.get("label_column"):
        print(f"  Labels: {profile['unique_labels']} classes")
        for lbl, cnt in list(profile["class_distribution"].items())[:8]:
            print(f"    {lbl:<25s} {cnt:>10,}  ({profile['class_percentages'][lbl]:.1f}%)")

    pipeline = PreprocessingPipeline(config)
    cleaned, report = pipeline.clean(df)
    print(f"  After cleaning: {report.rows_after:,} rows, {report.columns_after} columns")

    # ── Step 3: Temporal split ───────────────────────────────────────
    print("[3/7] Temporal split...")
    ts_col = config["dataset"]["timestamp_column"]
    label_col = config["dataset"]["label_column"]
    eval_cfg = config["evaluation"]

    ts_info = get_temporal_info(cleaned, ts_col)
    if ts_info.get("available"):
        print(f"  Time range: {ts_info['min']} → {ts_info['max']}")
        plot_traffic_over_time(cleaned[ts_col], filename="demo_traffic.png")

    splits = temporal_split(
        cleaned,
        timestamp_column=ts_col,
        train_fraction=eval_cfg["train_fraction"],
        validation_fraction=eval_cfg["validation_fraction"],
        test_fraction=eval_cfg["test_fraction"],
    )
    print(f"  Train: {len(splits['train']):,} | Val: {len(splits['validation']):,} | Test: {len(splits['test']):,}")

    _, X_train, y_train = pipeline.fit_transform(splits["train"])
    _, X_test, y_test = pipeline.transform(splits["test"])

    plot_class_distribution(y_train, title="Training Set Class Distribution", filename="demo_class_dist.png")

    # ── Step 4: Train baseline ───────────────────────────────────────
    print("[4/7] Training LightGBM baseline...")
    algo = "lightgbm"
    params = config["models"].get(algo, {})
    model = BaselineIDS(algo, params=params, random_seed=seed)
    model.fit(X_train, y_train)
    print(f"  Training time: {model.training_time:.2f}s")

    y_pred = model.predict(X_test)
    positive = config["classification"].get("positive_label", "ATTACK")
    metrics = compute_metrics(y_test, y_pred, positive_label=positive)
    print(f"  Temporal test F1:     {metrics['f1']:.4f}")
    print(f"  Temporal test Recall: {metrics['recall']:.4f}")
    if metrics.get("fpr") is not None:
        print(f"  Temporal test FPR:    {metrics['fpr']:.4f}")

    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, metrics["confusion_labels"],
                          title="Demo — Temporal Confusion Matrix",
                          filename="demo_confusion_matrix.png")

    # ── Step 5: Windowed performance ─────────────────────────────────
    print("[5/7] Computing windowed performance...")
    window_size = eval_cfg.get("window_size", 5000)
    windowed = compute_windowed_metrics(y_test, y_pred, window_size=window_size, positive_label=positive)
    f1_values = [w["f1"] for w in windowed]
    print(f"  Windows: {len(windowed)}")
    print(f"  F1 range: {min(f1_values):.4f} – {max(f1_values):.4f}")

    # ── Step 6: ADWIN drift detection ────────────────────────────────
    print("[6/7] Running ADWIN drift detection...")
    detector = create_detector(config["drift"])
    event_logger = DriftEventLogger(root / "results" / "drift")

    stream = CSVStream(X_test, y_test, feature_names=pipeline.feature_columns)
    errors: list[int] = []
    all_preds: list[str] = []
    drift_positions: list[int] = []

    for event in tqdm(stream, total=len(stream), desc="  Streaming", ncols=80):
        x = event["features"]
        y_true_i = event["label"]
        idx = event["index"]

        y_pred_i = model.predict_single(x)
        all_preds.append(y_pred_i)
        error = 0 if y_pred_i == y_true_i else 1
        errors.append(error)

        detector.update(float(error))
        if detector.drift_detected():
            rolling = np.mean(errors[-500:]) if len(errors) >= 500 else np.mean(errors)
            event_logger.log_drift(
                detector_name=detector.name,
                stream_position=idx,
                metric_context={"rolling_error_500": round(float(rolling), 4)},
                detector_state=detector.get_state(),
            )
            drift_positions.append(idx)

    event_logger.save_csv()
    event_logger.save_json()

    print(f"  Drift events detected: {event_logger.count}")
    for ev in event_logger.events[:10]:
        print(f"    #{ev.event_id} at position {ev.stream_position}")
    if event_logger.count > 10:
        print(f"    ... and {event_logger.count - 10} more")

    # ── Step 7: Visualizations ───────────────────────────────────────
    print("[7/7] Generating visualizations...")
    plot_performance_over_time(
        windowed,
        drift_positions=drift_positions,
        title="Demo — F1 Over Time with Drift Events",
        filename="demo_f1_drift.png",
    )
    plot_error_rate_with_drift(
        [float(e) for e in errors],
        drift_positions,
        window_size=min(500, max(len(errors) // 20, 10)),
        title="Demo — Error Rate with Drift",
        filename="demo_error_rate.png",
    )

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print(f"  Dataset:        {profile['rows']:,} flows")
    print(f"  Model:          LightGBM")
    print(f"  Temporal F1:    {metrics['f1']:.4f}")
    print(f"  F1 range:       {min(f1_values):.4f} – {max(f1_values):.4f}")
    print(f"  Drift events:   {event_logger.count}")
    print(f"  Figures saved:  results/figures/")
    print(f"  Drift log:      results/drift/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
