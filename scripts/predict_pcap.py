#!/usr/bin/env python3
"""Predict on a Wireshark PCAP capture or CICFlowMeter CSV.

Usage:
    # From a CICFlowMeter CSV (recommended — same schema as training data)
    python scripts/predict_pcap.py --cicflow path/to/flowmeter_output.csv

    # From a raw PCAP file (requires tshark installed)
    python scripts/predict_pcap.py --pcap path/to/capture.pcap

    # From a raw PCAP with limited packets
    python scripts/predict_pcap.py --pcap path/to/capture.pcap --max-packets 50000

Outputs predictions + drift alerts to stdout and results/live/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.data.pcap_loader import load_pcap_as_flows, load_cicflowmeter_csv
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.streaming.stream import CSVStream
from adaptive_ids.drift.detectors import create_detector
from adaptive_ids.drift.events import DriftEventLogger
from adaptive_ids.evaluation.metrics import compute_metrics
from adaptive_ids.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAPT-IDS \u2014 Predict on PCAP/Wireshark data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcap", type=str, help="Path to .pcap/.pcapng file")
    group.add_argument("--cicflow", type=str, help="Path to CICFlowMeter CSV output")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model (.joblib)")
    parser.add_argument("--max-packets", type=int, default=None, help="Max packets to process from PCAP")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["logging"]["level"])
    logger = get_logger("predict_pcap")

    root = get_project_root()
    results_dir = root / "results" / "live"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  \u2014  Live Traffic Prediction")
    print("=" * 60 + "\n")

    # Load flows
    if args.pcap:
        print(f"[1] Loading PCAP: {args.pcap}")
        flows_df = load_pcap_as_flows(args.pcap, max_packets=args.max_packets)
    else:
        print(f"[1] Loading CICFlowMeter CSV: {args.cicflow}")
        flows_df = load_cicflowmeter_csv(args.cicflow)

    print(f"    Flows extracted: {len(flows_df):,}")

    if len(flows_df) == 0:
        print("    No flows extracted. Check the input file.")
        sys.exit(1)

    # Load model
    if args.model:
        model_path = Path(args.model)
    else:
        candidates = [
            root / "results" / "temporal" / "lightgbm" / "lightgbm_temporal.joblib",
            root / "results" / "baseline" / "lightgbm" / "lightgbm_baseline.joblib",
        ]
        model_path = None
        for c in candidates:
            if c.exists():
                model_path = c
                break
        if model_path is None:
            print("    No trained model found. Run train_baseline.py or evaluate_temporal.py first.")
            sys.exit(1)

    print(f"[2] Loading model: {model_path.name}")
    model = BaselineIDS.load(model_path)

    # Prepare features
    pipeline = PreprocessingPipeline(config)
    flows_df["Label"] = "UNKNOWN"

    try:
        cleaned, _ = pipeline.clean(flows_df)
        feature_cols = [c for c in model.metadata.get("classes", []) or pipeline.feature_columns
                        if c in cleaned.columns]

        if not feature_cols:
            feature_cols = [c for c in cleaned.columns
                           if c not in {"Label", "Timestamp", "_original_label"}
                           and cleaned[c].dtype in ["float64", "int64", "float32"]]

        X = cleaned[feature_cols].values.astype(np.float64)
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        print(f"    Error during preprocessing: {e}")
        sys.exit(1)

    print(f"    Features: {X.shape[1]} | Flows: {X.shape[0]:,}")

    # Predict
    print("[3] Running predictions...")
    predictions = model.predict(X)

    n_attack = int((predictions == "ATTACK").sum())
    n_benign = int((predictions == "BENIGN").sum())

    print(f"\n    RESULTS:")
    print(f"    Total flows:    {len(predictions):,}")
    print(f"    BENIGN:         {n_benign:,} ({n_benign/len(predictions)*100:.1f}%)")
    print(f"    ATTACK:         {n_attack:,} ({n_attack/len(predictions)*100:.1f}%)")

    # Drift monitoring
    print("\n[4] Monitoring for drift...")
    detector = create_detector(config["drift"])
    event_logger = DriftEventLogger(results_dir)

    if n_attack > 0:
        error_signal = [0.0 if p == "BENIGN" else 1.0 for p in predictions]
    else:
        error_signal = [0.5] * len(predictions)

    for i, val in enumerate(error_signal):
        detector.update(val)
        if detector.drift_detected():
            event_logger.log_drift(detector.name, i)

    if event_logger.count > 0:
        print(f"    Drift events:   {event_logger.count}")
        print(f"    (Distribution shift detected \u2014 model may need retraining)")
        event_logger.save_csv("live_drift_events.csv")
    else:
        print(f"    No drift detected in this capture")

    # Save predictions
    output = pd.DataFrame({"prediction": predictions})
    if "Timestamp" in cleaned.columns:
        output["timestamp"] = cleaned["Timestamp"].values[:len(predictions)]
    output.to_csv(results_dir / "predictions.csv", index=False)

    summary = {
        "input": args.pcap or args.cicflow,
        "total_flows": len(predictions),
        "benign": n_benign,
        "attack": n_attack,
        "attack_pct": round(n_attack / len(predictions) * 100, 2),
        "drift_events": event_logger.count,
        "model": str(model_path),
    }
    with open(results_dir / "prediction_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n    Predictions saved to: {results_dir / 'predictions.csv'}")
    print(f"    Summary saved to:     {results_dir / 'prediction_summary.json'}")

    if n_attack > 0:
        print(f"\n    *** {n_attack} POTENTIAL ATTACKS DETECTED ***")
        print(f"    Review results/live/predictions.csv for details")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
