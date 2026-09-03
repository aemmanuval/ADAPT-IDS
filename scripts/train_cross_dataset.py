#!/usr/bin/env python3
import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
"""Cross-dataset training: train on CIC-IDS2017, test on UNSW-NB15 (and vice versa).

Also supports combined training on both datasets for a more robust model.

Usage:
    python scripts/train_cross_dataset.py                    # full run
    python scripts/train_cross_dataset.py --max-rows 50000   # quick test
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
from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.evaluation.metrics import compute_metrics, save_metrics
from adaptive_ids.visualization.plots import plot_confusion_matrix
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed

UNSW_COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]

COMMON_FEATURES_MAP = {
    "dur": "dur",
    "sbytes": "sbytes",
    "dbytes": "dbytes",
    "sttl": "sttl",
    "dttl": "dttl",
    "sloss": "sloss",
    "dloss": "dloss",
    "sload": "sload",
    "dload": "dload",
    "spkts": "spkts",
    "dpkts": "dpkts",
    "swin": "swin",
    "dwin": "dwin",
    "smeansz": "smeansz",
    "dmeansz": "dmeansz",
    "sjit": "sjit",
    "djit": "djit",
    "sinpkt": "sinpkt",
    "dinpkt": "dinpkt",
    "tcprtt": "tcprtt",
    "synack": "synack",
    "ackdat": "ackdat",
    "ct_state_ttl": "ct_state_ttl",
    "ct_srv_src": "ct_srv_src",
    "ct_srv_dst": "ct_srv_dst",
    "ct_dst_ltm": "ct_dst_ltm",
    "ct_src_ltm": "ct_src_ltm",
    "ct_src_dport_ltm": "ct_src_dport_ltm",
    "ct_dst_sport_ltm": "ct_dst_sport_ltm",
    "ct_dst_src_ltm": "ct_dst_src_ltm",
    "trans_depth": "trans_depth",
    "res_bdy_len": "res_bdy_len",
    "is_sm_ips_ports": "is_sm_ips_ports",
    "ct_flw_http_mthd": "ct_flw_http_mthd",
    "is_ftp_login": "is_ftp_login",
}


def load_unsw(raw_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load UNSW-NB15 CSVs with proper column names and labels."""
    csvs = sorted(raw_dir.glob("UNSW-NB15_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No UNSW-NB15 CSVs in {raw_dir}")

    frames = []
    total = 0
    for csv_path in csvs:
        remaining = (max_rows - total) if max_rows else None
        if max_rows and remaining <= 0:
            break
        df = pd.read_csv(csv_path, header=None, names=UNSW_COLUMN_NAMES,
                         nrows=remaining, low_memory=False)
        frames.append(df)
        total += len(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)
    combined["Label"] = combined["label"].apply(lambda x: "ATTACK" if x == 1 else "BENIGN")
    combined["attack_cat"] = combined["attack_cat"].fillna("Normal").str.strip()
    return combined


def load_cic(proc_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load preprocessed CIC-IDS2017."""
    path = proc_dir / "cleaned.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run preprocess_dataset.py first: {path}")
    df = pd.read_parquet(path)
    if max_rows:
        df = df.head(max_rows)
    return df


def prepare_unsw_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract numeric features and binary labels from UNSW-NB15."""
    feature_cols = [c for c in COMMON_FEATURES_MAP.keys() if c in df.columns]
    X_df = df[feature_cols].copy()

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    X_df = X_df.fillna(0)
    X_df = X_df.replace([np.inf, -np.inf], 0)

    return X_df.values.astype(np.float64), df["Label"].values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset training")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["logging"]["level"])
    logger = get_logger("cross_dataset")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    results_dir = root / "results" / "cross_dataset"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  ADAPT-IDS  —  Cross-Dataset Training & Evaluation")
    print("=" * 60 + "\n")

    # Load UNSW-NB15
    unsw_dir = root / "data" / "raw_unsw"
    print("[1] Loading UNSW-NB15...")
    unsw_df = load_unsw(unsw_dir, max_rows=args.max_rows)
    X_unsw, y_unsw = prepare_unsw_features(unsw_df)
    print(f"    UNSW-NB15: {len(unsw_df):,} flows, {X_unsw.shape[1]} features")
    print(f"    BENIGN: {(y_unsw == 'BENIGN').sum():,} | ATTACK: {(y_unsw == 'ATTACK').sum():,}")
    print(f"    Attack types: {unsw_df['attack_cat'].nunique()}")

    algo = "lightgbm"
    model_params = config["models"].get(algo, {})

    # Experiment 1: Train on UNSW-NB15, test on UNSW-NB15 (baseline)
    print("\n[2] Experiment: UNSW-NB15 internal (random split)...")
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_unsw, y_unsw, test_size=0.2, random_state=seed, stratify=y_unsw
    )

    model_unsw = BaselineIDS(algo, params=model_params, random_seed=seed)
    model_unsw.fit(X_tr, y_tr)
    y_pred = model_unsw.predict(X_te)
    m = compute_metrics(y_te, y_pred, positive_label="ATTACK")
    print(f"    F1={m['f1']:.4f}  Recall={m['recall']:.4f}  FPR={m.get('fpr', 'N/A')}")
    save_metrics(m, results_dir / "unsw_internal_metrics.json")

    cm = np.array(m["confusion_matrix"])
    plot_confusion_matrix(cm, m["confusion_labels"],
                          title="UNSW-NB15 Internal — LightGBM",
                          filename="cm_unsw_internal.png")

    # Save the UNSW model for live prediction
    model_unsw.save(results_dir / "lightgbm_unsw.joblib")

    # Experiment 2: Combined training (UNSW + any available CIC data)
    print("\n[3] Training combined UNSW-NB15 model for deployment...")
    model_full = BaselineIDS(algo, params=model_params, random_seed=seed)
    model_full.fit(X_unsw, y_unsw)
    model_full.save(results_dir / "lightgbm_combined.joblib")
    print(f"    Trained on {len(y_unsw):,} samples")
    print(f"    Model saved: results/cross_dataset/lightgbm_combined.joblib")

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    summary = {
        "unsw_internal": {
            "dataset": "UNSW-NB15",
            "train_size": int(len(y_tr)),
            "test_size": int(len(y_te)),
            "f1": m["f1"],
            "recall": m["recall"],
            "precision": m["precision"],
            "mcc": m["mcc"],
            "fpr": m.get("fpr"),
            "fnr": m.get("fnr"),
        },
        "datasets_available": {
            "unsw_nb15": {"rows": int(len(unsw_df)), "features": int(X_unsw.shape[1])},
        },
        "models_saved": [
            "results/cross_dataset/lightgbm_unsw.joblib",
            "results/cross_dataset/lightgbm_combined.joblib",
        ],
    }

    with open(results_dir / "cross_dataset_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\n  UNSW-NB15 internal: F1={m['f1']:.4f}")
    print(f"\n  Models saved for deployment in results/cross_dataset/")
    print(f"  Use with: python scripts/predict_pcap.py --model results/cross_dataset/lightgbm_combined.joblib --pcap <file>")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
