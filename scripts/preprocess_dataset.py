#!/usr/bin/env python3
"""Preprocess CIC-IDS2017: clean, encode labels, save processed data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.data.loader import load_dataset, compute_dataset_hash
from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
from adaptive_ids.utils.logging import setup_logging, get_logger
from adaptive_ids.utils.reproducibility import set_global_seed


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("preprocess")
    seed = config["experiment"]["random_seed"]
    set_global_seed(seed)

    root = get_project_root()
    raw_dir = root / config["dataset"]["raw_dir"]
    proc_dir = root / config["dataset"]["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Preprocessing Pipeline ===")

    df = load_dataset(
        raw_dir,
        max_rows=config["dataset"].get("max_rows"),
        sample_fraction=config["dataset"].get("sample_fraction", 1.0),
        random_seed=seed,
    )
    data_hash = compute_dataset_hash(df)
    logger.info("Dataset hash: %s", data_hash)

    pipeline = PreprocessingPipeline(config)
    cleaned, report = pipeline.clean(df)
    report.feature_columns = [
        c for c in cleaned.columns
        if c not in {config["dataset"]["label_column"], config["dataset"]["timestamp_column"]}
    ]

    cleaned.to_parquet(proc_dir / "cleaned.parquet", index=False)
    logger.info("Saved cleaned data: %d rows, %d cols → %s",
                len(cleaned), len(cleaned.columns), proc_dir / "cleaned.parquet")

    report.save(proc_dir / "preprocessing_report.json")

    summary = {
        "dataset_hash": data_hash,
        "rows_before": report.rows_before,
        "rows_after": report.rows_after,
        "columns_before": report.columns_before,
        "columns_after": report.columns_after,
        "feature_columns": report.feature_columns,
        "label_mapping": report.label_mapping,
        "seed": seed,
    }
    with open(proc_dir / "preprocessing_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("=== Preprocessing complete ===")
    logger.info("Rows: %d → %d", report.rows_before, report.rows_after)
    logger.info("Columns: %d → %d", report.columns_before, report.columns_after)
    logger.info("Duplicates removed: %d", report.rows_dropped_duplicate)
    logger.info("Features: %d", len(report.feature_columns))


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
