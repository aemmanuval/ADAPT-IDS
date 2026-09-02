#!/usr/bin/env python3
"""Inspect CIC-IDS2017 dataset: columns, labels, timestamps, quality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config, get_project_root
from adaptive_ids.data.loader import discover_csv_files, load_dataset, inspect_dataset
from adaptive_ids.utils.logging import setup_logging, get_logger


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    setup_logging(config["logging"]["level"])
    logger = get_logger("inspect")

    root = get_project_root()
    raw_dir = root / config["dataset"]["raw_dir"]

    logger.info("=== CIC-IDS2017 Dataset Inspection ===")
    logger.info("Raw directory: %s", raw_dir)

    try:
        csv_files = discover_csv_files(raw_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Found %d CSV files:", len(csv_files))
    for f in csv_files:
        size_mb = f.stat().st_size / 1e6
        logger.info("  %s (%.1f MB)", f.name, size_mb)

    max_rows = config["dataset"].get("max_rows")
    sample_frac = config["dataset"].get("sample_fraction", 1.0)

    df = load_dataset(
        raw_dir,
        max_rows=max_rows,
        sample_fraction=sample_frac,
        random_seed=config["experiment"]["random_seed"],
    )

    profile = inspect_dataset(df)
    profile["config_used"] = {
        "max_rows": max_rows,
        "sample_fraction": sample_frac,
    }

    logger.info("\n=== Dataset Profile ===")
    logger.info("Rows: %d", profile["rows"])
    logger.info("Columns: %d", profile["columns"])
    logger.info("Missing values: %d", profile["total_missing"])
    logger.info("Infinite values: %d", profile.get("total_infinite", 0))
    logger.info("Duplicate rows: %d", profile["duplicate_rows"])
    logger.info("Memory: %.1f MB", profile["memory_mb"])

    if profile.get("label_column"):
        logger.info("\nLabel column: %s", profile["label_column"])
        logger.info("Unique labels: %d", profile["unique_labels"])
        logger.info("Class distribution:")
        for label, count in profile["class_distribution"].items():
            pct = profile["class_percentages"][label]
            logger.info("  %-25s %8d  (%.2f%%)", label, count, pct)

    if profile.get("timestamp_column"):
        logger.info("\nTimestamp column: %s", profile["timestamp_column"])
        logger.info("Range: %s → %s", profile["timestamp_min"], profile["timestamp_max"])

    if profile.get("constant_columns"):
        logger.info("\nConstant/near-constant columns: %s", profile["constant_columns"])

    if profile.get("potential_id_leakage_columns"):
        logger.info("Potential ID/leakage columns: %s", profile["potential_id_leakage_columns"])

    out_path = root / config["dataset"]["processed_dir"] / "dataset_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(profile, fh, indent=2, default=str)
    logger.info("\nProfile saved to %s", out_path)


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)
