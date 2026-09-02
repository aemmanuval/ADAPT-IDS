"""Multi-dataset support for cross-dataset evaluation.

Handles schema differences between:
  - CIC-IDS2017 (primary)
  - CSE-CIC-IDS2018 (same CICFlowMeter format, direct compatibility)
  - UNSW-NB15 (different schema, requires feature mapping)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_ids.utils.logging import get_logger

logger = get_logger("data.multi_dataset")

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "cic_ids_2017": {
        "name": "CIC-IDS2017",
        "label_column": "Label",
        "timestamp_column": "Timestamp",
        "benign_labels": ["BENIGN"],
        "encoding": "utf-8",
    },
    "cse_cic_ids_2018": {
        "name": "CSE-CIC-IDS2018",
        "label_column": "Label",
        "timestamp_column": "Timestamp",
        "benign_labels": ["Benign"],
        "encoding": "utf-8",
    },
    "unsw_nb15": {
        "name": "UNSW-NB15",
        "label_column": "label",
        "attack_cat_column": "attack_cat",
        "timestamp_column": None,
        "benign_labels": [0, "0", "Normal", "normal"],
        "encoding": "utf-8",
        "feature_mapping": {
            "dur": "Flow Duration",
            "spkts": "Total Fwd Packets",
            "dpkts": "Total Backward Packets",
            "sbytes": "Total Length of Fwd Packets",
            "dbytes": "Total Length of Bwd Packets",
            "sttl": "ip.ttl",
            "swin": "Init_Win_bytes_forward",
            "dwin": "Init_Win_bytes_backward",
            "smean": "Fwd Packet Length Mean",
            "dmean": "Bwd Packet Length Mean",
            "sinpkt": "Flow IAT Mean",
            "dinpkt": "Bwd IAT Mean",
            "sjit": "Flow IAT Std",
            "djit": "Bwd IAT Std",
            "ct_srv_src": "ct_srv_src",
            "ct_srv_dst": "ct_srv_dst",
        },
    },
}


def load_multi_dataset(
    raw_dir: str | Path,
    dataset_name: str,
    *,
    max_rows: int | None = None,
    sample_fraction: float = 1.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a dataset by name, normalising labels and column names.

    Returns (dataframe, dataset_config).
    """
    raw_dir = Path(raw_dir)
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    logger.info("Loading dataset: %s from %s", config["name"], raw_dir)

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir} for dataset {config['name']}"
        )

    frames = []
    total = 0
    for csv_path in csvs:
        remaining = None
        if max_rows is not None:
            remaining = max_rows - total
            if remaining <= 0:
                break

        df = pd.read_csv(
            csv_path,
            nrows=remaining,
            encoding=config.get("encoding", "utf-8"),
            low_memory=False,
        )
        df.columns = df.columns.str.strip()
        frames.append(df)
        total += len(df)

    combined = pd.concat(frames, ignore_index=True)

    if sample_fraction < 1.0:
        combined = combined.sample(frac=sample_fraction, random_state=random_seed)

    if "feature_mapping" in config:
        rename_map = {
            k: v for k, v in config["feature_mapping"].items()
            if k in combined.columns
        }
        combined = combined.rename(columns=rename_map)
        logger.info("Mapped %d columns for %s", len(rename_map), config["name"])

    label_col = config["label_column"]
    if label_col in combined.columns:
        combined[label_col] = combined[label_col].astype(str).str.strip()
        benign = config["benign_labels"]
        combined["Label"] = combined[label_col].apply(
            lambda x: "BENIGN" if x in [str(b) for b in benign] else "ATTACK"
        )
        if label_col != "Label":
            combined["_original_label"] = combined[label_col]

    logger.info(
        "Loaded %s: %d rows, %d columns",
        config["name"], len(combined), len(combined.columns),
    )
    return combined, config


def get_common_features(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    exclude: set[str] | None = None,
) -> list[str]:
    """Find numeric features common to both DataFrames."""
    exclude = exclude or {"Label", "Timestamp", "_original_label"}
    num_a = set(df_a.select_dtypes(include="number").columns) - exclude
    num_b = set(df_b.select_dtypes(include="number").columns) - exclude
    common = sorted(num_a & num_b)
    logger.info("Common numeric features: %d", len(common))
    return common
