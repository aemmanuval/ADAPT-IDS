"""CIC-IDS2017 dataset loader with chunked-read and sampling support."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_ids.utils.logging import get_logger

logger = get_logger("data.loader")

KNOWN_LABEL_COLUMNS = ["Label", " Label"]
KNOWN_TIMESTAMP_COLUMNS = ["Timestamp", " Timestamp"]


def discover_csv_files(raw_dir: str | Path) -> list[Path]:
    """Return sorted list of CSV files under *raw_dir*."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {raw_dir}\n"
            f"Download CIC-IDS2017 CSVs from:\n"
            f"  https://www.unb.ca/cic/datasets/ids-2017.html\n"
            f"Place the CSV files in: {raw_dir}"
        )
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            f"Download CIC-IDS2017 and place CSV files there."
        )
    logger.info("Found %d CSV file(s) in %s", len(csvs), raw_dir)
    return csvs


def load_single_csv(
    path: Path,
    *,
    max_rows: int | None = None,
    sample_fraction: float = 1.0,
    chunk_size: int = 100_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load one CSV with optional row-limit or sampling.

    CIC-IDS2017 CSVs have inconsistent whitespace in headers;
    this function strips them.
    """
    logger.info("Loading %s (max_rows=%s, sample=%.2f)", path.name, max_rows, sample_fraction)

    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            if max_rows and max_rows < chunk_size:
                df = pd.read_csv(path, nrows=max_rows, encoding=encoding, low_memory=False)
            elif sample_fraction < 1.0:
                df = pd.read_csv(path, encoding=encoding, low_memory=False)
                df = df.sample(frac=sample_fraction, random_state=random_seed)
            else:
                df = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            if encoding == "cp1252":
                raise
            continue

    df.columns = df.columns.str.strip()
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path.name)
    return df


def load_dataset(
    raw_dir: str | Path,
    *,
    max_rows: int | None = None,
    sample_fraction: float = 1.0,
    chunk_size: int = 100_000,
    random_seed: int = 42,
    file_pattern: str | None = None,
) -> pd.DataFrame:
    """Load and concatenate all CIC-IDS2017 CSV files."""
    raw_dir = Path(raw_dir)
    csvs = discover_csv_files(raw_dir)
    if file_pattern:
        csvs = [c for c in csvs if file_pattern in c.name]
        if not csvs:
            raise FileNotFoundError(f"No CSV files matching '{file_pattern}' in {raw_dir}")

    frames: list[pd.DataFrame] = []
    total_rows = 0
    for csv_path in csvs:
        remaining = None
        if max_rows is not None:
            remaining = max_rows - total_rows
            if remaining <= 0:
                break

        df = load_single_csv(
            csv_path,
            max_rows=remaining,
            sample_fraction=sample_fraction,
            chunk_size=chunk_size,
            random_seed=random_seed,
        )
        frames.append(df)
        total_rows += len(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows, %d columns", len(combined), len(combined.columns))
    return combined


def compute_dataset_hash(df: pd.DataFrame, n_sample: int = 10_000) -> str:
    """Deterministic hash of a sample for experiment tracking."""
    sample = df.head(n_sample)
    raw = pd.util.hash_pandas_object(sample).values.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a profile dict summarising the dataset."""
    label_col = _find_column(df, KNOWN_LABEL_COLUMNS)
    ts_col = _find_column(df, KNOWN_TIMESTAMP_COLUMNS)

    profile: dict[str, Any] = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "infinite_values": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    numeric_cols = df.select_dtypes(include="number").columns
    inf_counts = {}
    for col in numeric_cols:
        n_inf = int((df[col] == float("inf")).sum() + (df[col] == float("-inf")).sum())
        if n_inf > 0:
            inf_counts[col] = n_inf
    profile["infinite_values"] = inf_counts
    profile["total_infinite"] = sum(inf_counts.values())

    if label_col:
        dist = df[label_col].value_counts()
        profile["label_column"] = label_col
        profile["class_distribution"] = dist.to_dict()
        profile["class_percentages"] = (dist / len(df) * 100).round(2).to_dict()
        profile["unique_labels"] = int(df[label_col].nunique())
    else:
        profile["label_column"] = None

    if ts_col:
        profile["timestamp_column"] = ts_col
        try:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            profile["timestamp_min"] = str(ts.min())
            profile["timestamp_max"] = str(ts.max())
        except Exception:
            profile["timestamp_min"] = None
            profile["timestamp_max"] = None
    else:
        profile["timestamp_column"] = None

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    profile["constant_columns"] = constant_cols

    potential_id_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("ip", "port", "flow id"))
    ]
    profile["potential_id_leakage_columns"] = potential_id_cols

    return profile


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
