"""End-to-end preprocessing pipeline for CIC-IDS2017 flow data.

Leakage-aware: all fitted transformations are learned only on training data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from adaptive_ids.utils.logging import get_logger

logger = get_logger("preprocessing")

LEAKAGE_RISK_COLUMNS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Protocol",
]


@dataclass
class PreprocessingReport:
    """Tracks every transformation applied to the data."""
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    rows_dropped_nan: int = 0
    rows_dropped_inf: int = 0
    rows_dropped_duplicate: int = 0
    columns_dropped_constant: list[str] = field(default_factory=list)
    columns_dropped_id: list[str] = field(default_factory=list)
    inf_replaced: dict[str, int] = field(default_factory=dict)
    nan_filled: dict[str, int] = field(default_factory=dict)
    label_mapping: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("Preprocessing report saved to %s", path)


class PreprocessingPipeline:
    """Stateful, leakage-safe preprocessing pipeline.

    Call ``fit_transform`` on training data, then ``transform`` on
    validation/test data using the same fitted state.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.pp_cfg = config.get("preprocessing", {})
        self.cls_cfg = config.get("classification", {})
        self.ds_cfg = config.get("dataset", {})

        self.label_column: str = self.ds_cfg.get("label_column", "Label")
        self.timestamp_column: str = self.ds_cfg.get("timestamp_column", "Timestamp")

        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] = []
        self.label_encoder: LabelEncoder | None = None
        self._is_fitted = False

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
        """Apply cleaning steps that are safe before train/test split.

        Returns cleaned dataframe and report. Does NOT fit scalers.
        """
        report = PreprocessingReport(
            rows_before=len(df),
            columns_before=len(df.columns),
        )

        df = df.copy()
        df.columns = df.columns.str.strip()

        if self.label_column not in df.columns:
            for candidate in [" Label", "label", " label"]:
                if candidate in df.columns:
                    df.rename(columns={candidate: self.label_column}, inplace=True)
                    break

        df = self._map_labels(df, report)
        df = self._handle_infinity(df, report)
        df = self._handle_nan(df, report)
        df = self._drop_id_columns(df, report)
        df = self._drop_constant_columns(df, report)

        if self.pp_cfg.get("remove_duplicates", True):
            n_before = len(df)
            df = df.drop_duplicates()
            report.rows_dropped_duplicate = n_before - len(df)

        report.rows_after = len(df)
        report.columns_after = len(df.columns)
        return df, report

    def fit_transform(
        self, df: pd.DataFrame, *, scale: bool = False
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Fit on training data and return (features_df, X, y).

        Learns scaler parameters and feature column list.
        """
        df, _ = self._ensure_clean(df)
        X_df, y = self._separate_features_label(df)

        self.feature_columns = list(X_df.columns)

        if scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X_df.values)
        else:
            X = X_df.values.astype(np.float64)

        self._is_fitted = True
        return X_df, X, y

    def transform(
        self, df: pd.DataFrame, *, scale: bool = False
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Transform new data using already-fitted state."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")

        df, _ = self._ensure_clean(df)
        X_df, y = self._separate_features_label(df)

        missing = set(self.feature_columns) - set(X_df.columns)
        if missing:
            for col in missing:
                X_df[col] = 0.0
                logger.warning("Missing feature '%s' filled with 0", col)

        X_df = X_df[self.feature_columns]

        if scale and self.scaler is not None:
            X = self.scaler.transform(X_df.values)
        else:
            X = X_df.values.astype(np.float64)

        return X_df, X, y

    def _ensure_clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingReport]:
        return self.clean(df)

    def _map_labels(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        if self.label_column not in df.columns:
            return df

        df[self.label_column] = df[self.label_column].astype(str).str.strip()

        if self.cls_cfg.get("task") == "binary":
            binary_map = self.cls_cfg.get("binary_mapping", {})
            benign_label = binary_map.get("BENIGN", "BENIGN")
            default_label = binary_map.get("default", "ATTACK")

            original_labels = df[self.label_column].unique().tolist()
            df[self.label_column] = df[self.label_column].apply(
                lambda x: "BENIGN" if x.upper() == benign_label.upper() else default_label
            )
            report.label_mapping = {
                "task": "binary",
                "original_labels": original_labels,
                "mapped_to": ["BENIGN", default_label],
            }
        else:
            report.label_mapping = {
                "task": "multiclass",
                "labels": df[self.label_column].unique().tolist(),
            }
        return df

    def _handle_infinity(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        if not self.pp_cfg.get("handle_infinity", True):
            return df

        numeric = df.select_dtypes(include="number")
        for col in numeric.columns:
            mask = np.isinf(df[col].values)
            n_inf = int(mask.sum())
            if n_inf > 0:
                report.inf_replaced[col] = n_inf
                df.loc[mask, col] = np.nan
        return df

    def _handle_nan(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        if not self.pp_cfg.get("handle_nan", True):
            return df

        numeric = df.select_dtypes(include="number")
        strategy = self.pp_cfg.get("nan_strategy", "median")

        for col in numeric.columns:
            n_nan = int(df[col].isna().sum())
            if n_nan > 0:
                report.nan_filled[col] = n_nan
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "zero":
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())

        n_before = len(df)
        df = df.dropna()
        report.rows_dropped_nan = n_before - len(df)
        return df

    def _drop_id_columns(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        if not self.pp_cfg.get("drop_id_columns", True):
            return df

        to_drop = [
            c for c in self.pp_cfg.get("id_columns", LEAKAGE_RISK_COLUMNS)
            if c in df.columns
        ]
        if self.timestamp_column in df.columns and self.timestamp_column not in to_drop:
            pass  # keep timestamp for temporal splitting

        report.columns_dropped_id = to_drop
        return df.drop(columns=to_drop, errors="ignore")

    def _drop_constant_columns(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        if not self.pp_cfg.get("drop_constant_features", True):
            return df

        threshold = self.pp_cfg.get("constant_threshold", 0.99)
        to_drop: list[str] = []
        numeric = df.select_dtypes(include="number")
        for col in numeric.columns:
            if df[col].nunique() <= 1:
                to_drop.append(col)
            elif (df[col].value_counts(normalize=True).iloc[0] >= threshold):
                to_drop.append(col)

        protected = {self.label_column, self.timestamp_column}
        to_drop = [c for c in to_drop if c not in protected]
        report.columns_dropped_constant = to_drop
        return df.drop(columns=to_drop, errors="ignore")

    def _separate_features_label(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        exclude = {self.label_column, self.timestamp_column}
        feature_cols = [c for c in df.columns if c not in exclude]

        X_df = df[feature_cols].copy()
        non_numeric = X_df.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            logger.warning("Dropping non-numeric columns from features: %s", non_numeric)
            X_df = X_df.drop(columns=non_numeric)

        y = df[self.label_column].values if self.label_column in df.columns else np.array([])
        return X_df, y
