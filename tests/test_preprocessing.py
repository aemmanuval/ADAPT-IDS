"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline


class TestCleaning:
    def test_nan_handling(self, sample_df, config):
        pipe = PreprocessingPipeline(config)
        cleaned, report = pipe.clean(sample_df)
        numeric = cleaned.select_dtypes(include="number")
        assert numeric.isna().sum().sum() == 0, "NaN values remain after cleaning"
        assert report.nan_filled, "NaN fill report is empty despite NaNs in input"

    def test_infinity_handling(self, sample_df, config):
        pipe = PreprocessingPipeline(config)
        cleaned, report = pipe.clean(sample_df)
        numeric = cleaned.select_dtypes(include="number")
        n_inf = np.isinf(numeric.values).sum()
        assert n_inf == 0, f"{n_inf} infinite values remain after cleaning"
        assert report.inf_replaced, "Inf replace report empty despite inf in input"

    def test_label_mapping_binary(self, sample_df, config):
        cfg = {**config, "classification": {"task": "binary", "binary_mapping": {"BENIGN": "BENIGN", "default": "ATTACK"}}}
        pipe = PreprocessingPipeline(cfg)
        cleaned, report = pipe.clean(sample_df)
        labels = cleaned["Label"].unique()
        assert set(labels) == {"BENIGN", "ATTACK"}, f"Unexpected labels: {labels}"

    def test_label_mapping_multiclass(self, sample_df, config):
        cfg = {**config, "classification": {"task": "multiclass"}}
        pipe = PreprocessingPipeline(cfg)
        cleaned, _ = pipe.clean(sample_df)
        assert cleaned["Label"].nunique() >= 3

    def test_id_columns_dropped(self, sample_df, config):
        sample_df["Flow ID"] = "test"
        sample_df["Source IP"] = "1.2.3.4"
        pipe = PreprocessingPipeline(config)
        cleaned, report = pipe.clean(sample_df)
        assert "Flow ID" not in cleaned.columns
        assert "Source IP" not in cleaned.columns
        assert "Flow ID" in report.columns_dropped_id

    def test_timestamp_preserved(self, sample_df, config):
        pipe = PreprocessingPipeline(config)
        cleaned, _ = pipe.clean(sample_df)
        assert "Timestamp" in cleaned.columns, "Timestamp must be preserved for temporal splitting"

    def test_duplicate_removal(self, sample_df, config):
        df = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        pipe = PreprocessingPipeline(config)
        _, report = pipe.clean(df)
        assert report.rows_dropped_duplicate >= 5


class TestFitTransform:
    def test_fit_transform_shapes(self, sample_df, config):
        pipe = PreprocessingPipeline(config)
        X_df, X, y = pipe.fit_transform(sample_df)
        assert X.shape[0] == len(y)
        assert X.shape[1] > 0
        assert len(y) > 0

    def test_transform_uses_fitted_columns(self, sample_df, config):
        pipe = PreprocessingPipeline(config)
        _, X_train, _ = pipe.fit_transform(sample_df.iloc[:700])
        _, X_test, _ = pipe.transform(sample_df.iloc[700:])
        assert X_train.shape[1] == X_test.shape[1]

    def test_no_label_in_features(self, sample_df, config):
        pipe = PreprocessingPipeline(config)
        X_df, _, _ = pipe.fit_transform(sample_df)
        assert "Label" not in X_df.columns
