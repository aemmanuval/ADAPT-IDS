"""Tests for evaluation metrics and temporal splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_ids.evaluation.metrics import compute_metrics, compute_windowed_metrics
from adaptive_ids.evaluation.temporal import temporal_split, random_split


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array(["ATTACK", "BENIGN", "ATTACK", "BENIGN"])
        m = compute_metrics(y, y, positive_label="ATTACK")
        assert m["f1"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_all_wrong(self):
        y_true = np.array(["ATTACK", "ATTACK", "ATTACK"])
        y_pred = np.array(["BENIGN", "BENIGN", "BENIGN"])
        m = compute_metrics(y_true, y_pred, positive_label="ATTACK")
        assert m["recall"] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = np.array(["A", "B", "A", "B", "A"])
        y_pred = np.array(["A", "A", "A", "B", "B"])
        m = compute_metrics(y_true, y_pred, positive_label="A")
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (2, 2)

    def test_fpr_fnr_binary(self):
        y_true = np.array(["ATTACK"] * 50 + ["BENIGN"] * 50)
        y_pred = np.array(["ATTACK"] * 40 + ["BENIGN"] * 10 + ["ATTACK"] * 5 + ["BENIGN"] * 45)
        m = compute_metrics(y_true, y_pred, positive_label="ATTACK")
        assert 0 <= m["fpr"] <= 1
        assert 0 <= m["fnr"] <= 1
        assert m["fnr"] == pytest.approx(10 / 50)  # 10 attacks missed
        assert m["fpr"] == pytest.approx(5 / 50)   # 5 benign flagged

    def test_support(self):
        y = np.array(["A", "B", "A"])
        m = compute_metrics(y, y)
        assert m["support"] == 3


class TestWindowedMetrics:
    def test_window_count(self):
        y_true = np.array(["A"] * 100)
        y_pred = np.array(["A"] * 100)
        windows = compute_windowed_metrics(y_true, y_pred, window_size=25)
        assert len(windows) == 4

    def test_window_ids_sequential(self):
        y_true = np.array(["A"] * 50)
        y_pred = np.array(["A"] * 50)
        windows = compute_windowed_metrics(y_true, y_pred, window_size=10)
        ids = [w["window_id"] for w in windows]
        assert ids == list(range(5))


class TestTemporalSplit:
    def test_no_overlap(self):
        n = 1000
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2017-07-03", periods=n, freq="min"),
            "Label": ["A"] * n,
            "feature": np.random.randn(n),
        })
        splits = temporal_split(df, train_fraction=0.7, validation_fraction=0.1, test_fraction=0.2)
        train_idx = set(splits["train"].index)
        test_idx = set(splits["test"].index)
        assert train_idx & test_idx == set(), "Train and test indices must not overlap"

    def test_temporal_ordering(self):
        n = 1000
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2017-07-03", periods=n, freq="min"),
            "Label": ["A"] * n,
            "feature": np.random.randn(n),
        })
        splits = temporal_split(df, train_fraction=0.7, validation_fraction=0.1, test_fraction=0.2)
        train_max = pd.to_datetime(splits["train"]["Timestamp"]).max()
        test_min = pd.to_datetime(splits["test"]["Timestamp"]).min()
        assert train_max < test_min, "max(train_ts) must < min(test_ts)"

    def test_sizes(self):
        n = 1000
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2017-07-03", periods=n, freq="min"),
            "Label": ["A"] * n,
            "feature": np.random.randn(n),
        })
        splits = temporal_split(df, train_fraction=0.7, validation_fraction=0.1, test_fraction=0.2)
        total = sum(len(s) for s in splits.values())
        assert total == n


class TestRandomSplit:
    def test_sizes(self):
        df = pd.DataFrame({"Label": ["A"] * 80 + ["B"] * 20, "x": range(100)})
        splits = random_split(df, train_fraction=0.8, test_fraction=0.2, random_seed=42)
        assert len(splits["train"]) + len(splits["test"]) == 100

    def test_stratification(self):
        df = pd.DataFrame({"Label": ["A"] * 80 + ["B"] * 20, "x": range(100)})
        splits = random_split(df, stratify_column="Label", random_seed=42)
        train_ratio = (splits["train"]["Label"] == "B").mean()
        assert 0.15 < train_ratio < 0.25
