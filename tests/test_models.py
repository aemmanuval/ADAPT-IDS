"""Tests for baseline model training and prediction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from adaptive_ids.models.baseline import BaselineIDS


class TestBaselineIDS:
    def _make_data(self, n=200, n_features=10):
        rng = np.random.RandomState(42)
        X = rng.randn(n, n_features)
        y = np.array(["ATTACK" if x > 0 else "BENIGN" for x in X[:, 0]])
        return X, y

    def test_lightgbm_train_predict(self):
        X, y = self._make_data()
        model = BaselineIDS("lightgbm", params={"n_estimators": 10, "verbose": -1}, random_seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({"ATTACK", "BENIGN"})

    def test_random_forest_train_predict(self):
        X, y = self._make_data()
        model = BaselineIDS("random_forest", params={"n_estimators": 10}, random_seed=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_predict_single(self):
        X, y = self._make_data()
        model = BaselineIDS("lightgbm", params={"n_estimators": 10, "verbose": -1}, random_seed=42)
        model.fit(X, y)
        pred = model.predict_single(X[0])
        assert pred in {"ATTACK", "BENIGN"}

    def test_save_load(self):
        X, y = self._make_data()
        model = BaselineIDS("lightgbm", params={"n_estimators": 10, "verbose": -1}, random_seed=42)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            model.save(path)
            loaded = BaselineIDS.load(path)
            preds_orig = model.predict(X)
            preds_loaded = loaded.predict(X)
            np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_metadata(self):
        X, y = self._make_data()
        model = BaselineIDS("lightgbm", params={"n_estimators": 10, "verbose": -1}, random_seed=42)
        model.fit(X, y)
        assert "training_time_s" in model.metadata
        assert model.metadata["algorithm"] == "lightgbm"

    def test_feature_importances(self):
        X, y = self._make_data()
        model = BaselineIDS("lightgbm", params={"n_estimators": 10, "verbose": -1}, random_seed=42)
        model.fit(X, y)
        imp = model.feature_importances()
        assert imp is not None
        assert len(imp) == X.shape[1]

    def test_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported"):
            BaselineIDS("xgboost")
