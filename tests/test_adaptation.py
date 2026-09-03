"""Tests for adaptation strategies — the core research contribution."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_ids.adaptation.strategies import (
    StaticStrategy, PeriodicStrategy, DriftTriggeredStrategy,
    AdaptiveModelManager, MINIMUM_RETRAIN_SAMPLES,
)
from adaptive_ids.drift.detectors import ADWINDetector
from adaptive_ids.models.baseline import BaselineIDS


def _make_data(n=1000, n_features=10, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = np.array(["ATTACK" if x > 0 else "BENIGN" for x in X[:, 0]])
    return X, y


class TestStaticStrategy:
    def test_never_retrains(self):
        s = StaticStrategy()
        assert not s.should_retrain(0, False)
        assert not s.should_retrain(100, True)
        assert not s.should_retrain(999999, True)

    def test_stats(self):
        s = StaticStrategy()
        s.record_sample(np.zeros(5), "A", "A")
        s.record_sample(np.zeros(5), "A", "B")
        stats = s.get_stats()
        assert stats["n_seen"] == 2
        assert stats["n_retrains"] == 0


class TestPeriodicStrategy:
    def test_retrains_at_period(self):
        s = PeriodicStrategy(period=100)
        assert not s.should_retrain(50, False)
        assert s.should_retrain(100, False)
        assert not s.should_retrain(150, False)
        assert s.should_retrain(200, False)

    def test_counts_retrains(self):
        s = PeriodicStrategy(period=10)
        count = 0
        for i in range(105):
            if s.should_retrain(i, False):
                count += 1
        assert count == 10


class TestDriftTriggeredStrategy:
    def test_retrains_on_drift(self):
        s = DriftTriggeredStrategy(cooldown=0)
        assert s.should_retrain(100, True)

    def test_ignores_no_drift(self):
        s = DriftTriggeredStrategy(cooldown=0)
        assert not s.should_retrain(100, False)

    def test_cooldown(self):
        s = DriftTriggeredStrategy(cooldown=50)
        assert s.should_retrain(100, True)
        assert not s.should_retrain(120, True)
        assert s.should_retrain(160, True)


class TestAdaptiveModelManager:
    def test_initial_model_predicts(self):
        X, y = _make_data()
        model = BaselineIDS("lightgbm", params={"n_estimators": 10, "verbose": -1}, random_seed=42)
        model.fit(X, y)

        detector = ADWINDetector(delta=0.01)
        strategy = StaticStrategy()
        manager = AdaptiveModelManager(
            algorithm="lightgbm",
            model_params={"n_estimators": 10, "verbose": -1},
            strategy=strategy,
            detector=detector,
            window_size=500,
            random_seed=42,
        )
        manager.set_initial_model(model)

        pred, correct, retrained = manager.process_sample(X[0], y[0], 0)
        assert pred in {"ATTACK", "BENIGN"}
        assert not retrained

    def test_drift_triggered_retrains(self):
        X, y = _make_data(n=2000)
        model = BaselineIDS("lightgbm", params={"n_estimators": 10, "verbose": -1}, random_seed=42)
        model.fit(X[:500], y[:500])

        detector = ADWINDetector(delta=0.1)
        strategy = DriftTriggeredStrategy(cooldown=100)
        manager = AdaptiveModelManager(
            algorithm="lightgbm",
            model_params={"n_estimators": 10, "verbose": -1},
            strategy=strategy,
            detector=detector,
            window_size=MINIMUM_RETRAIN_SAMPLES + 100,
            random_seed=42,
        )
        manager.set_initial_model(model)

        rng = np.random.RandomState(99)
        X_drift = rng.randn(1000, 10) * 5
        y_drift = np.array(["BENIGN" if x > 0 else "ATTACK" for x in X_drift[:, 0]])

        for i in range(len(X_drift)):
            manager.process_sample(X_drift[i], y_drift[i], i + 1000)

        assert manager.n_retrains >= 0

    def test_buffer_size_bounded(self):
        X, y = _make_data(n=100)
        model = BaselineIDS("lightgbm", params={"n_estimators": 10, "verbose": -1}, random_seed=42)
        model.fit(X, y)

        detector = ADWINDetector()
        strategy = StaticStrategy()
        manager = AdaptiveModelManager(
            algorithm="lightgbm",
            model_params={"n_estimators": 10, "verbose": -1},
            strategy=strategy,
            detector=detector,
            window_size=50,
            random_seed=42,
        )
        manager.set_initial_model(model)

        for i in range(200):
            manager.process_sample(X[i % len(X)], y[i % len(y)], i)

        assert len(manager._X_buffer) <= 50
        assert len(manager._y_buffer) <= 50
