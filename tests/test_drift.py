"""Tests for drift detection and event logging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from adaptive_ids.drift.detectors import ADWINDetector, create_detector
from adaptive_ids.drift.events import DriftEventLogger, DriftEvent
from adaptive_ids.drift.synthetic import SyntheticDriftGenerator


class TestADWINDetector:
    def test_initialization(self):
        det = ADWINDetector(delta=0.002)
        assert det.name == "ADWIN"
        assert not det.drift_detected()

    def test_no_drift_on_stable_stream(self):
        det = ADWINDetector(delta=0.002)
        rng = np.random.RandomState(42)
        for _ in range(500):
            det.update(float(rng.binomial(1, 0.1)))
        state = det.get_state()
        assert state["n_seen"] == 500

    def test_drift_on_distribution_change(self):
        det = ADWINDetector(delta=0.01)
        rng = np.random.RandomState(42)
        # Stable low-error phase
        for _ in range(500):
            det.update(float(rng.binomial(1, 0.05)))
        # High-error phase
        detected = False
        for _ in range(500):
            det.update(float(rng.binomial(1, 0.6)))
            if det.drift_detected():
                detected = True
                break
        assert detected, "ADWIN should detect shift from 5% to 60% error"

    def test_reset(self):
        det = ADWINDetector()
        for _ in range(100):
            det.update(0.0)
        det.reset()
        state = det.get_state()
        assert state["n_seen"] == 0
        assert state["n_drifts"] == 0

    def test_get_state_keys(self):
        det = ADWINDetector()
        det.update(1.0)
        state = det.get_state()
        assert "detector" in state
        assert "n_seen" in state
        assert "window_size" in state


class TestCreateDetector:
    def test_create_adwin(self):
        cfg = {"detector": "adwin", "adwin": {"delta": 0.005}}
        det = create_detector(cfg)
        assert det.name == "ADWIN"

    def test_unknown_detector(self):
        with pytest.raises(ValueError, match="Unknown drift detector"):
            create_detector({"detector": "unknown"})


class TestDriftEventLogger:
    def test_log_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = DriftEventLogger(tmp)
            logger.log_drift("ADWIN", 100)
            logger.log_drift("ADWIN", 200)
            assert logger.count == 2

    def test_save_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = DriftEventLogger(tmp)
            logger.log_drift("ADWIN", 42, metric_context={"error": 0.3})
            path = logger.save_csv()
            assert path.exists()
            assert path.stat().st_size > 0

    def test_save_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = DriftEventLogger(tmp)
            logger.log_drift("ADWIN", 99)
            path = logger.save_json()
            data = json.loads(path.read_text())
            assert len(data) == 1
            assert data[0]["stream_position"] == 99

    def test_event_serialization(self):
        event = DriftEvent(
            event_id=1, event_type="drift", detector="ADWIN",
            stream_position=500, model_version="v1",
        )
        d = event.to_dict()
        assert d["event_id"] == 1
        assert d["detector"] == "ADWIN"


class TestSyntheticDrift:
    def test_sudden_drift_changes_distribution(self):
        gen = SyntheticDriftGenerator(seed=42)
        X = np.random.RandomState(42).randn(1000, 5)
        y = np.array(["A"] * 1000)
        X_out, y_out = gen.sudden_drift(X, y, position=0.5, magnitude=2.0)
        mean_before = X_out[:500, 0].mean()
        mean_after = X_out[500:, 0].mean()
        assert abs(mean_after - mean_before) > 0.5, "Sudden drift should shift means"

    def test_original_unchanged(self):
        gen = SyntheticDriftGenerator(seed=42)
        X_orig = np.random.RandomState(42).randn(100, 3)
        X_copy = X_orig.copy()
        y = np.array(["A"] * 100)
        gen.sudden_drift(X_orig, y)
        np.testing.assert_array_equal(X_orig, X_copy)

    def test_gradual_drift(self):
        gen = SyntheticDriftGenerator(seed=42)
        X = np.random.RandomState(42).randn(1000, 2)
        y = np.array(["A"] * 1000)
        X_out, _ = gen.gradual_drift(X, y, start=0.3, end=0.7, magnitude=2.0)
        mean_before = X_out[:300, 0].mean()
        mean_after = X_out[700:, 0].mean()
        assert abs(mean_after - mean_before) > 0.3, "Gradual drift should shift means"

    def test_recurring_drift(self):
        gen = SyntheticDriftGenerator(seed=42)
        X = np.random.RandomState(42).randn(1000, 2)
        y = np.array(["A"] * 1000)
        X_out, _ = gen.recurring_drift(X, y, cycle_length=0.25, magnitude=2.0)
        assert not np.allclose(X_out[:250], X_out[250:500])
