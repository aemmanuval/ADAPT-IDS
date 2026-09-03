"""Adaptation strategies for drift-aware IDS.

Compares:
  A. Static model (no adaptation)
  B. Periodic retraining (every N samples)
  C. Drift-triggered retraining (retrain only when drift is detected)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np

from adaptive_ids.models.baseline import BaselineIDS
from adaptive_ids.drift.detectors import DriftDetector
from adaptive_ids.utils.logging import get_logger

logger = get_logger("adaptation")

MINIMUM_RETRAIN_SAMPLES = 500


class AdaptationStrategy(ABC):
    """Base interface for all adaptation strategies."""

    @abstractmethod
    def should_retrain(self, stream_position: int, drift_detected: bool) -> bool:
        ...

    @abstractmethod
    def record_sample(self, x: np.ndarray, y_true: str, y_pred: str) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        ...


class StaticStrategy(AdaptationStrategy):
    """No adaptation — baseline static model."""

    def __init__(self) -> None:
        self._n_seen = 0

    @property
    def name(self) -> str:
        return "static"

    def should_retrain(self, stream_position: int, drift_detected: bool) -> bool:
        return False

    def record_sample(self, x: np.ndarray, y_true: str, y_pred: str) -> None:
        self._n_seen += 1

    def get_stats(self) -> dict[str, Any]:
        return {"strategy": self.name, "n_seen": self._n_seen, "n_retrains": 0}


class PeriodicStrategy(AdaptationStrategy):
    """Retrain every *period* samples regardless of drift."""

    def __init__(self, period: int = 10000) -> None:
        self.period = period
        self._n_seen = 0
        self._last_retrain = 0
        self._n_retrains = 0

    @property
    def name(self) -> str:
        return f"periodic_{self.period}"

    def should_retrain(self, stream_position: int, drift_detected: bool) -> bool:
        if stream_position - self._last_retrain >= self.period:
            self._last_retrain = stream_position
            self._n_retrains += 1
            return True
        return False

    def record_sample(self, x: np.ndarray, y_true: str, y_pred: str) -> None:
        self._n_seen += 1

    def get_stats(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "period": self.period,
            "n_seen": self._n_seen,
            "n_retrains": self._n_retrains,
        }


class DriftTriggeredStrategy(AdaptationStrategy):
    """Retrain only when the drift detector fires."""

    def __init__(self, cooldown: int = 2000) -> None:
        self.cooldown = cooldown
        self._n_seen = 0
        self._last_retrain = -cooldown
        self._n_retrains = 0

    @property
    def name(self) -> str:
        return "drift_triggered"

    def should_retrain(self, stream_position: int, drift_detected: bool) -> bool:
        if drift_detected and (stream_position - self._last_retrain >= self.cooldown):
            self._last_retrain = stream_position
            self._n_retrains += 1
            return True
        return False

    def record_sample(self, x: np.ndarray, y_true: str, y_pred: str) -> None:
        self._n_seen += 1

    def get_stats(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "cooldown": self.cooldown,
            "n_seen": self._n_seen,
            "n_retrains": self._n_retrains,
        }


class AdaptiveModelManager:
    """Manages model retraining using a sliding window of recent labelled data.

    When a strategy triggers retraining, the manager retrains the model
    on the most recent *window_size* samples from the stream.
    """

    def __init__(
        self,
        algorithm: str,
        model_params: dict[str, Any],
        strategy: AdaptationStrategy,
        detector: DriftDetector,
        window_size: int = 20000,
        random_seed: int = 42,
    ) -> None:
        self.algorithm = algorithm
        self.model_params = model_params
        self.strategy = strategy
        self.detector = detector
        self.window_size = window_size
        self.random_seed = random_seed

        self.model: BaselineIDS | None = None
        self._X_buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._y_buffer: deque[str] = deque(maxlen=window_size)
        self._retrain_log: list[dict[str, Any]] = []
        self._total_retrain_time: float = 0.0

    def set_initial_model(self, model: BaselineIDS) -> None:
        self.model = model

    def train_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = BaselineIDS(
            self.algorithm, params=self.model_params, random_seed=self.random_seed
        )
        self.model.fit(X, y)

    def process_sample(
        self, x: np.ndarray, y_true: str, stream_position: int
    ) -> tuple[str, bool, bool]:
        """Process one sample. Returns (prediction, is_correct, did_retrain)."""
        y_pred = self.model.predict_single(x)
        is_correct = y_pred == y_true
        error = 0.0 if is_correct else 1.0

        self._X_buffer.append(x)
        self._y_buffer.append(y_true)

        self.detector.update(error)
        drift = self.detector.drift_detected()

        self.strategy.record_sample(x, y_true, y_pred)
        did_retrain = False

        if self.strategy.should_retrain(stream_position, drift):
            if len(self._X_buffer) >= MINIMUM_RETRAIN_SAMPLES:
                self._retrain(stream_position)
                did_retrain = True

        return y_pred, is_correct, did_retrain

    def _retrain(self, stream_position: int) -> None:
        X_train = np.array(self._X_buffer)
        y_train = np.array(self._y_buffer)

        unique_labels = set(y_train)
        if len(unique_labels) < 2:
            logger.warning(
                "Skipping retrain at %d: only %d class(es) in window",
                stream_position, len(unique_labels),
            )
            return

        t0 = time.perf_counter()
        new_model = BaselineIDS(
            self.algorithm, params=self.model_params, random_seed=self.random_seed
        )
        new_model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        self.model = new_model
        self._total_retrain_time += elapsed
        self._retrain_log.append({
            "position": stream_position,
            "window_size": len(X_train),
            "classes": list(unique_labels),
            "train_time_s": round(elapsed, 3),
        })
        logger.info(
            "Retrained at position %d (window=%d, classes=%d, %.2fs)",
            stream_position, len(X_train), len(unique_labels), elapsed,
        )

    @property
    def retrain_log(self) -> list[dict[str, Any]]:
        return self._retrain_log

    @property
    def total_retrain_time(self) -> float:
        return self._total_retrain_time

    @property
    def n_retrains(self) -> int:
        return len(self._retrain_log)
