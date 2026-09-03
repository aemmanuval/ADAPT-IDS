"""Controlled synthetic-drift generators for experimental evaluation.

Generates modified copies of the data — never mutates the original.
Supports both covariate shift (feature distributions change) and
concept drift (label relationship changes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from adaptive_ids.utils.logging import get_logger

logger = get_logger("drift.synthetic")


class SyntheticDriftGenerator:
    """Inject controlled drift into a feature matrix for experiments."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def sudden_drift(
        self,
        X: np.ndarray,
        y: np.ndarray,
        position: float = 0.6,
        magnitude: float = 0.3,
        affected_features: list[int] | None = None,
        label_flip_rate: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Abrupt shift in feature distributions AND label relationships."""
        X_out = X.copy()
        y_out = y.copy()
        n = X_out.shape[0]
        split = int(n * position)

        if affected_features is None:
            affected_features = list(range(X_out.shape[1]))

        for f in affected_features:
            std = X_out[:split, f].std() + 1e-10
            X_out[split:, f] += magnitude * std * self.rng.choice([-1, 1])

        if label_flip_rate > 0:
            flip_mask = self.rng.random(n - split) < label_flip_rate
            unique_labels = list(set(y_out))
            if len(unique_labels) >= 2:
                for i in range(split, n):
                    if flip_mask[i - split]:
                        current = y_out[i]
                        other = [l for l in unique_labels if l != current]
                        y_out[i] = self.rng.choice(other)

        logger.info("Sudden drift at position %d/%d, magnitude=%.2f, label_flip=%.2f", split, n, magnitude, label_flip_rate)
        return X_out, y_out

    def gradual_drift(
        self,
        X: np.ndarray,
        y: np.ndarray,
        start: float = 0.4,
        end: float = 0.7,
        magnitude: float = 0.3,
        affected_features: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Progressive shift between *start* and *end* fractions."""
        X_out = X.copy()
        n = X_out.shape[0]
        s, e = int(n * start), int(n * end)

        if affected_features is None:
            affected_features = list(range(X_out.shape[1]))

        for f in affected_features:
            std = X_out[:s, f].std() + 1e-10
            direction = self.rng.choice([-1, 1])
            for i in range(s, n):
                progress = min((i - s) / max(e - s, 1), 1.0)
                X_out[i, f] += progress * magnitude * std * direction

        logger.info("Gradual drift [%d-%d]/%d, magnitude=%.2f", s, e, n, magnitude)
        return X_out, y.copy()

    def incremental_drift(
        self,
        X: np.ndarray,
        y: np.ndarray,
        magnitude: float = 0.1,
        affected_features: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Continuous slow shift across the entire stream."""
        X_out = X.copy()
        n = X_out.shape[0]

        if affected_features is None:
            affected_features = list(range(X_out.shape[1]))

        for f in affected_features:
            std = X_out[:, f].std() + 1e-10
            direction = self.rng.choice([-1, 1])
            for i in range(n):
                X_out[i, f] += (i / n) * magnitude * std * direction

        logger.info("Incremental drift over %d samples, magnitude=%.2f", n, magnitude)
        return X_out, y.copy()

    def recurring_drift(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cycle_length: float = 0.25,
        magnitude: float = 0.3,
        affected_features: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drift that appears and disappears periodically."""
        X_out = X.copy()
        n = X_out.shape[0]
        period = int(n * cycle_length)

        if affected_features is None:
            affected_features = list(range(X_out.shape[1]))

        for f in affected_features:
            std = X_out[:, f].std() + 1e-10
            direction = self.rng.choice([-1, 1])
            for i in range(n):
                cycle = (i // period) % 2
                if cycle == 1:
                    X_out[i, f] += magnitude * std * direction

        logger.info("Recurring drift (period=%d), magnitude=%.2f", period, magnitude)
        return X_out, y.copy()
