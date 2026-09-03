"""Phase 7: Active learning for label-efficient adaptation.

Instead of requiring every sample to be labelled, the system selects
only the most informative samples for analyst review. This reduces
labelling cost while maintaining adaptation quality.

Strategies:
  - Uncertainty sampling: query samples where model confidence is lowest
  - Margin sampling: query samples where top-2 class probabilities are closest
  - Random sampling: baseline — query random subset
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from adaptive_ids.utils.logging import get_logger

logger = get_logger("adaptation.active_learning")


class QueryStrategy(ABC):
    """Selects which samples to request labels for."""

    @abstractmethod
    def select(self, probabilities: np.ndarray, n_query: int) -> np.ndarray:
        """Return indices of samples to query."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class UncertaintySampling(QueryStrategy):
    """Query samples where the model is least confident."""

    @property
    def name(self) -> str:
        return "uncertainty"

    def select(self, probabilities: np.ndarray, n_query: int) -> np.ndarray:
        confidence = probabilities.max(axis=1)
        n_query = min(n_query, len(confidence))
        return np.argsort(confidence)[:n_query]


class MarginSampling(QueryStrategy):
    """Query samples where the margin between top-2 predictions is smallest."""

    @property
    def name(self) -> str:
        return "margin"

    def select(self, probabilities: np.ndarray, n_query: int) -> np.ndarray:
        if probabilities.shape[1] < 2:
            return np.arange(min(n_query, len(probabilities)))

        sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        n_query = min(n_query, len(margins))
        return np.argsort(margins)[:n_query]


class RandomSampling(QueryStrategy):
    """Query random samples — baseline for comparison."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    @property
    def name(self) -> str:
        return "random"

    def select(self, probabilities: np.ndarray, n_query: int) -> np.ndarray:
        n_query = min(n_query, len(probabilities))
        return self.rng.choice(len(probabilities), size=n_query, replace=False)


class ActiveLearningManager:
    """Manages the active learning loop for label-efficient adaptation.

    Flow:
      1. Model predicts on unlabelled batch
      2. Query strategy selects most informative samples
      3. Only selected samples get "labelled" (from ground truth in experiments)
      4. Model retrains on labelled pool
    """

    def __init__(
        self,
        query_strategy: QueryStrategy,
        label_budget_pct: float = 0.1,
    ) -> None:
        self.strategy = query_strategy
        self.label_budget_pct = label_budget_pct
        self._total_samples = 0
        self._total_labelled = 0
        self._labelled_X: list[np.ndarray] = []
        self._labelled_y: list[str] = []

    def query_and_label(
        self,
        model: Any,
        X_batch: np.ndarray,
        y_true_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select samples to label from a batch, return labelled (X, y).

        In a real system, y_true_batch would come from an analyst.
        In experiments, we use the ground truth to simulate labelling.
        """
        n_query = max(1, int(len(X_batch) * self.label_budget_pct))
        self._total_samples += len(X_batch)

        try:
            probs = model.predict_proba(X_batch)
        except Exception:
            indices = np.random.choice(len(X_batch), size=n_query, replace=False)
            probs = None

        if probs is not None:
            indices = self.strategy.select(probs, n_query)
        
        X_selected = X_batch[indices]
        y_selected = y_true_batch[indices]

        self._total_labelled += len(indices)
        self._labelled_X.append(X_selected)
        self._labelled_y.append(y_selected)

        return X_selected, y_selected

    def get_labelled_pool(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all labelled samples collected so far."""
        if not self._labelled_X:
            return np.array([]), np.array([])
        return np.concatenate(self._labelled_X), np.concatenate(self._labelled_y)

    @property
    def label_efficiency(self) -> float:
        if self._total_samples == 0:
            return 0.0
        return self._total_labelled / self._total_samples

    def get_stats(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.name,
            "budget_pct": self.label_budget_pct,
            "total_samples": self._total_samples,
            "total_labelled": self._total_labelled,
            "label_efficiency": round(self.label_efficiency, 4),
        }
