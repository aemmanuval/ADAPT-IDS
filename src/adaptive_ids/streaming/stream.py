"""Traffic-stream abstractions for ordered evaluation and drift detection.

Provides an iterator interface so the drift pipeline sees data one sample
(or batch) at a time, preserving chronological ordering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

import numpy as np
import pandas as pd

from adaptive_ids.utils.logging import get_logger

logger = get_logger("streaming")


class TrafficStream(ABC):
    """Base interface for ordered traffic data streams."""

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class CSVStream(TrafficStream):
    """Stream events from an in-memory DataFrame, one sample at a time."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: list[str] | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.timestamps = timestamps
        self._position = 0

    def __len__(self) -> int:
        return self.X.shape[0]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._position = 0
        for i in range(len(self)):
            event = {
                "index": i,
                "features": self.X[i],
                "label": self.y[i] if i < len(self.y) else None,
            }
            if self.timestamps is not None and i < len(self.timestamps):
                event["timestamp"] = self.timestamps[i]
            self._position = i + 1
            yield event

    def reset(self) -> None:
        self._position = 0

    @property
    def position(self) -> int:
        return self._position


class BatchStream(TrafficStream):
    """Yields non-overlapping batches from a stream source."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 1000,
        *,
        feature_names: list[str] | None = None,
        timestamps: np.ndarray | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.feature_names = feature_names
        self.timestamps = timestamps
        self._position = 0

    def __len__(self) -> int:
        return self.X.shape[0]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._position = 0
        n = self.X.shape[0]
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = {
                "batch_start": start,
                "batch_end": end,
                "features": self.X[start:end],
                "labels": self.y[start:end],
            }
            if self.timestamps is not None:
                batch["timestamps"] = self.timestamps[start:end]
            self._position = end
            yield batch

    def reset(self) -> None:
        self._position = 0

    @property
    def position(self) -> int:
        return self._position
