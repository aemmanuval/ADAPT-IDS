"""Drift-detection wrappers with a uniform interface.

First implementation: ADWIN via the River library.
Interface is designed so DDM, EDDM, and Page-Hinkley can be added later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adaptive_ids.utils.logging import get_logger

logger = get_logger("drift.detectors")


class DriftDetector(ABC):
    """Abstract interface for all drift detectors."""

    @abstractmethod
    def update(self, value: float) -> None:
        """Feed one observation (e.g. error signal 0/1)."""

    @abstractmethod
    def drift_detected(self) -> bool:
        ...

    @abstractmethod
    def warning_detected(self) -> bool:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Serialisable snapshot of internal state for logging."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class ADWINDetector(DriftDetector):
    """ADWIN (Adaptive Windowing) drift detector backed by River.

    Monitors a 0/1 error stream. Detects when the mean error rate in
    recent data differs significantly from the historical mean — i.e.
    the model's accuracy distribution has changed.

    Parameters
    ----------
    delta : float
        Confidence parameter. Smaller → fewer false alarms, slower detection.
    clock : int
        How many items between compression attempts.
    max_buckets : int
        Maximum number of buckets per level.
    min_window_length : int
        Minimum window size before detection can fire.
    grace_period : int
        Ignore the first *grace_period* samples.
    """

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
    ) -> None:
        from river.drift import ADWIN

        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period

        self._adwin = ADWIN(
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
        )
        self._drift = False
        self._n_seen = 0
        self._n_drifts = 0

    @property
    def name(self) -> str:
        return "ADWIN"

    def update(self, value: float) -> None:
        self._n_seen += 1
        self._adwin.update(value)
        self._drift = self._adwin.drift_detected
        if self._drift:
            self._n_drifts += 1

    def drift_detected(self) -> bool:
        return self._drift

    def warning_detected(self) -> bool:
        return False  # River's ADWIN does not expose a separate warning zone

    def reset(self) -> None:
        from river.drift import ADWIN

        self._adwin = ADWIN(
            delta=self.delta,
            clock=self.clock,
            max_buckets=self.max_buckets,
            min_window_length=self.min_window_length,
            grace_period=self.grace_period,
        )
        self._drift = False
        self._n_seen = 0
        self._n_drifts = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "detector": self.name,
            "delta": self.delta,
            "n_seen": self._n_seen,
            "n_drifts": self._n_drifts,
            "window_size": self._adwin.width,
            "estimation": self._adwin.estimation,
        }


class DDMDetector(DriftDetector):
    """DDM (Drift Detection Method) via River."""

    def __init__(self, warm_start: int = 30, warning_threshold: float = 2.0, drift_threshold: float = 3.0) -> None:
        from river.drift.binary import DDM
        self._ddm = DDM(warm_start=warm_start, warning_threshold=warning_threshold, drift_threshold=drift_threshold)
        self._drift = False
        self._warning = False
        self._n_seen = 0
        self._n_drifts = 0
        self._params = {"warm_start": warm_start, "warning_threshold": warning_threshold, "drift_threshold": drift_threshold}

    @property
    def name(self) -> str:
        return "DDM"

    def update(self, value: float) -> None:
        self._n_seen += 1
        self._ddm.update(int(value))
        self._drift = self._ddm.drift_detected
        self._warning = self._ddm.warning_detected
        if self._drift:
            self._n_drifts += 1

    def drift_detected(self) -> bool:
        return self._drift

    def warning_detected(self) -> bool:
        return self._warning

    def reset(self) -> None:
        from river.drift.binary import DDM
        self._ddm = DDM(**self._params)
        self._drift = False
        self._warning = False
        self._n_seen = 0
        self._n_drifts = 0

    def get_state(self) -> dict[str, Any]:
        return {"detector": self.name, "n_seen": self._n_seen, "n_drifts": self._n_drifts}


class EDDMDetector(DriftDetector):
    """EDDM (Early Drift Detection Method) via River."""

    def __init__(self, warm_start: int = 30, alpha: float = 0.95, beta: float = 0.9) -> None:
        from river.drift.binary import EDDM
        self._eddm = EDDM(warm_start=warm_start, alpha=alpha, beta=beta)
        self._drift = False
        self._warning = False
        self._n_seen = 0
        self._n_drifts = 0
        self._params = {"warm_start": warm_start, "alpha": alpha, "beta": beta}

    @property
    def name(self) -> str:
        return "EDDM"

    def update(self, value: float) -> None:
        self._n_seen += 1
        self._eddm.update(int(value))
        self._drift = self._eddm.drift_detected
        self._warning = self._eddm.warning_detected
        if self._drift:
            self._n_drifts += 1

    def drift_detected(self) -> bool:
        return self._drift

    def warning_detected(self) -> bool:
        return self._warning

    def reset(self) -> None:
        from river.drift.binary import EDDM
        self._eddm = EDDM(**self._params)
        self._drift = False
        self._warning = False
        self._n_seen = 0
        self._n_drifts = 0

    def get_state(self) -> dict[str, Any]:
        return {"detector": self.name, "n_seen": self._n_seen, "n_drifts": self._n_drifts}


class PageHinkleyDetector(DriftDetector):
    """Page-Hinkley drift detector via River."""

    def __init__(self, min_instances: int = 30, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999) -> None:
        from river.drift import PageHinkley
        self._ph = PageHinkley(min_instances=min_instances, delta=delta, threshold=threshold, alpha=alpha)
        self._drift = False
        self._n_seen = 0
        self._n_drifts = 0
        self._params = {"min_instances": min_instances, "delta": delta, "threshold": threshold, "alpha": alpha}

    @property
    def name(self) -> str:
        return "PageHinkley"

    def update(self, value: float) -> None:
        self._n_seen += 1
        self._ph.update(value)
        self._drift = self._ph.drift_detected
        if self._drift:
            self._n_drifts += 1

    def drift_detected(self) -> bool:
        return self._drift

    def warning_detected(self) -> bool:
        return False

    def reset(self) -> None:
        from river.drift import PageHinkley
        self._ph = PageHinkley(**self._params)
        self._drift = False
        self._n_seen = 0
        self._n_drifts = 0

    def get_state(self) -> dict[str, Any]:
        return {"detector": self.name, "n_seen": self._n_seen, "n_drifts": self._n_drifts}


def create_detector(config: dict[str, Any]) -> DriftDetector:
    """Factory: build a drift detector from configuration."""
    name = config.get("detector", "adwin").lower()

    if name == "adwin":
        params = config.get("adwin", {})
        return ADWINDetector(**params)
    elif name == "ddm":
        return DDMDetector()
    elif name == "eddm":
        return EDDMDetector()
    elif name in ("page_hinkley", "pagehinkley", "ph"):
        return PageHinkleyDetector()
    else:
        raise ValueError(
            f"Unknown drift detector: {name}. "
            f"Supported: adwin, ddm, eddm, page_hinkley"
        )
