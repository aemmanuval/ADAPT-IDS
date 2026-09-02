"""Drift-event logging and persistence."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from adaptive_ids.utils.logging import get_logger

logger = get_logger("drift.events")


@dataclass
class DriftEvent:
    """Structured record of a single detected drift."""
    event_id: int
    event_type: str  # "drift" or "warning"
    detector: str
    stream_position: int
    timestamp: str = ""
    model_version: str = "baseline_v1"
    window_size: int = 0
    metric_context: dict[str, Any] = field(default_factory=dict)
    detector_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DriftEventLogger:
    """Collects and persists drift events."""

    def __init__(self, log_dir: str | Path = "results/drift") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[DriftEvent] = []
        self._next_id = 1

    def log_drift(
        self,
        detector_name: str,
        stream_position: int,
        *,
        timestamp: str = "",
        model_version: str = "baseline_v1",
        window_size: int = 0,
        metric_context: dict[str, Any] | None = None,
        detector_state: dict[str, Any] | None = None,
    ) -> DriftEvent:
        event = DriftEvent(
            event_id=self._next_id,
            event_type="drift",
            detector=detector_name,
            stream_position=stream_position,
            timestamp=timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
            model_version=model_version,
            window_size=window_size,
            metric_context=metric_context or {},
            detector_state=detector_state or {},
        )
        self.events.append(event)
        self._next_id += 1
        logger.info(
            "DRIFT #%d at position %d (detector=%s)",
            event.event_id, stream_position, detector_name,
        )
        return event

    def save_csv(self, filename: str = "drift_events.csv") -> Path:
        path = self.log_dir / filename
        if not self.events:
            logger.info("No drift events to save")
            return path

        rows = [e.to_dict() for e in self.events]
        flat_rows = []
        for r in rows:
            flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
            for k, v in r.items():
                if isinstance(v, dict):
                    flat[k] = json.dumps(v)
            flat_rows.append(flat)

        keys = list(flat_rows[0].keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_rows)

        logger.info("Saved %d drift events to %s", len(self.events), path)
        return path

    def save_json(self, filename: str = "drift_events.json") -> Path:
        path = self.log_dir / filename
        data = [e.to_dict() for e in self.events]
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.info("Saved %d drift events (JSON) to %s", len(self.events), path)
        return path

    @property
    def count(self) -> int:
        return len(self.events)
