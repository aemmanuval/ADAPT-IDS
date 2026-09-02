"""Experiment metadata tracking — saves every run's configuration and results."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from adaptive_ids.utils.logging import get_logger

logger = get_logger("experiments.tracker")


class ExperimentTracker:
    """Records experiment metadata for reproducibility."""

    def __init__(self, results_dir: str | Path = "results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.record: dict[str, Any] = {
            "experiment_id": f"exp_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def set(self, key: str, value: Any) -> None:
        self.record[key] = value

    def update(self, data: dict[str, Any]) -> None:
        self.record.update(data)

    def save(self, subdir: str = "", filename: str = "experiment.json") -> Path:
        out_dir = self.results_dir / subdir if subdir else self.results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        with open(path, "w") as fh:
            json.dump(self.record, fh, indent=2, default=str)
        logger.info("Experiment record saved to %s", path)
        return path
