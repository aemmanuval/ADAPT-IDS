"""Simple filesystem-based model registry for experiment tracking."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from adaptive_ids.utils.logging import get_logger

logger = get_logger("models.registry")


class ModelRegistry:
    """Tracks model versions and their metadata on disk."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "registry.json"
        self._load_index()

    def _load_index(self) -> None:
        if self.index_path.exists():
            with open(self.index_path) as fh:
                self.index: list[dict[str, Any]] = json.load(fh)
        else:
            self.index = []

    def _save_index(self) -> None:
        with open(self.index_path, "w") as fh:
            json.dump(self.index, fh, indent=2, default=str)

    def register(
        self,
        model_id: str,
        model_path: str | Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        entry = {
            "model_id": model_id,
            "model_path": str(model_path),
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **metadata,
        }
        self.index.append(entry)
        self._save_index()
        logger.info("Registered model: %s", model_id)
        return entry

    def list_models(self) -> list[dict[str, Any]]:
        return list(self.index)

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        for entry in self.index:
            if entry["model_id"] == model_id:
                return entry
        return None

    def latest(self) -> dict[str, Any] | None:
        return self.index[-1] if self.index else None
