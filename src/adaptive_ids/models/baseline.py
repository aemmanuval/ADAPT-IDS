"""Baseline IDS classifiers — LightGBM and Random Forest."""

from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from adaptive_ids.utils.logging import get_logger

logger = get_logger("models.baseline")


class BaselineIDS:
    """Thin wrapper around a classifier with timing and metadata."""

    def __init__(
        self,
        algorithm: str,
        params: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.algorithm = algorithm
        self.params = params or {}
        self.random_seed = random_seed
        self.label_encoder = LabelEncoder()
        self.model: Any = None
        self.training_time: float = 0.0
        self.metadata: dict[str, Any] = {}

        self._build_model()

    def _build_model(self) -> None:
        if self.algorithm == "lightgbm":
            import lightgbm as lgb
            p = {**self.params, "random_state": self.random_seed}
            self.model = lgb.LGBMClassifier(**p)
        elif self.algorithm == "random_forest":
            p = {**self.params, "random_state": self.random_seed}
            self.model = RandomForestClassifier(**p)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info(
            "Training %s on %d samples, %d features",
            self.algorithm, X.shape[0], X.shape[1],
        )
        y_enc = self.label_encoder.fit_transform(y)

        t0 = time.perf_counter()
        self.model.fit(X, y_enc)
        self.training_time = time.perf_counter() - t0

        self.metadata.update({
            "algorithm": self.algorithm,
            "params": self.params,
            "random_seed": self.random_seed,
            "training_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "training_time_s": round(self.training_time, 3),
            "classes": self.label_encoder.classes_.tolist(),
        })
        logger.info("Training completed in %.2fs", self.training_time)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_single(self, x: np.ndarray) -> str:
        """Predict a single sample (1-D array)."""
        x_2d = x.reshape(1, -1)
        return self.predict(x_2d)[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "metadata": self.metadata,
            "algorithm": self.algorithm,
            "params": self.params,
        }
        joblib.dump(data, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineIDS":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.label_encoder = data["label_encoder"]
        obj.metadata = data["metadata"]
        obj.algorithm = data["algorithm"]
        obj.params = data["params"]
        obj.training_time = data["metadata"].get("training_time_s", 0)
        obj.random_seed = data["metadata"].get("random_seed", 42)
        return obj

    def feature_importances(self) -> np.ndarray | None:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None
