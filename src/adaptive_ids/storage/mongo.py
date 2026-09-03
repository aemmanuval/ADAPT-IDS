"""MongoDB storage for ADAPT-IDS — logs attacks, predictions, drift events.

Collections:
  - predictions:  every flow prediction with timestamp and features
  - attacks:      detected attacks with full flow details
  - drift_events: concept drift detections
  - model_log:    model retraining history
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from adaptive_ids.utils.logging import get_logger

logger = get_logger("storage.mongo")

try:
    from pymongo import MongoClient, DESCENDING
    from pymongo.errors import ConnectionFailure
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False


class MongoStorage:
    """MongoDB persistence layer for attack logs and system events."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "adapt_ids",
    ) -> None:
        if not HAS_MONGO:
            raise ImportError("pymongo not installed. Run: pip install pymongo")

        self.uri = uri
        self.db_name = database
        self.client: MongoClient | None = None
        self.db: Any = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._ensure_indexes()
            self._connected = True
            logger.info("Connected to MongoDB: %s/%s", self.uri, self.db_name)
            return True
        except Exception as e:
            logger.warning("MongoDB not available: %s. Using in-memory fallback.", e)
            self._connected = False
            return False

    def _ensure_indexes(self) -> None:
        self.db.predictions.create_index([("timestamp", DESCENDING)])
        self.db.predictions.create_index([("prediction", 1)])
        self.db.attacks.create_index([("timestamp", DESCENDING)])
        self.db.attacks.create_index([("severity", 1)])
        self.db.drift_events.create_index([("timestamp", DESCENDING)])

    @property
    def connected(self) -> bool:
        return self._connected

    def log_prediction(
        self,
        prediction: str,
        confidence: float,
        features: dict[str, float] | None = None,
        source_ip: str = "",
        dest_ip: str = "",
        dest_port: int = 0,
        protocol: str = "",
    ) -> str | None:
        doc = {
            "timestamp": datetime.now(timezone.utc),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "dest_port": dest_port,
            "protocol": protocol,
            "features": features or {},
        }

        if self._connected:
            result = self.db.predictions.insert_one(doc)
            return str(result.inserted_id)
        return None

    def log_attack(
        self,
        confidence: float,
        source_ip: str = "",
        dest_ip: str = "",
        dest_port: int = 0,
        protocol: str = "",
        features: dict[str, float] | None = None,
        severity: str = "medium",
        description: str = "",
    ) -> str | None:
        doc = {
            "timestamp": datetime.now(timezone.utc),
            "type": "attack_detected",
            "confidence": round(confidence, 4),
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "dest_port": dest_port,
            "protocol": protocol,
            "severity": severity,
            "description": description,
            "features": features or {},
            "acknowledged": False,
        }

        if self._connected:
            result = self.db.attacks.insert_one(doc)
            logger.warning(
                "ATTACK logged: %s → %s:%d (conf=%.2f, severity=%s)",
                source_ip, dest_ip, dest_port, confidence, severity,
            )
            return str(result.inserted_id)
        return None

    def log_drift_event(
        self,
        detector: str,
        stream_position: int,
        detector_state: dict[str, Any] | None = None,
    ) -> str | None:
        doc = {
            "timestamp": datetime.now(timezone.utc),
            "detector": detector,
            "stream_position": stream_position,
            "detector_state": detector_state or {},
        }

        if self._connected:
            result = self.db.drift_events.insert_one(doc)
            return str(result.inserted_id)
        return None

    def log_retrain(
        self,
        model_version: str,
        trigger: str,
        training_samples: int,
        training_time_s: float,
        metrics: dict[str, Any] | None = None,
    ) -> str | None:
        doc = {
            "timestamp": datetime.now(timezone.utc),
            "model_version": model_version,
            "trigger": trigger,
            "training_samples": training_samples,
            "training_time_s": round(training_time_s, 3),
            "metrics": metrics or {},
        }

        if self._connected:
            result = self.db.model_log.insert_one(doc)
            return str(result.inserted_id)
        return None

    def get_recent_attacks(self, limit: int = 50) -> list[dict]:
        if not self._connected:
            return []
        cursor = self.db.attacks.find(
            {}, {"_id": 0}
        ).sort("timestamp", DESCENDING).limit(limit)
        return [{**doc, "timestamp": doc["timestamp"].isoformat()} for doc in cursor]

    def get_recent_predictions(self, limit: int = 100) -> list[dict]:
        if not self._connected:
            return []
        cursor = self.db.predictions.find(
            {}, {"_id": 0}
        ).sort("timestamp", DESCENDING).limit(limit)
        return [{**doc, "timestamp": doc["timestamp"].isoformat()} for doc in cursor]

    def get_attack_stats(self) -> dict[str, Any]:
        if not self._connected:
            return {"total_attacks": 0, "total_predictions": 0}

        return {
            "total_attacks": self.db.attacks.count_documents({}),
            "total_predictions": self.db.predictions.count_documents({}),
            "total_drift_events": self.db.drift_events.count_documents({}),
            "total_retrains": self.db.model_log.count_documents({}),
            "unacknowledged_attacks": self.db.attacks.count_documents({"acknowledged": False}),
        }

    def close(self) -> None:
        if self.client:
            self.client.close()
            self._connected = False
