"""FastAPI service for ADAPT-IDS production deployment.

Endpoints:
  POST /predict        — classify a single flow or batch
  POST /predict/batch  — classify a CSV upload
  GET  /model/info     — current model metadata
  GET  /drift/status   — drift detection status
  GET  /drift/events   — recent drift events
  GET  /health         — service health check

Run:
    uvicorn adaptive_ids.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from adaptive_ids.utils.logging import get_logger

logger = get_logger("api")

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")


app = FastAPI(
    title="ADAPT-IDS API",
    description="Adaptive Intrusion Detection Under Concept & Feature Drift",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_detector = None
_storage = None
_drift_events: list[dict] = []
_stats = {"predictions": 0, "attacks_detected": 0, "drift_events": 0, "start_time": time.time()}


class FlowFeatures(BaseModel):
    """Single network flow for prediction."""
    features: list[float]


class BatchFlowFeatures(BaseModel):
    """Batch of flows for prediction."""
    flows: list[list[float]]


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    drift_warning: bool


class BatchPredictionResponse(BaseModel):
    predictions: list[str]
    n_attack: int
    n_benign: int
    drift_warning: bool


def load_model(model_path: str | Path | None = None):
    """Load the IDS model at startup."""
    global _model, _detector
    from adaptive_ids.models.baseline import BaselineIDS
    from adaptive_ids.drift.detectors import UnsupervisedDriftDetector

    if model_path is None:
        root = Path(__file__).resolve().parents[3]
        candidates = [
            root / "results" / "cross_dataset" / "lightgbm_combined.joblib",
            root / "results" / "temporal" / "lightgbm" / "lightgbm_temporal.joblib",
            root / "results" / "baseline" / "lightgbm" / "lightgbm_baseline.joblib",
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break

    if model_path and Path(model_path).exists():
        _model = BaselineIDS.load(model_path)
        _detector = UnsupervisedDriftDetector(delta=0.002)
        logger.info("Model loaded: %s", model_path)
    else:
        logger.warning("No model found. Predict endpoints will return errors.")


@app.on_event("startup")
async def startup():
    global _storage
    load_model()
    try:
        from adaptive_ids.storage.mongo import MongoStorage
        _storage = MongoStorage()
        if not _storage.connect():
            _storage = None
    except Exception:
        _storage = None


@app.get("/")
async def root():
    return {
        "service": "ADAPT-IDS API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": ["/predict", "/predict/batch", "/health", "/model/info", "/drift/status", "/drift/events", "/attacks"],
    }


@app.get("/attacks")
async def get_attacks(limit: int = 50):
    """Get recent attack detections from MongoDB."""
    if _storage and _storage.connected:
        attacks = _storage.get_recent_attacks(limit)
        stats = _storage.get_attack_stats()
        return {"attacks": attacks, "stats": stats}
    return {"attacks": list(_drift_events[-limit:]), "stats": _stats, "source": "in-memory"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "uptime_s": round(time.time() - _stats["start_time"], 1),
        "predictions": _stats["predictions"],
        "attacks_detected": _stats["attacks_detected"],
        "drift_events": _stats["drift_events"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(flow: FlowFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    x = np.array(flow.features).reshape(1, -1)
    prediction = _model.predict(x)[0]

    proba = _model.predict_proba(x)
    confidence = float(proba.max())

    drift_warning = False
    if _detector:
        _detector.update(confidence)
        if _detector.drift_detected():
            drift_warning = True
            _stats["drift_events"] += 1
            _drift_events.append({
                "position": _stats["predictions"],
                "confidence": confidence,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

    _stats["predictions"] += 1
    if prediction == "ATTACK":
        _stats["attacks_detected"] += 1
        if _storage and _storage.connected:
            _storage.log_attack(confidence=confidence)

    if _storage and _storage.connected:
        _storage.log_prediction(prediction=prediction, confidence=confidence)

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        drift_warning=drift_warning,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchFlowFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    X = np.array(batch.flows)
    predictions = _model.predict(X).tolist()

    n_attack = predictions.count("ATTACK")
    n_benign = predictions.count("BENIGN")

    proba = _model.predict_proba(X)
    confidences = proba.max(axis=1)

    drift_warning = False
    if _detector:
        for conf in confidences:
            _detector.update(float(conf))
            if _detector.drift_detected():
                drift_warning = True
                _stats["drift_events"] += 1

    _stats["predictions"] += len(predictions)
    _stats["attacks_detected"] += n_attack

    return BatchPredictionResponse(
        predictions=predictions,
        n_attack=n_attack,
        n_benign=n_benign,
        drift_warning=drift_warning,
    )


@app.get("/model/info")
async def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return _model.metadata


@app.get("/drift/status")
async def drift_status():
    if _detector is None:
        return {"detector": None}
    return _detector.get_state()


@app.get("/drift/events")
async def drift_events(limit: int = 50):
    return {"events": _drift_events[-limit:], "total": len(_drift_events)}
