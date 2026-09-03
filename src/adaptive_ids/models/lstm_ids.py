"""LSTM-based IDS classifier for comparison with LightGBM.

Treats each flow's features as a single-timestep sequence.
For multi-step sequences, features can be windowed (future enhancement).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import joblib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

from adaptive_ids.utils.logging import get_logger

logger = get_logger("models.lstm")


class LSTMNetwork(nn.Module):
    """Bidirectional LSTM with attention for flow classification."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x.squeeze(1))
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context)


class LSTMClassifier:
    """Wrapper matching the BaselineIDS interface for drop-in comparison."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 1024,
        random_seed: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.model: LSTMNetwork | None = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.device = torch.device("cpu")
        self.training_time: float = 0.0
        self.metadata: dict[str, Any] = {}

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    @property
    def algorithm(self) -> str:
        return "lstm"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Training LSTM on %d samples, %d features", X.shape[0], X.shape[1])

        X_scaled = self.scaler.fit_transform(X)
        y_enc = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)

        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)
        y_tensor = torch.LongTensor(y_enc)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = LSTMNetwork(
            input_size=X.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes,
            dropout=self.dropout,
        ).to(self.device)

        class_counts = np.bincount(y_enc)
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        class_weights = torch.FloatTensor(weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        t0 = time.perf_counter()
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            avg_loss = total_loss / total
            acc = correct / total
            scheduler.step(avg_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("  Epoch %d/%d — loss=%.4f acc=%.4f", epoch + 1, self.epochs, avg_loss, acc)

        self.training_time = time.perf_counter() - t0
        self.metadata = {
            "algorithm": "lstm",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "epochs": self.epochs,
            "training_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "training_time_s": round(self.training_time, 3),
            "classes": self.label_encoder.classes_.tolist(),
            "final_loss": round(avg_loss, 4),
            "final_accuracy": round(acc, 4),
        }
        logger.info("LSTM training completed in %.2fs (final acc=%.4f)", self.training_time, acc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)

        with torch.no_grad():
            all_preds = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                outputs = self.model(batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.append(predicted.cpu().numpy())

        y_enc = np.concatenate(all_preds)
        return self.label_encoder.inverse_transform(y_enc)

    def predict_single(self, x: np.ndarray) -> str:
        return self.predict(x.reshape(1, -1))[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)

        with torch.no_grad():
            all_probs = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_state": self.model.state_dict() if self.model else None,
            "model_config": {
                "input_size": self.model.batch_norm.num_features if self.model else 0,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_classes": len(self.label_encoder.classes_) if hasattr(self.label_encoder, "classes_") else 2,
                "dropout": self.dropout,
            },
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "metadata": self.metadata,
        }
        joblib.dump(data, path)
        logger.info("LSTM model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.label_encoder = data["label_encoder"]
        obj.scaler = data["scaler"]
        obj.metadata = data["metadata"]
        obj.device = torch.device("cpu")
        obj.batch_size = 1024

        cfg = data["model_config"]
        obj.hidden_size = cfg["hidden_size"]
        obj.num_layers = cfg["num_layers"]
        obj.dropout = cfg["dropout"]

        obj.model = LSTMNetwork(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
        )
        obj.model.load_state_dict(data["model_state"])
        obj.model.eval()
        return obj

    def feature_importances(self) -> np.ndarray | None:
        return None
