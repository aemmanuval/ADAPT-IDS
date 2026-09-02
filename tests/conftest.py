"""Shared test fixtures for ADAPT-IDS."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_ids.config.settings import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_df():
    """Minimal CIC-IDS2017-like DataFrame for unit tests."""
    rng = np.random.RandomState(42)
    n = 1000
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2017-07-03 09:00", periods=n, freq="s"),
        "Flow Duration": rng.exponential(50000, n),
        "Total Fwd Packets": rng.poisson(5, n).astype(float),
        "Total Backward Packets": rng.poisson(3, n).astype(float),
        "Total Length of Fwd Packets": rng.exponential(500, n),
        "Total Length of Bwd Packets": rng.exponential(300, n),
        "Flow Bytes/s": rng.exponential(10000, n),
        "Flow Packets/s": rng.exponential(100, n),
        "Flow IAT Mean": rng.exponential(5000, n),
        "Flow IAT Std": rng.exponential(3000, n),
        "Fwd IAT Mean": rng.exponential(4000, n),
        "Bwd IAT Mean": rng.exponential(4000, n),
        "Fwd Packet Length Mean": rng.exponential(200, n),
        "Bwd Packet Length Mean": rng.exponential(150, n),
        "SYN Flag Count": rng.binomial(1, 0.3, n).astype(float),
        "ACK Flag Count": rng.binomial(1, 0.7, n).astype(float),
        "Init_Win_bytes_forward": rng.randint(0, 65535, n).astype(float),
        "Init_Win_bytes_backward": rng.randint(0, 65535, n).astype(float),
        "Label": ["BENIGN"] * 700 + ["DDoS"] * 150 + ["PortScan"] * 100 + ["Bot"] * 50,
    })
    df.loc[10, "Flow Bytes/s"] = np.inf
    df.loc[20, "Flow Packets/s"] = -np.inf
    df.loc[30, "Flow Duration"] = np.nan
    return df


@pytest.fixture
def sample_X_y(sample_df, config):
    """Preprocessed X, y arrays from sample_df."""
    from adaptive_ids.preprocessing.pipeline import PreprocessingPipeline
    pipeline = PreprocessingPipeline(config)
    _, X, y = pipeline.fit_transform(sample_df)
    return X, y


@pytest.fixture
def binary_sample_df(sample_df):
    """sample_df with binary labels (BENIGN / ATTACK)."""
    df = sample_df.copy()
    df["Label"] = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")
    return df
