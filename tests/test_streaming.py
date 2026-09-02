"""Tests for the streaming abstraction."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_ids.streaming.stream import CSVStream, BatchStream


class TestCSVStream:
    def test_length(self):
        X = np.random.randn(100, 5)
        y = np.array(["A"] * 50 + ["B"] * 50)
        stream = CSVStream(X, y)
        assert len(stream) == 100

    def test_deterministic_order(self):
        X = np.arange(20).reshape(10, 2).astype(float)
        y = np.array(["A"] * 10)
        stream = CSVStream(X, y)
        indices = [event["index"] for event in stream]
        assert indices == list(range(10)), "Stream must yield events in order"

    def test_all_samples_yielded(self):
        n = 50
        X = np.random.randn(n, 3)
        y = np.array(["X"] * n)
        stream = CSVStream(X, y)
        events = list(stream)
        assert len(events) == n

    def test_reset(self):
        X = np.random.randn(10, 2)
        y = np.array(["A"] * 10)
        stream = CSVStream(X, y)
        _ = list(stream)
        assert stream.position == 10
        stream.reset()
        assert stream.position == 0
        events = list(stream)
        assert len(events) == 10

    def test_features_match(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array(["A", "B"])
        stream = CSVStream(X, y)
        events = list(stream)
        np.testing.assert_array_equal(events[0]["features"], [1.0, 2.0])
        assert events[1]["label"] == "B"


class TestBatchStream:
    def test_batch_sizes(self):
        X = np.random.randn(105, 3)
        y = np.array(["A"] * 105)
        stream = BatchStream(X, y, batch_size=20)
        batches = list(stream)
        sizes = [b["features"].shape[0] for b in batches]
        assert sum(sizes) == 105
        assert sizes[-1] == 5  # remainder batch

    def test_end_of_stream(self):
        X = np.random.randn(40, 2)
        y = np.array(["A"] * 40)
        stream = BatchStream(X, y, batch_size=40)
        batches = list(stream)
        assert len(batches) == 1
        assert batches[0]["batch_end"] == 40
