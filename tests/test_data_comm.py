"""Partitioners and byte counting."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, MetricRecord, RecordDict

from fedlab.comm import recorddict_bytes
from fedlab.data import (
    label_histogram,
    partition_dirichlet,
    partition_iid,
    partition_shards,
    partition_writers,
)

LABELS = np.repeat(np.arange(10), 600)  # 6000 samples, 10 balanced classes


def _assert_exact_cover(parts: list[np.ndarray], n: int) -> None:
    allix = np.concatenate(parts)
    assert len(allix) == n, "samples lost or duplicated"
    assert np.array_equal(np.sort(allix), np.arange(n))


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 100.0])
def test_dirichlet_preserves_dataset(alpha: float) -> None:
    _assert_exact_cover(partition_dirichlet(LABELS, 20, alpha, seed=1), len(LABELS))


def test_dirichlet_skew_decreases_with_alpha() -> None:
    def mean_max_share(alpha: float) -> float:
        h = label_histogram(LABELS, partition_dirichlet(LABELS, 20, alpha, seed=1), 10)
        h = h[h.sum(1) > 0]
        return float((h.max(1) / h.sum(1)).mean())

    shares = [mean_max_share(a) for a in (0.1, 0.5, 1.0, 100.0)]
    assert shares == sorted(shares, reverse=True)
    assert shares[0] > 0.6  # alpha=0.1: most of a client's data is one class
    assert shares[-1] < 0.2  # alpha=100: close to uniform (0.1)


def test_shards_two_per_client() -> None:
    parts = partition_shards(LABELS, 30, seed=0)
    _assert_exact_cover(parts, len(LABELS))  # 6000 / 60 shards = 100, no remainder
    assert all(len(p) == 200 for p in parts)
    classes = [len(np.unique(LABELS[p])) for p in parts]
    assert max(classes) <= 4 and np.mean(classes) <= 2.0


def test_iid_preserves_dataset() -> None:
    parts = partition_iid(LABELS, 7, seed=0)
    _assert_exact_cover(parts, len(LABELS))


def test_writer_split_is_disjoint_and_fixed() -> None:
    writers = np.repeat([f"w{i}" for i in range(50)], 40)
    train, test = partition_writers(writers, num_writers=10, subset_seed=0)
    allix = np.concatenate([*train, test])
    assert len(np.unique(allix)) == len(allix) == 10 * 40
    for ix in train:
        assert len(np.unique(writers[ix])) == 1
    train2, test2 = partition_writers(writers, num_writers=10, subset_seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(train, train2, strict=True))
    assert np.array_equal(test, test2)


def test_byte_count_is_numel_times_itemsize() -> None:
    arrays = [np.zeros((3, 5), np.float32), np.ones(7, np.float64), np.arange(4, dtype=np.int16)]
    content = RecordDict(
        {
            "a": ArrayRecord({"w": Array(arrays[0]), "b": Array(arrays[1])}),
            "c": ArrayRecord({"x": Array(arrays[2])}),
            "metrics": MetricRecord({"num-examples": 3}),
        }
    )
    raw, serialized = recorddict_bytes(content)
    assert raw == sum(a.size * a.itemsize for a in arrays) == 15 * 4 + 7 * 8 + 4 * 2
    assert serialized > raw  # np.save headers and key strings
