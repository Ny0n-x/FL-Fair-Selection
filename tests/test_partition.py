import numpy as np
import pytest

from data.partition import get_partition, label_matrix

N_CLIENTS = 50


@pytest.fixture(scope="module")
def labels():
    # Synthetic stand-in for CIFAR-10: 50,000 samples, 10 balanced classes.
    return np.repeat(np.arange(10), 5000)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 100])
def test_same_seed_identical(labels, alpha):
    a = get_partition("cifar10", alpha, N_CLIENTS, 0, labels=labels)
    b = get_partition("cifar10", alpha, N_CLIENTS, 0, labels=labels)
    for i in range(N_CLIENTS):
        assert np.array_equal(a["train"][i], b["train"][i])
        assert np.array_equal(a["test"][i], b["test"][i])
    assert np.array_equal(a["data_sizes"], b["data_sizes"])


def test_different_seed_differs(labels):
    a = get_partition("cifar10", 0.5, N_CLIENTS, 0, labels=labels)
    b = get_partition("cifar10", 0.5, N_CLIENTS, 1, labels=labels)
    assert not np.array_equal(a["data_sizes"], b["data_sizes"])


@pytest.mark.parametrize("alpha", [0.1, 0.5, 100])
def test_disjoint_complete_and_split(labels, alpha):
    p = get_partition("cifar10", alpha, N_CLIENTS, 0, labels=labels)
    all_idx = np.concatenate([np.concatenate([t, e]) for t, e in zip(p["train"], p["test"])])
    assert len(all_idx) == len(labels)
    assert len(np.unique(all_idx)) == len(labels)  # no overlap, nothing dropped
    for t, e in zip(p["train"], p["test"]):
        assert len(e) >= 1 and len(t) >= 1
        assert abs(len(e) / (len(t) + len(e)) - 0.2) < 0.02
    assert np.array_equal(p["data_sizes"], [len(t) for t in p["train"]])


def test_alpha_controls_skew(labels):
    def mean_classes(alpha):
        p = get_partition("cifar10", alpha, N_CLIENTS, 0, labels=labels)
        m = label_matrix(labels, p, 10)
        return (m > 0.01 * m.sum(1, keepdims=True)).sum(1).mean()

    # alpha=0.1 should be much sparser than alpha=100
    assert mean_classes(0.1) < mean_classes(100) - 3