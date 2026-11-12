"""Dirichlet label-skew partitioning (Person A).

Contract (shared with the rest of the team):

    get_partition(dataset, alpha, n_clients, seed) -> dict with
        "train":      list[np.ndarray]  train indices per client
        "test":       list[np.ndarray]  test indices per client (20% held out per client)
        "data_sizes": np.ndarray       d_i = len(train[i]); w_i = d_i / sum_j d_j

Indices refer to the dataset's training split (CIFAR-10: 50,000 images).
Output is deterministic for a given (dataset, alpha, n_clients, seed).
"""
from __future__ import annotations

import numpy as np

TEST_FRACTION = 0.2
MIN_CLIENT_SIZE = 10      # resample if any client gets fewer samples than this
MAX_RESAMPLES = 1000


def load_labels(dataset: str, data_root: str = "data/raw") -> np.ndarray:
    """Return the training-set label array for a dataset (labels only used)."""
    if dataset == "cifar10":
        from torchvision.datasets import CIFAR10

        return np.asarray(CIFAR10(root=data_root, train=True, download=True).targets)
    if dataset == "mnist":
        from torchvision.datasets import MNIST

        return np.asarray(MNIST(root=data_root, train=True, download=True).targets)
    if dataset == "femnist":
        raise NotImplementedError("FEMNIST uses its natural per-writer split; not implemented yet.")
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _dirichlet_split(labels: np.ndarray, alpha: float, n_clients: int,
                     rng: np.random.Generator) -> list[np.ndarray]:
    """Per-class Dirichlet(alpha) allocation of sample indices to clients."""
    classes = np.unique(labels)
    for _ in range(MAX_RESAMPLES):
        buckets: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            props = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for i, part in enumerate(np.split(idx, cuts)):
                buckets[i].extend(part.tolist())
        if min(len(b) for b in buckets) >= MIN_CLIENT_SIZE:
            return [np.array(sorted(b), dtype=np.int64) for b in buckets]
    raise RuntimeError(
        f"No partition with min client size >= {MIN_CLIENT_SIZE} after "
        f"{MAX_RESAMPLES} resamples (alpha={alpha}, n_clients={n_clients})."
    )


def get_partition(dataset: str, alpha: float, n_clients: int, seed: int,
                  labels: np.ndarray | None = None,
                  data_root: str = "data/raw") -> dict:
    """Partition a dataset across clients with Dirichlet(alpha) label skew.

    `labels` can be passed directly (used by tests); otherwise loaded from `dataset`.
    """
    if labels is None:
        labels = load_labels(dataset, data_root)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    client_idx = _dirichlet_split(labels, alpha, n_clients, rng)

    train, test = [], []
    for idx in client_idx:
        idx = rng.permutation(idx)
        n_test = max(1, int(round(TEST_FRACTION * len(idx))))
        test.append(np.sort(idx[:n_test]))
        train.append(np.sort(idx[n_test:]))

    data_sizes = np.array([len(t) for t in train], dtype=np.int64)
    return {"train": train, "test": test, "data_sizes": data_sizes}


def label_matrix(labels: np.ndarray, part: dict, n_classes: int | None = None) -> np.ndarray:
    """(clients x classes) count matrix over each client's train indices, for heatmaps."""
    labels = np.asarray(labels)
    n_classes = n_classes or int(labels.max()) + 1
    mat = np.zeros((len(part["train"]), n_classes), dtype=np.int64)
    for i, idx in enumerate(part["train"]):
        mat[i] = np.bincount(labels[idx], minlength=n_classes)
    return mat