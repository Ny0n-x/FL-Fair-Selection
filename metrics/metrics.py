"""Fairness metrics over per-client accuracy and selection counts.

Every client counts once. Variance and std use ddof=0.
Worst-10% is the mean of the lowest ceil(0.1 * N) clients.
"""
import math

import numpy as np


def _arr(x):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("expected a non-empty 1-D sequence")
    if np.isnan(a).any():
        raise ValueError("NaN in input")
    return a


def jain_index(x):
    """(sum x)^2 / (n * sum x^2). Ranges from 1/n to 1, where 1 means all equal."""
    a = _arr(x)
    denom = a.size * np.sum(a ** 2)
    return 1.0 if denom == 0 else float(np.sum(a) ** 2 / denom)


def coeff_variation(x):
    """std / mean."""
    a = _arr(x)
    m = a.mean()
    return float("nan") if m == 0 else float(a.std(ddof=0) / m)


def variance(x):
    return float(_arr(x).var(ddof=0))


def min_acc(x):
    return float(_arr(x).min())


def worst_frac_mean(x, frac=0.1):
    """Mean of the lowest ceil(frac * n) values."""
    a = np.sort(_arr(x))
    k = max(1, math.ceil(frac * a.size))
    return float(a[:k].mean())


def entropy_normalized(x):
    """Entropy (nats) of x / sum(x). Max is log(n)."""
    a = _arr(x)
    if (a < 0).any():
        raise ValueError("entropy needs non-negative values")
    s = a.sum()
    if s == 0:
        return float("nan")
    p = a[a > 0] / s
    return float(-(p * np.log(p)).sum())


def selection_freq_variance(selection_counts):
    """Variance of per-client selection counts."""
    return variance(selection_counts)


def outcome_metrics(client_acc):
    """All outcome metrics for one vector of per-client accuracies."""
    a = _arr(client_acc)
    return {
        "mean_client_acc": float(a.mean()),
        "variance": variance(a),
        "std": float(a.std(ddof=0)),
        "min": min_acc(a),
        "worst10_mean": worst_frac_mean(a, 0.1),
        "jain": jain_index(a),
        "cv": coeff_variation(a),
        "entropy": entropy_normalized(a),
    }


def rounds_to_target(global_acc, target, smooth=5):
    """First round (1-based) where the `smooth`-round moving average reaches `target`.
    Returns None if it never does."""
    a = _arr(global_acc)
    w = max(1, min(smooth, a.size))
    ma = np.convolve(a, np.ones(w) / w, mode="valid")  # ma[j] covers rounds j+1 .. j+w
    hit = np.nonzero(ma >= target)[0]
    return None if hit.size == 0 else int(hit[0] + w)


def size_buckets(data_sizes, n_buckets=4):
    """Equal-count bucket index per client by data size. 0 holds the smallest clients."""
    sizes = _arr(data_sizes)
    b = np.empty(sizes.size, dtype=int)
    for i, idx in enumerate(np.array_split(np.argsort(sizes, kind="stable"), n_buckets)):
        b[idx] = i
    return b
