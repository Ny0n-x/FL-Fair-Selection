"""Fairness metrics over per-client accuracies and selection counts.

Conventions (agree these with the team before results come in):
- Every client counts once (unweighted). That is the individual-fairness notion we use.
  Jain's index here is over individual clients, not demographic groups (FairFed's notion).
- Accuracies can be fractions in [0, 1] or percentages. Jain, CV and entropy do not depend on
  the scale; variance, std, min and worst-10% do, so report them in percentage points.
- Variance and std are population (ddof=0) values over the N clients.
- Worst-10% is the mean of the lowest ceil(0.1 * N) clients (N=50 -> 5 clients).
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
    """(sum x)^2 / (n * sum x^2), in [1/n, 1]; 1 means all clients equal."""
    a = _arr(x)
    denom = a.size * np.sum(a ** 2)
    return 1.0 if denom == 0 else float(np.sum(a) ** 2 / denom)


def coeff_variation(x):
    """std / mean (population std)."""
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
    """Shannon entropy (nats) of x / sum(x). Maximum is log(n) when all clients are equal."""
    a = _arr(x)
    if (a < 0).any():
        raise ValueError("entropy needs non-negative values")
    s = a.sum()
    if s == 0:
        return float("nan")
    p = a[a > 0] / s
    return float(-(p * np.log(p)).sum())


def selection_freq_variance(selection_counts):
    """Population variance of per-client selection counts (participation fairness)."""
    return variance(selection_counts)


def outcome_metrics(client_acc):
    """All outcome-fairness metrics for one vector of per-client accuracies."""
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
    """First round (1-based) where the trailing `smooth`-round mean of global accuracy
    reaches `target`; None if it never does. Smoothing stops one lucky round from counting."""
    a = _arr(global_acc)
    w = max(1, min(smooth, a.size))
    ma = np.convolve(a, np.ones(w) / w, mode="valid")  # ma[j] covers rounds j+1 .. j+w
    hit = np.nonzero(ma >= target)[0]
    return None if hit.size == 0 else int(hit[0] + w)


def size_buckets(data_sizes, n_buckets=4):
    """Bucket index per client (0 = smallest data), equal-count buckets by data-size rank.
    Ties are broken by client id, so with near-equal sizes the buckets carry no meaning."""
    sizes = _arr(data_sizes)
    b = np.empty(sizes.size, dtype=int)
    for i, idx in enumerate(np.array_split(np.argsort(sizes, kind="stable"), n_buckets)):
        b[idx] = i
    return b
