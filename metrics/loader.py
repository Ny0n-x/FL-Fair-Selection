"""Load results JSONs, compute metrics per run, aggregate mean +- std over seeds.

Reads the format in results/schema.md: run identity (selector, dataset, alpha, good_def,
seed) comes from `config`. load_run() maps every file onto the canonical names in KEYS, and
also accepts the aliases in ALIASES (main.py currently writes `cfg` / `selected_ids`).

Usage: python -m metrics.loader [results_dir] [--out tables/summary.csv]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from metrics.metrics import (outcome_metrics, rounds_to_target, selection_freq_variance,
                             size_buckets)

KEYS = {
    "selector": "selector",
    "dataset": "dataset",
    "alpha": "alpha",
    "good": "good_def",
    "seed": "seed",
    "selected": "selected",                  # list[list[int]], one per round
    "global_acc": "global_acc",              # list[float], one per round
    "client_acc": "client_acc",              # list[float], final per-client accuracy, length N
    "client_acc_rounds": "client_acc_rounds",  # optional: list[list[float]], one per round
    "counts": "selection_counts",            # list[int], length N
    "sizes": "data_sizes",                   # list[int], length N
}
ALIASES = {"selected": ["selected_ids"], "client_acc": ["client_acc_final"]}
IDENTITY = ("selector", "dataset", "alpha", "good", "seed")
OPTIONAL = ("client_acc_rounds",)
GROUP = ["selector", "dataset", "alpha", "good"]
LAST = 10
# Rounds-to-target: target = TARGET_FRAC * Random's last-10-round global accuracy for the same
# (dataset, alpha), averaged over seeds. Shared target, so methods are compared on one bar.
REFERENCE, TARGET_FRAC, SMOOTH = "random", 0.9, 5


def load_run(path):
    """Read one results JSON and return it with canonical key names (see KEYS)."""
    with open(path) as f:
        raw = json.load(f)
    cfg = raw.get("config") or raw.get("cfg") or {}
    r = dict(raw)
    for k in IDENTITY:  # identity lives in config; a top-level value also counts
        name = KEYS[k]
        if name not in r and name in cfg:
            r[name] = cfg[name]
    for k, alts in ALIASES.items():
        if KEYS[k] not in r:
            for a in alts:
                if a in raw:
                    r[KEYS[k]] = raw[a]
                    break
    missing = [v for k, v in KEYS.items() if k not in OPTIONAL and v not in r]
    if missing:
        raise KeyError(f"{os.path.basename(path)} is missing keys: {missing}")
    r[KEYS["alpha"]] = float(r[KEYS["alpha"]])
    return r


def run_metrics(r):
    """Per-run row: final-round metrics plus last-10-round means where the data allows."""
    row = {k: r[KEYS[k]] for k in ("selector", "dataset", "alpha", "good", "seed")}
    g = np.asarray(r[KEYS["global_acc"]], dtype=float)
    row["global_acc_final"] = float(g[-1])
    row["global_acc_last10"] = float(g[-LAST:].mean())
    for k, v in outcome_metrics(r[KEYS["client_acc"]]).items():
        row[f"{k}_final"] = v
    per_round = r.get(KEYS["client_acc_rounds"])
    if per_round:
        tail = [outcome_metrics(a) for a in per_round[-LAST:]]
        for k in tail[0]:
            row[f"{k}_last10"] = float(np.mean([t[k] for t in tail]))
    row["sel_freq_var"] = selection_freq_variance(r[KEYS["counts"]])
    row["_global_acc"] = g
    return row


def add_rounds_to_target(df):
    """Adds target_acc and rounds_to_target. Runs that never reach the target get NaN,
    so check n_reached in the summary before reading a mean."""
    ref = (df[df["selector"] == REFERENCE].groupby(["dataset", "alpha"])["global_acc_last10"]
           .mean() * TARGET_FRAC)
    df["target_acc"] = [ref.get((d, a), np.nan) for d, a in zip(df["dataset"], df["alpha"])]
    rtt = []
    for g, t in zip(df["_global_acc"], df["target_acc"]):
        hit = None if np.isnan(t) else rounds_to_target(g, t, SMOOTH)
        rtt.append(np.nan if hit is None else hit)
    df["rounds_to_target"] = rtt
    df["reached_target"] = ~np.isnan(df["rounds_to_target"])
    return df.drop(columns="_global_acc")


def load_dir(results_dir="results"):
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"no JSON files in {results_dir}")
    return add_rounds_to_target(pd.DataFrame([run_metrics(load_run(p)) for p in paths]))


def summarize(df):
    """mean and std over seeds for each (selector, dataset, alpha, good) cell."""
    metric_cols = [c for c in df.columns if c not in GROUP + ["seed", "reached_target"]]
    agg = df.groupby(GROUP)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = df.groupby(GROUP)["seed"].nunique()
    agg["n_reached"] = df.groupby(GROUP)["reached_target"].sum()
    return agg.reset_index()


def size_breakdown(runs, n_buckets=4):
    """Per-client accuracy and selection count by data-size bucket (0 = smallest clients).
    Buckets are formed within each run, since partitions differ across seeds.
    Returns one row per (method, alpha, bucket): mean +- std over seeds."""
    rows = []
    for r in runs:
        sizes = np.asarray(r[KEYS["sizes"]], dtype=float)
        acc = np.asarray(r[KEYS["client_acc"]], dtype=float)
        counts = np.asarray(r[KEYS["counts"]], dtype=float)
        b = size_buckets(sizes, n_buckets)
        for i in range(n_buckets):
            m = b == i
            rows.append({**{k: r[KEYS[k]] for k in GROUP + ["seed"]}, "bucket": i,
                         "size_min": sizes[m].min(), "size_max": sizes[m].max(),
                         "acc": acc[m].mean(), "count": counts[m].mean()})
    df = pd.DataFrame(rows)
    agg = df.groupby(GROUP + ["bucket"]).agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        count_mean=("count", "mean"), count_std=("count", "std"),
        size_min=("size_min", "min"), size_max=("size_max", "max"),
        n_seeds=("seed", "nunique"))
    return agg.reset_index()


def fmt_table(summary, cols):
    """Readable 'mean +- std' table for the given metric names."""
    out = summary[GROUP + ["n_seeds"]].copy()
    for c in cols:
        out[c] = [f"{m:.4f} +- {s:.4f}" if not np.isnan(s) else f"{m:.4f}"
                  for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_std"])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = summarize(load_dir(a.results_dir))
    cols = ["global_acc_final", "worst10_mean_final", "min_final", "variance_final",
            "jain_final", "cv_final", "sel_freq_var", "rounds_to_target"]
    t = fmt_table(s, cols)
    t["n_reached"] = s["n_reached"]
    print(t.to_string(index=False))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        s.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
