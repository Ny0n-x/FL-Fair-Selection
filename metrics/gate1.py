"""Gate 1 diagnostic for Random runs.

Per run: Spearman(count, acc), Spearman(size, acc) and worst-10% / mean.
Pooled over seeds per (dataset, alpha): the per-seed rho are combined with Fisher's z,
using the Fieller-Hartley-Pearson variance 1.06 / (N - 3) for Spearman's rho.
With N=50, |rho| needs to be above about 0.28 for p < 0.05 in a single run.

Usage: python -m metrics.gate1 results/random_*.json [--out gate1.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from metrics.loader import KEYS, load_run
from metrics.metrics import worst_frac_mean

FHP = 1.06  # variance factor for atanh(Spearman rho)
PAIRS = (("count", "counts"), ("size", "sizes"))


def gate1(r):
    acc = np.asarray(r[KEYS["client_acc"]], dtype=float)
    counts = np.asarray(r[KEYS["counts"]], dtype=float)
    sizes = np.asarray(r[KEYS["sizes"]], dtype=float)
    rho_c, p_c = spearmanr(counts, acc)
    rho_s, p_s = spearmanr(sizes, acc)
    return {
        "dataset": r[KEYS["dataset"]], "alpha": r[KEYS["alpha"]], "seed": r[KEYS["seed"]],
        "n_clients": int(acc.size),
        "rho_count_acc": float(rho_c), "p_count_acc": float(p_c),
        "rho_size_acc": float(rho_s), "p_size_acc": float(p_s),
        "worst10_over_mean": worst_frac_mean(acc) / float(acc.mean()),
        "count_mean": float(counts.mean()), "count_std": float(counts.std(ddof=0)),
    }


def fisher_combine(rhos, n, level=0.95):
    """Combine Spearman rho from independent runs with n clients each.
    Returns (pooled rho, CI low, CI high, two-sided p). NaN rho are dropped."""
    r = np.asarray(rhos, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0 or n <= 3:
        return (float("nan"),) * 4
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = np.sqrt(FHP / (n - 3) / r.size)
    zbar = z.mean()
    h = norm.ppf(0.5 + level / 2) * se
    p = 2 * norm.sf(abs(zbar) / se)
    return float(np.tanh(zbar)), float(np.tanh(zbar - h)), float(np.tanh(zbar + h)), float(p)


def pool(per_run, alpha_sig=0.05):
    """One row per (dataset, alpha) from a DataFrame of gate1() rows."""
    rows = []
    for (ds, a), g in per_run.groupby(["dataset", "alpha"]):
        n = int(g["n_clients"].min())
        row = {"dataset": ds, "alpha": a, "n_seeds": int(g["seed"].nunique()), "n_clients": n}
        for name, _ in PAIRS:
            rho, lo, hi, p = fisher_combine(g[f"rho_{name}_acc"], n)
            row.update({f"rho_{name}": rho, f"rho_{name}_lo": lo, f"rho_{name}_hi": hi,
                        f"p_{name}": p,
                        f"sig_seeds_{name}": int((g[f"p_{name}_acc"] < alpha_sig).sum())})
        w = g["worst10_over_mean"]
        row["worst10_over_mean"] = float(w.mean())
        row["worst10_over_mean_std"] = float(w.std(ddof=1)) if len(w) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def gate1_table(paths):
    return pd.DataFrame([gate1(load_run(p)) for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None, help="write the pooled table as CSV")
    a = ap.parse_args()
    per_run = gate1_table(a.files)
    for f, g in zip(a.files, per_run.to_dict("records")):
        print(f"{f}\n  alpha={g['alpha']} seed={g['seed']}"
              f"  counts {g['count_mean']:.1f} +- {g['count_std']:.1f}\n"
              f"  (a) rho(count, acc) = {g['rho_count_acc']:+.3f}  p={g['p_count_acc']:.3g}\n"
              f"  (b) rho(size,  acc) = {g['rho_size_acc']:+.3f}  p={g['p_size_acc']:.3g}\n"
              f"  (c) worst10 / mean  = {g['worst10_over_mean']:.3f}")
    pooled = pool(per_run)
    print("\nPooled over seeds (Fisher z, 95% CI; sig = seeds with p < 0.05 on their own)")
    for g in pooled.to_dict("records"):
        print(f"  {g['dataset']} alpha={g['alpha']:g}  seeds={g['n_seeds']}  N={g['n_clients']}")
        for name, label in (("count", "(a) rho(count, acc)"), ("size", "(b) rho(size,  acc)")):
            print(f"    {label} = {g[f'rho_{name}']:+.3f} "
                  f"[{g[f'rho_{name}_lo']:+.3f}, {g[f'rho_{name}_hi']:+.3f}]"
                  f"  p={g[f'p_{name}']:.3g}  sig {g[f'sig_seeds_{name}']}/{g['n_seeds']}")
        sd = g["worst10_over_mean_std"]
        print(f"    (c) worst10 / mean  = {g['worst10_over_mean']:.3f}"
              + ("" if np.isnan(sd) else f" +- {sd:.3f}"))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pooled.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
