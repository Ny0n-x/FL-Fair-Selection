"""Gate 1 diagnostic on Random runs (Person C makes the go/no-go call).

For each run prints:
  (a) Spearman(selection count, final client accuracy)
  (b) Spearman(data size, final client accuracy)
  (c) worst-10% mean accuracy / mean client accuracy
With N=50 a Spearman rho needs |rho| > ~0.28 for p < 0.05, so read rho together with p.

Usage: python -m metrics.gate1 results/random_cifar10_0.1_count_0.json [more files ...]
"""
import argparse

import numpy as np
from scipy.stats import spearmanr

from metrics.loader import KEYS, load_run
from metrics.metrics import worst_frac_mean


def gate1(r):
    acc = np.asarray(r[KEYS["client_acc"]], dtype=float)
    counts = np.asarray(r[KEYS["counts"]], dtype=float)
    sizes = np.asarray(r[KEYS["sizes"]], dtype=float)
    rho_c, p_c = spearmanr(counts, acc)
    rho_s, p_s = spearmanr(sizes, acc)
    return {
        "alpha": r[KEYS["alpha"]], "seed": r[KEYS["seed"]],
        "rho_count_acc": float(rho_c), "p_count_acc": float(p_c),
        "rho_size_acc": float(rho_s), "p_size_acc": float(p_s),
        "worst10_over_mean": worst_frac_mean(acc) / float(acc.mean()),
        "count_mean": float(counts.mean()), "count_std": float(counts.std(ddof=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    for f in ap.parse_args().files:
        g = gate1(load_run(f))
        print(f"{f}\n  alpha={g['alpha']} seed={g['seed']}"
              f"  counts {g['count_mean']:.1f} +- {g['count_std']:.1f}\n"
              f"  (a) rho(count, acc) = {g['rho_count_acc']:+.3f}  p={g['p_count_acc']:.3g}\n"
              f"  (b) rho(size,  acc) = {g['rho_size_acc']:+.3f}  p={g['p_size_acc']:.3g}\n"
              f"  (c) worst10 / mean  = {g['worst10_over_mean']:.3f}")


if __name__ == "__main__":
    main()
