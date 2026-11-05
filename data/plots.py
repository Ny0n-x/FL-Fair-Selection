"""Partition diagnostics: data-size histogram + label heatmap per alpha.

Usage:  python -m data.plots --dataset cifar10 --n_clients 50 --seed 0
Saves PNGs to results/figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.partition import get_partition, label_matrix, load_labels

ALPHAS = (0.1, 0.5, 100)


def plot_alpha(labels, alpha, n_clients, seed, out_dir: Path):
    part = get_partition("cifar10", alpha, n_clients, seed, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(part["data_sizes"], bins=15, color="#4C72B0", edgecolor="white")
    ax.set(xlabel="client train size", ylabel="# clients", title=f"Data sizes, alpha={alpha}")
    fig.tight_layout()
    fig.savefig(out_dir / f"datasize_hist_alpha{alpha}.png", dpi=150)
    plt.close(fig)

    mat = label_matrix(labels, part)
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set(xlabel="class", ylabel="client", title=f"Label distribution, alpha={alpha}")
    fig.colorbar(im, ax=ax, label="# samples")
    fig.tight_layout()
    fig.savefig(out_dir / f"label_heatmap_alpha{alpha}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--n_clients", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.dataset)
    for a in ALPHAS:
        plot_alpha(labels, a, args.n_clients, args.seed, out_dir)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()