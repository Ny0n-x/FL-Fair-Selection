"""Write fake results JSONs in the results/schema.md format, to exercise the loader and
plots without training. The numbers are made up; never mix them with real results.

Usage: python -m metrics.make_fake_results [out_dir]
"""
import json
import os
import sys

import numpy as np

N, K, ROUNDS = 50, 10, 100


def fake_run(selector, alpha, seed, good="count"):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(200, 2000, N) if alpha < 1 else np.full(N, 800)
    if selector != "random":
        order = rng.permutation(N)
        selected = [order[(r * K) % N:(r * K) % N + K].tolist() for r in range(ROUNDS)]
    else:
        selected = [rng.choice(N, K, replace=False).tolist() for _ in range(ROUNDS)]
    counts = np.bincount(np.concatenate(selected), minlength=N)
    # Accuracy rises with participation and data size; noise grows with skew.
    noise = 0.15 if alpha <= 0.1 else 0.05
    # Arbitrary per-method shift so fake methods do not sit on one point in the plots.
    shift = (sum(map(ord, selector + good)) % 7 - 3) * 0.006
    base = 0.45 + shift + 0.004 * (counts - counts.mean()) + 0.00003 * (sizes - sizes.mean())
    client_rounds = [np.clip(base * (1 - np.exp(-t / 20)) + rng.normal(0, noise, N), 0, 1)
                     for t in range(1, ROUNDS + 1)]
    return {
        "config": {"selector": selector, "dataset": "cifar10", "alpha": alpha,
                   "good_def": good, "seed": seed, "n_clients": N, "k": K, "rounds": ROUNDS},
        "selected": selected,
        "global_acc": [float(a.mean()) for a in client_rounds],
        "client_acc": client_rounds[-1].tolist(),
        "client_acc_rounds": [a.tolist() for a in client_rounds],
        "selection_counts": counts.tolist(), "data_sizes": sizes.tolist(),
        "queried_per_round": [0] * ROUNDS,
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "fake")
    os.makedirs(out, exist_ok=True)
    # Fake prop_fair / guarantee runs reuse the round-robin schedule; they only exercise
    # the plot styling and legend, not any real selector behaviour.
    runs = [("random", "count"), ("round_robin", "count"), ("power_of_choice", "count"),
            ("prop_fair", "count"), ("prop_fair", "weighted"),
            ("guarantee", "count"), ("guarantee", "weighted")]
    for sel, good in runs:
        for alpha in (0.1, 0.5, 100):
            for seed in (0, 1, 2):
                r = fake_run(sel, alpha, seed, good)
                path = os.path.join(out, f"{sel}_cifar10_{alpha}_{good}_{seed}.json")
                with open(path, "w") as f:
                    json.dump(r, f)
    print(f"wrote fake results to {out}")


if __name__ == "__main__":
    main()
