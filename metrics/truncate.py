"""Cut finished runs down to fewer rounds, and show where the curves stop improving.

With a constant learning rate, a run of R rounds is the first R rounds of a longer run with
the same config: nothing else depends on the total number of rounds. So the round budget can be
picked after the fact from the long runs, without rerunning anything. With lr_schedule: cosine
the rate in every round depends on the total, so such runs cannot be cut (--report still works).

Usage:
    python -m metrics.truncate results --report                 # accuracy at 100, 200, ... rounds
    python -m metrics.truncate results --rounds 300 --out results_r300
    python -m metrics.loader results_r300 --out results_r300/summary.csv   # tables/plots as usual
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from metrics.metrics import outcome_metrics

LAST = 10   # same window as metrics.loader's *_last10 columns


def truncate(run, rounds):
    """The results dict main.py would have written with cfg['rounds'] = rounds."""
    sched = run["cfg"].get("lr_schedule") or "constant"
    if sched != "constant":
        raise ValueError(f"cannot cut a run with lr_schedule={sched!r}: its learning rate in every "
                         "round depends on the total number of rounds, so rerun with --rounds")
    n = len(run["global_acc"])
    if rounds > n:
        raise ValueError(f"run has only {n} rounds, cannot cut to {rounds}")
    out = dict(run)
    out["cfg"] = {**run["cfg"], "rounds": rounds}
    sel = run["selected_ids"][:rounds]
    out["selected_ids"] = sel
    out["global_acc"] = run["global_acc"][:rounds]
    out["client_acc_rounds"] = run["client_acc_rounds"][:rounds]
    out["client_acc"] = out["client_acc_rounds"][-1]
    counts = [0] * len(run["selection_counts"])
    for ids in sel:
        for i in ids:
            counts[i] += 1
    out["selection_counts"] = counts
    out["queried_per_round"] = run["queried_per_round"][:rounds]
    out["extra_queries"] = sum(out["queried_per_round"])
    if "train_loss" in run:
        out["train_loss"] = [e for e in run["train_loss"] if e[0] < rounds]
    return out


def report(paths, step=100):
    """Mean over seeds of global acc and worst-10% client acc (both last-10-round means) at
    every `step` rounds, per selector and alpha."""
    groups = defaultdict(list)
    for p in paths:
        r = json.load(open(p))
        groups[(r["cfg"]["selector"], float(r["cfg"]["alpha"]))].append(r)
    n_max = min(len(r["global_acc"]) for rs in groups.values() for r in rs)
    marks = list(range(step, n_max + 1, step))
    print("global acc / worst-10% client acc, mean of the last 10 rounds before each mark, "
          "averaged over seeds")
    print(f"{'selector':<16}{'alpha':>6} {'seeds':>5}  " + "".join(f"{f'@{m}':>14}" for m in marks))
    for (sel, alpha), rs in sorted(groups.items()):
        cells = []
        for m in marks:
            g = np.mean([np.mean(r["global_acc"][m - LAST:m]) for r in rs])
            w = np.mean([np.mean([outcome_metrics(a)["worst10_mean"]
                                  for a in r["client_acc_rounds"][m - LAST:m]]) for r in rs])
            cells.append(f"{g:.3f}/{w:.3f}")
        print(f"{sel:<16}{alpha:>6} {len(rs):>5}  " + "".join(f"{c:>14}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--report", action="store_true", help="print accuracy every --step rounds")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--rounds", type=int, help="cut every run to this many rounds")
    ap.add_argument("--out", help="where to write the cut runs (with --rounds)")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.results_dir, "*.json")))
    if not paths:
        raise SystemExit(f"no results JSONs in {a.results_dir}")
    if a.report:
        report(paths, a.step)
    if a.rounds:
        if not a.out or os.path.abspath(a.out) == os.path.abspath(a.results_dir):
            raise SystemExit("--rounds needs an --out folder different from the input")
        os.makedirs(a.out, exist_ok=True)
        for p in paths:
            with open(os.path.join(a.out, os.path.basename(p)), "w") as f:
                json.dump(truncate(json.load(open(p)), a.rounds), f)
        print(f"wrote {len(paths)} runs cut to {a.rounds} rounds to {a.out}")


if __name__ == "__main__":
    main()
