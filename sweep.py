"""Run a grid of main.py configs: selectors x alphas x good_defs x seeds.

Runs whose results JSON already exists are skipped, and a run that was cut off resumes from its
last checkpoint in results/ckpt/ (see main.py), so an interrupted sweep can be restarted with the
same command. Each run's stdout/stderr goes to results/logs/.

Usage:
    python sweep.py --selectors random --alphas 0.1 0.5 1.0 --seeds 0 1 2 3 4
    python sweep.py --selectors random guarantee --good count weighted --jobs 2
    python sweep.py ... --dry-run                     # print the commands only
    python sweep.py ... -- --rounds 5                 # anything after -- goes to main.py
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

# selectors whose behaviour does not depend on good_def; run once with good=count
GOOD_AGNOSTIC = {"random", "power_of_choice", "divfl", "round_robin", "qffl", "propfair"}


def result_path(results_dir, selector, dataset, alpha, good, seed):
    """Same name main.py writes: alpha is a Python float, so 1 -> '1.0'."""
    return os.path.join(results_dir, f"{selector}_{dataset}_{float(alpha)}_{good}_{seed}.json")


def grid(selectors, alphas, goods, seeds):
    for sel, a, good, seed in itertools.product(selectors, alphas, goods, seeds):
        if sel in GOOD_AGNOSTIC and good != goods[0]:
            continue
        yield sel, a, (goods[0] if sel in GOOD_AGNOSTIC else good), seed


def run_one(job, config, extra, log_dir, env):
    sel, a, good, seed = job
    cmd = [sys.executable, "main.py", "--config", config, "--selector", sel, "--alpha", str(a),
           "--good_def", good, "--seed", str(seed), *extra]
    log = os.path.join(log_dir, f"{sel}_{a}_{good}_{seed}.log")
    t0 = time.time()
    with open(log, "a") as f:   # append, so the log of a resumed run keeps the earlier part
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return job, rc, time.time() - t0, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--selectors", nargs="+", default=["random"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1])
    ap.add_argument("--good", nargs="+", default=["count"], choices=("count", "weighted"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--results", default="results", help="where main.py writes (for skipping)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun even if the JSON exists")
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]
    dataset = yaml.safe_load(open(a.config))["dataset"]

    jobs, skipped = [], 0
    for job in grid(a.selectors, a.alphas, a.good, a.seeds):
        if not a.force and os.path.exists(result_path(a.results, job[0], dataset, *job[1:])):
            skipped += 1
        else:
            jobs.append(job)
    print(f"{len(jobs)} to run, {skipped} already done")
    if a.dry_run:
        for sel, al, good, seed in jobs:
            print(f"  python main.py --config {a.config} --selector {sel} --alpha {al} "
                  f"--good_def {good} --seed {seed} {' '.join(extra)}".rstrip())
        return

    log_dir = os.path.join(a.results, "logs")
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ)
    if a.jobs > 1:  # parallel torch processes oversubscribe the cores unless split
        env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // a.jobs)))
    failed = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, j, a.config, extra, log_dir, env) for j in jobs]
        for i, fut in enumerate(futs, 1):
            job, rc, dt, log = fut.result()
            status = "ok" if rc == 0 else f"FAILED (exit {rc}, see {log})"
            print(f"[{i}/{len(jobs)}] {job[0]} alpha={job[1]} good={job[2]} seed={job[3]}"
                  f"  {dt:.0f}s  {status}", flush=True)
            if rc != 0:
                failed.append(job)
    if failed:
        sys.exit(f"{len(failed)} run(s) failed")


if __name__ == "__main__":
    main()
