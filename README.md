# FL-Fair-Selection
FL client selection via fair division

## Running

    python main.py --rounds 5                  # random, default config (100 rounds)
    python main.py --selector power_of_choice
    python main.py --selector divfl --alpha 0.5 --seed 1
    python main.py --selector guarantee --alpha 0.1 --seed 0

Defaults are in `configs/default.yaml`; command-line flags override them.

## Evaluation (metrics/)
Reads results JSONs in the `results/schema.md` format.

    python -m metrics.loader results                  # summary table over seeds
    python -m metrics.gate1 results/random_*.json     # Gate 1 per run and pooled over seeds
    python -m metrics.compare results --latex tables/main_{alpha}.tex   # paired vs Random
    python -m metrics.plots results --out figures     # report figures

Gate 1 pools the per-seed Spearman rho with Fisher's z (variance 1.06 / (N - 3)) and
reports the pooled rho, its 95% CI and how many seeds are significant on their own.
`compare` pairs each method with Random on the same (dataset, alpha, seed) and reports the
mean difference, a bootstrap 95% CI and paired t / Wilcoxon p-values. With fewer than 6
seeds Wilcoxon cannot reach p < 0.05, so the LaTeX dagger uses the t-test by default.

## Sweeps

    python sweep.py --selectors random --alphas 0.1 0.5 1.0 --seeds 0 1 2 3 4
    python sweep.py --selectors random guarantee --good count weighted --jobs 2
    python sweep.py ... --dry-run          # list the runs
    python sweep.py ... -- --rounds 5      # pass extra flags to main.py

Finished runs (their results JSON exists) are skipped, so rerunning resumes. Selectors that
ignore good_def run once. Logs go to results/logs/.

Every client counts once. Variance uses ddof=0. Worst-10% is the mean of the lowest 10% of clients.
