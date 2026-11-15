# FL-Fair-Selection

Client selection for federated learning through fair-division principles.
We ask whether giving every client a fair share of participation (selection-stage fairness) changes the
global accuracy or the spread of accuracy across clients, compared with Random, Round-Robin,
Power-of-Choice, DivFL and SUBTRUNC.

**Our selectors** (`selection/`): `guarantee` (every client is picked at least once in any window of
`T` rounds; feasible iff `N <= K*T`), `deficit` (sampling proportional to the participation deficit,
`--good_def count|weighted`), and `round_robin` (exact equal counts, used as a reference).
**Baselines** (`baselines/`): `random`, `power_of_choice`, `divfl`, `subtrunc`.
`divfl` and `subtrunc` are simplified re-implementations (stale update vectors, see the docstrings),
not the authors' code.

## Setup used for the reported results

Plain PyTorch. FedAvg weighted by client data size, N = 50 clients, K = 10 per round,
300 rounds, 2 local epochs, batch 32, SGD with momentum 0.9, lr 0.01 with cosine decay to 0, a small CNN.
Datasets: CIFAR-10 and MNIST with Dirichlet(alpha) label skew, alpha in {0.1, 0.5, 1.0, 100}, at least 50
samples per client, 80/20 train/test split per client. Count fairness unless stated (`good_def`).
Defaults of the selectors: `T = 10`, `M = 1` (only `M = 1` is implemented), `eps = 0.1`.
SUBTRUNC is run with `b = 6`, `lambda = 0.95`, no normalisation, `r = 10`.

| Group | Datasets / alphas | Seeds |
|---|---|---|
| Random, Power-of-Choice (d = 2K), Round-Robin, Deficit (count), SUBTRUNC | CIFAR-10, alpha 0.1 and 0.5 | 0-4 |
| all methods (Deficit also weighted) | CIFAR-10, alpha 0.1 / 0.5 / 1.0 / 100 | 0-2 |
| all methods except SUBTRUNC | MNIST, alpha 0.1 / 0.5 / 1.0 | 0-2 |
| Deficit with eps 0.01 and 1.0 (ablation) | CIFAR-10, alpha 0.1 | 0-2 |



## Install

    python -m venv .venv
    .venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt   # for an NVIDIA GPU install the CUDA build of torch first
    python -m pytest -q

CIFAR-10 and MNIST download to `data/raw/` on first use.

## Run one config

    python main.py --selector guarantee --alpha 0.1 --seed 0 --rounds 300
    python main.py --selector subtrunc --subtrunc_b 6 --subtrunc_lambda 0.95 --rounds 300
    python main.py --dataset mnist --selector divfl --rounds 300

Defaults are in `configs/default.yaml` (its `rounds` is 300); command-line flags override them.
One config writes one JSON to `results/` (format: `results/schema.md`). A killed run resumes from
`results/ckpt/` when the same command is repeated.

## Reproduce the experiments

    # CIFAR-10 main grid (seeds 0-2); sweep.py skips runs whose JSON exists
    python sweep.py --selectors random power_of_choice divfl guarantee round_robin deficit --alphas 0.1 0.5 1.0 100 --seeds 0 1 2 --good count weighted --jobs 2 -- --rounds 300
    python sweep.py --selectors subtrunc --alphas 0.1 0.5 1.0 100 --seeds 0 1 2 --jobs 2 -- --rounds 300 --subtrunc_b 6 --subtrunc_lambda 0.95

    # extra seeds on the headline cells
    python sweep.py --selectors random power_of_choice round_robin deficit --alphas 0.1 0.5 --seeds 3 4 --good count --jobs 2 -- --rounds 300
    python sweep.py --selectors subtrunc --alphas 0.1 0.5 --seeds 3 4 --jobs 2 -- --rounds 300 --subtrunc_b 6 --subtrunc_lambda 0.95

    # MNIST
    python sweep.py --selectors random power_of_choice divfl guarantee round_robin deficit --alphas 0.1 0.5 1.0 --seeds 0 1 2 --good count weighted --jobs 2 -- --rounds 300 --dataset mnist

    # Deficit eps ablation
    python sweep.py --selectors deficit --alphas 0.1 --seeds 0 1 2 --good count --jobs 2 -- --rounds 300 --eps 0.01
    python sweep.py --selectors deficit --alphas 0.1 --seeds 0 1 2 --good count --jobs 2 -- --rounds 300 --eps 1.0

`--dry-run` lists the runs. Selectors that ignore `good_def` run once. Logs go to `results/logs/`.
Runs with non-default knobs get a suffix in the file name (`fl/naming.py`), so variants never overwrite
each other. Result names do not contain the number of rounds, so use a separate results folder for
a different round budget. A 300-round run takes about 10-15 minutes on a laptop RTX 3060.
`notebooks/colab_sweep.ipynb` (CIFAR-10) and `notebooks/colab_mnist.ipynb` (MNIST) run the same
commands on Colab.

## Tables, figures and statistics

Run the plots per dataset, otherwise CIFAR-10 and MNIST curves are mixed. By default the plots use only the
seeds that all methods share (0-2); `--all-seeds` uses every seed, `--variants` adds the ablation runs.

    python -m metrics.loader results --out results/summary.csv
    python -m metrics.compare results --out results/compare.csv --latex "tables/{dataset}_{alpha}.tex"
    python -m metrics.gate1 (Get-ChildItem results\random_cifar10_*.json).FullName --out results/gate1_pooled.csv   # PowerShell
    python -m metrics.gate1 results/random_cifar10_*.json --out results/gate1_pooled.csv                            # bash
    python -m metrics.plots results --dataset cifar10 --out figures/cifar10 --alpha 0.1
    python -m metrics.plots results --dataset mnist --out figures/mnist --alpha 0.1
    python -m metrics.plots results --dataset cifar10 --out figures/cifar10_variants --alpha 0.1 --variants

Metrics: global accuracy, per-client accuracy variance, minimum and worst-10% (mean of the lowest
ceil(0.1 N) clients), Jain index, selection-count variance, extra client queries per round, rounds to a target
accuracy (90% of Random's last-10-round accuracy). Every client counts once; variance uses ddof = 0.

`metrics.compare` pairs each method with Random on (dataset, alpha, seed) and reports the mean paired difference,
a paired t-test and a Wilcoxon p-value. The p-values are nominal (no multiple-comparison correction). The printed
bootstrap intervals are over only 3-5 paired differences and are too narrow to quote. A dagger in the LaTeX tables
marks a significant paired difference to Random in either direction. `metrics.gate1` tests whether the number of
times a client is selected under Random predicts its accuracy (Spearman rho, pooled over seeds with Fisher's z).

## Layout

| Path | Content |
|---|---|
| `main.py` | one run (training loop, checkpoints) |
| `sweep.py` | grids of runs |
| `configs/default.yaml` | default hyper-parameters |
| `selection/` | our selectors and the participation tracker |
| `baselines/` | Random, Power-of-Choice, DivFL, SUBTRUNC |
| `data/partition.py` | Dirichlet partitioner (`data/plots.py`: partition diagnostics) |
| `fl/naming.py` | result file names |
| `metrics/` | loader, paired comparison, Gate 1, plots |
| `results/` | one JSON per run, `summary.csv`, `compare.csv`, `gate1_pooled.csv`, `CODE_VERSION.txt` |
| `tables/`, `figures/` | LaTeX tables and figures used in the report |
| `tests/` | pytest suite |
| `notebooks/` | Colab drivers |

## Notes on reproducibility

GPU runs are not bit-identical even with a fixed seed (CUDA kernels are nondeterministic), so a rerun gives close
but not identical curves. The reported runs were produced on a single laptop GPU (RTX 3060) with `sweep.py`.
SUBTRUNC uses `b = 6` because the paper's `b = 1.10` was inert at our loss scale in a 20-round, single-seed
diagnostic (its selections matched lambda = 0, i.e. DivFL, up to run-to-run noise); that diagnostic is not part
of the result set. SUBTRUNC refreshes the loss of all 50 clients every round (50 extra queries per round).