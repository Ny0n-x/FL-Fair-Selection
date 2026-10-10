# FL-Fair-Selection
FL client selection via fair division

## Evaluation (metrics/)
Reads results JSONs in the `results/schema.md` format.

    python -m metrics.loader results                  # summary table over seeds
    python -m metrics.gate1 results/random_*.json     # Gate 1 numbers
    python -m metrics.plots results --out figures     # report figures

Every client counts once. Variance uses ddof=0. Worst-10% is the mean of the lowest 10% of clients.
