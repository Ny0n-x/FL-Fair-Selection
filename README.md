# FL-Fair-Selection
FL client selection via fair division

## Evaluation (metrics/)
Reads results JSONs in the `results/schema.md` format (`config` holds the run identity).

    python -m metrics.loader results                  # mean +- std over seeds, one row per method
    python -m metrics.gate1 results/random_*.json     # Gate 1: Spearman(count, acc), Spearman(size, acc), worst10/mean
    python -m metrics.plots results --out figures     # histograms, trade-off, selection counts, size breakdown
    python -m metrics.make_fake_results results/fake  # made-up runs for testing the pipeline only

Metric conventions: every client counts once; variance/std are population (ddof=0);
worst-10% = mean of the lowest ceil(0.1 N) clients; Jain's index is over individual clients.
