# Results JSON schema

One file per run: `results/{selector}_{dataset}_{alpha}_{good_def}_{seed}{variant}.json`, where `variant`
is empty for default knobs and otherwise lists the non-default knobs (see `fl/naming.py`),
e.g. `deficit_cifar10_0.1_count_0_eps0.01.json`.

Fields written by `main.py`:

| Field | Type | Meaning |
|---|---|---|
| cfg | dict | the full config of the run (selector, dataset, alpha, seed, rounds, lr_schedule, knobs) |
| selected_ids | list[list[int]] | the K client ids chosen in each round |
| global_acc | list[float] | global accuracy after each round, on the union of the client test splits |
| client_acc | list[float] | final per-client test accuracy (length N) |
| client_acc_rounds | list[list[float]] | per-client test accuracy after each round |
| selection_counts | list[int] | times each client was selected (length N) |
| data_sizes | list[int] | number of training samples of each client |
| queried_per_round | list[int] | extra client queries (loss or update evaluations) in each round |
| extra_queries | int | sum of `queried_per_round` |
| train_loss | list[[round, loss]] | training loss of the global model, logged at rounds 0, 4, 9, 14, ... |

`metrics/loader.py` also accepts the names `config` for `cfg` and `selected` for `selected_ids`;
`client_acc_rounds` and `queried_per_round` are optional there.