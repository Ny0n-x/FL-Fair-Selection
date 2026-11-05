# Results JSON schema

One file per run: `results/{selector}_{dataset}_{alpha}_{good}_{seed}.json`

| Field | Type | Meaning |
|---|---|---|
| selected | list[list[int]] | K client ids chosen in each round |
| global_acc | list[float] | global test accuracy after each round |
| client_acc | list[float] | final per-client test accuracy (length N) |
| selection_counts | list[int] | times each client was selected (length N) |
| data_sizes | list[int] | local data size of each client |
| queried_per_round | list[int] | number of clients whose loss was queried each round |
| config | dict | the full config used for the run |

Optional (used by metrics/ when present):

| Field | Type | Meaning |
|---|---|---|
| client_acc_rounds | list[list[float]] | per-client test accuracy after each round, used for last-10-round metrics |

The loader also accepts the names main.py used before: `cfg` for `config`, `selected_ids` for
`selected`, and a run total `extra_queries` in place of `queried_per_round`.
