"""Result file naming, shared by main.py and sweep.py (keep this module torch-free).

Base name:  {selector}_{dataset}_{alpha}_{good_def}_{seed}
Variant tag appended when a knob that matters for this run differs from its default,
e.g. power_of_choice with d=50 -> ..._d50, guarantee with T=5 -> ..._T5.
Runs with all-default knobs keep the old name, so existing results stay valid.
"""

# knobs that can change any run's outcome
COMMON = {"n_clients": 50, "k": 10, "lr": 0.01, "local_epochs": 2, "batch_size": 32,
          "lr_schedule": "cosine", "lr_min": 0.0, "partition": "dirichlet"}
# knobs that only matter for one selector (e.g. T is ignored by random)
PER_SELECTOR = {
    "power_of_choice": {"d": None},
    "guarantee": {"T": 10, "M": 1, "filler": "random"},
    "deficit": {"eps": 0.1},
    "subtrunc": {"subtrunc_b": 1.10, "subtrunc_lambda": 0.5, "subtrunc_r": 10,
                 "subtrunc_norm": False},
}
# NOT tagged: rounds, ckpt_*, config path. With lr_schedule constant a shorter run is a prefix of a
# longer one; with cosine it is not (the schedule depends on rounds), so keep different budgets in
# different result folders. sweep.py checks an existing JSON's rounds and lr_schedule before
# skipping its job, and stops if they differ (sweep.budget_mismatch).


def variant_tag(cfg, common=True):
    """'_key{val}' for every knob that differs from its default ('' if none).

    common=False tags only the selector's own knobs (PER_SELECTOR), e.g. to label SUBTRUNC b=6
    apart from b=1.10 while runs that differ in a shared knob (k, lr, ...) stay comparable.
    """
    knobs = {**(COMMON if common else {}), **PER_SELECTOR.get(cfg["selector"], {})}
    parts = []
    for key in sorted(knobs):
        val = cfg.get(key, knobs[key])
        if val is None and knobs[key] is None:
            continue
        # a float knob given as an int (sweep.py reads '--subtrunc_b 6' as YAML -> 6, main.py's
        # argparse -> 6.0) must give the same name, or sweep.py never finds main.py's results
        if isinstance(knobs[key], float) and isinstance(val, int) and not isinstance(val, bool):
            val = float(val)
        if val != knobs[key]:
            parts.append(f"{key}{val}")
    return "".join("_" + p for p in parts)


def result_name(cfg):
    return (f"{cfg['selector']}_{cfg['dataset']}_{float(cfg['alpha'])}_{cfg['good_def']}"
            f"_{cfg['seed']}{variant_tag(cfg)}")
