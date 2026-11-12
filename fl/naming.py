"""Result file naming, shared by main.py and sweep.py (keep this module torch-free).

Base name:  {selector}_{dataset}_{alpha}_{good_def}_{seed}
Variant tag appended when a knob that matters for this run differs from its default,
e.g. power_of_choice with d=50 -> ..._d50, guarantee with T=5 -> ..._T5.
Runs with all-default knobs keep the old name, so existing results stay valid.
"""

# knobs that can change any run's outcome
COMMON = {"n_clients": 50, "k": 10, "lr": 0.01, "local_epochs": 2, "batch_size": 32,
          "lr_schedule": "cosine", "partition": "dirichlet"}
# knobs that only matter for one selector (e.g. T is ignored by random)
PER_SELECTOR = {
    "power_of_choice": {"d": None},
    "guarantee": {"T": 10, "M": 1, "filler": "random"},
    "deficit": {"eps": 0.1},
    "subtrunc": {"subtrunc_b": 1.10, "subtrunc_lambda": 0.5, "subtrunc_r": 10,
                 "subtrunc_norm": False},
}
# NOT tagged: rounds, ckpt_*, config path. A run with fewer rounds is a prefix of a longer one.


def variant_tag(cfg):
    knobs = {**COMMON, **PER_SELECTOR.get(cfg["selector"], {})}
    parts = []
    for key in sorted(knobs):
        val = cfg.get(key, knobs[key])
        if val is None and knobs[key] is None:
            continue
        if val != knobs[key]:
            parts.append(f"{key}{val}")
    return "".join("_" + p for p in parts)


def result_name(cfg):
    return (f"{cfg['selector']}_{cfg['dataset']}_{float(cfg['alpha'])}_{cfg['good_def']}"
            f"_{cfg['seed']}{variant_tag(cfg)}")
