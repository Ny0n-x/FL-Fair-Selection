import json

import pytest

from metrics.loader import load_dir, load_run, summarize

BASE = {"global_acc": [0.1, 0.2, 0.3], "selection_counts": [2, 1, 0],
        "data_sizes": [10, 20, 30], "selected": [[0, 1], [0, 2], [1, 0]][:3]}
CFG = {"selector": "random", "dataset": "cifar10", "alpha": 0.1, "good_def": "count", "seed": 0}


def _write(tmp_path, name, rec):
    p = tmp_path / name
    p.write_text(json.dumps(rec))
    return str(p)


def test_schema_md_format(tmp_path):
    r = load_run(_write(tmp_path, "a.json", {**BASE, "client_acc": [0.5, 0.6, 0.7],
                                             "config": CFG}))
    assert r["selector"] == "random" and r["good_def"] == "count" and r["alpha"] == 0.1


def test_main_py_aliases(tmp_path):
    rec = {k: v for k, v in BASE.items() if k != "selected"}
    rec.update({"selected_ids": BASE["selected"], "client_acc": [0.5, 0.6, 0.7],
                "cfg": {**CFG, "alpha": 100}})
    r = load_run(_write(tmp_path, "b.json", rec))
    assert r["selected"] == BASE["selected"] and r["alpha"] == 100.0


def test_top_level_identity_and_old_key(tmp_path):
    r = load_run(_write(tmp_path, "c.json", {**BASE, **CFG, "client_acc_final": [0.5, 0.6, 0.7]}))
    assert r["client_acc"] == [0.5, 0.6, 0.7]


def test_missing_field_is_reported(tmp_path):
    with pytest.raises(KeyError, match="client_acc"):
        load_run(_write(tmp_path, "d.json", {**BASE, "config": CFG}))


def test_summary_over_seeds(tmp_path):
    for seed in (0, 1):
        _write(tmp_path, f"s{seed}.json", {**BASE, "client_acc": [0.5, 0.6, 0.7],
                                           "config": {**CFG, "seed": seed}})
    s = summarize(load_dir(str(tmp_path)))
    assert len(s) == 1 and s["n_seeds"].iloc[0] == 2


def test_selector_variants_are_separate_rows(tmp_path):
    acc = {"client_acc": [0.5, 0.6, 0.7]}
    st = {**CFG, "selector": "subtrunc"}
    for seed in (0, 1):
        _write(tmp_path, f"d{seed}.json", {**BASE, **acc, "cfg": {**st, "seed": seed}})
        _write(tmp_path, f"b{seed}.json", {**BASE, **acc, "cfg": {**st, "seed": seed,
                                                                 "subtrunc_b": 6.0}})
        _write(tmp_path, f"k{seed}.json", {**BASE, **acc, "cfg": {**st, "seed": seed, "k": 20}})
    r = load_run(str(tmp_path / "b0.json"))
    assert r["selector"] == "subtrunc_subtrunc_b6.0" and r["base_selector"] == "subtrunc"
    df = load_dir(str(tmp_path))
    # k is a shared knob: not folded in, so the k=20 runs pool with the default ones
    assert df["selector"].value_counts().to_dict() == {"subtrunc": 4, "subtrunc_subtrunc_b6.0": 2}
    assert set(summarize(df)["selector"]) == {"subtrunc", "subtrunc_subtrunc_b6.0"}


def test_mixed_rounds_or_schedule_warns(tmp_path):
    acc = {"client_acc": [0.5, 0.6, 0.7]}
    _write(tmp_path, "a.json", {**BASE, **acc, "cfg": {**CFG, "lr_schedule": "cosine"}})
    _write(tmp_path, "b.json", {**BASE, **acc, "cfg": {**CFG, "seed": 1, "lr_schedule": "cosine"}})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # same setup: no warning
        load_dir(str(tmp_path))
    _write(tmp_path, "c.json", {**BASE, **acc, "cfg": {**CFG, "seed": 2}})   # constant (old)
    with pytest.warns(UserWarning, match="lr_scheduleconstant"):
        load_dir(str(tmp_path))
    _write(tmp_path, "c.json", {**BASE, **acc, "global_acc": [0.1] * 6,
                                "cfg": {**CFG, "seed": 2, "lr_schedule": "cosine"}})
    with pytest.warns(UserWarning, match="rounds=6"):
        load_dir(str(tmp_path))
