import os

from fl.naming import result_name
from sweep import grid, parse_extra, result_path

BASE = {"selector": "random", "dataset": "cifar10", "alpha": 0.1, "good_def": "count", "seed": 0,
        "n_clients": 50, "k": 10, "lr": 0.01, "local_epochs": 2, "batch_size": 32,
        "lr_schedule": "cosine", "d": None, "T": 10, "M": 1, "eps": 0.1, "filler": "random",
        "rounds": 300}


def cfg(**kw):
    return {**BASE, **kw}


def test_default_names_are_unchanged():
    assert result_name(cfg()) == "random_cifar10_0.1_count_0"
    assert result_name(cfg(selector="guarantee", alpha=1)) == "guarantee_cifar10_1.0_count_0"
    assert result_name(cfg(selector="power_of_choice")) == "power_of_choice_cifar10_0.1_count_0"


def test_variants_get_distinct_names():
    poc = "power_of_choice"
    assert result_name(cfg(selector=poc, d=50)) != result_name(cfg(selector=poc))
    assert result_name(cfg(selector=poc, d=50)) != result_name(cfg(selector=poc, d=30))
    g = "guarantee"
    assert len({result_name(cfg(selector=g)), result_name(cfg(selector=g, T=5)),
                result_name(cfg(selector=g, M=2)), result_name(cfg(selector=g, filler="loss"))}) == 4
    e = "deficit"
    assert result_name(cfg(selector=e, eps=0.05)) != result_name(cfg(selector=e))
    assert result_name(cfg(k=20)) != result_name(cfg())
    assert result_name(cfg(lr=0.05)) != result_name(cfg())


def test_knobs_a_selector_ignores_do_not_change_its_name():
    assert result_name(cfg(T=3, d=50, eps=0.5, filler="loss")) == result_name(cfg())  # random
    assert result_name(cfg(selector="deficit", T=3, d=50)) == result_name(cfg(selector="deficit"))


def test_rounds_not_in_name():
    assert result_name(cfg(rounds=5)) == result_name(cfg(rounds=300))


def test_missing_keys_count_as_default():
    c = {k: v for k, v in BASE.items() if k not in ("batch_size", "lr_schedule", "T")}
    assert result_name({**c, "selector": "guarantee"}) == "guarantee_cifar10_0.1_count_0"


def test_result_path_with_overrides_matches_main_name():
    p = result_path("results", "power_of_choice", "cifar10", 0.1, "count", 0, cfg=cfg(d=50))
    assert p == os.path.join("results", result_name(cfg(selector="power_of_choice", d=50)) + ".json")
    assert result_path("results", "random", "cifar10", 1, "count", 0) == \
        os.path.join("results", "random_cifar10_1.0_count_0.json")


def test_parse_extra():
    assert parse_extra(["--d", "50", "--rounds=300", "--filler", "loss", "--lr", "0.05"]) == \
        {"d": 50, "rounds": 300, "filler": "loss", "lr": 0.05}


def test_guarantee_runs_once_per_seed_regardless_of_good_def():
    jobs = list(grid(["guarantee", "deficit"], [0.1], ["count", "weighted"], [0, 1]))
    assert sum(j[0] == "guarantee" for j in jobs) == 2
    assert sum(j[0] == "deficit" for j in jobs) == 4
