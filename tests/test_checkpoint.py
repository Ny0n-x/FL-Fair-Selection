"""A run cut off mid-way and restarted must give the same results JSON as an uninterrupted run."""
import json

import pytest
import torch

import main

torch.set_num_threads(1)   # tiny model: more threads only add overhead

N, K, ROUNDS = 6, 3, 6


def fake_partition(cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    def split(m):
        return torch.randn(m, 3, 32, 32, generator=g), torch.randint(0, 10, (m,), generator=g)
    train = [split(20 + 4 * i) for i in range(N)]
    test = [split(10) for _ in range(N)]
    return train, test


def cfg(selector):
    return {"dataset": "cifar10", "alpha": 0.1, "n_clients": N, "k": K, "rounds": ROUNDS,
            "local_epochs": 1, "batch_size": 8, "lr": 0.01, "seed": 0, "selector": selector,
            "good_def": "count", "d": None, "T": 3, "M": 1, "eps": 0.1, "filler": "random",
            "ckpt_every": 2}


class Crash(Exception):
    pass


@pytest.mark.parametrize("selector", ["random", "power_of_choice", "divfl", "guarantee"])
def test_resume_matches_uninterrupted(selector, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "load_partition", fake_partition)
    monkeypatch.setattr(main, "DEV", torch.device("cpu"))
    name = f"results/{selector}_cifar10_0.1_count_0"

    (tmp_path / "full").mkdir()
    monkeypatch.chdir(tmp_path / "full")
    main.run(cfg(selector))
    full = json.load(open(f"{name}.json"))

    (tmp_path / "resumed").mkdir()
    monkeypatch.chdir(tmp_path / "resumed")
    real_print, rounds_seen = print, []
    def crash_after_round_4(*a, **kw):   # main prints once per finished round
        if a and str(a[0]).startswith("round"):
            rounds_seen.append(a[0])
            if len(rounds_seen) == 5:
                raise Crash
        real_print(*a, **kw)
    monkeypatch.setattr("builtins.print", crash_after_round_4)
    with pytest.raises(Crash):
        main.run(cfg(selector))
    monkeypatch.setattr("builtins.print", real_print)
    ckpt = torch.load(f"results/ckpt/{name[len('results/'):]}.pt", weights_only=False)
    assert ckpt["round"] == 4                       # last multiple of ckpt_every before the crash

    main.run(cfg(selector))
    resumed = json.load(open(f"{name}.json"))
    assert resumed == full
    assert not (tmp_path / "resumed/results/ckpt" / f"{name[len('results/'):]}.pt").exists()


@pytest.mark.parametrize("selector", ["random", "power_of_choice", "divfl", "guarantee"])
def test_truncated_long_run_equals_short_run(selector, tmp_path, monkeypatch):
    """metrics.truncate relies on this: the first R rounds of a long run are an R-round run."""
    from metrics.truncate import truncate
    monkeypatch.setattr(main, "load_partition", fake_partition)
    monkeypatch.setattr(main, "DEV", torch.device("cpu"))
    monkeypatch.chdir(tmp_path)
    path = f"results/{selector}_cifar10_0.1_count_0.json"
    main.run(cfg(selector))
    long = json.load(open(path))
    short_cfg = {**cfg(selector), "rounds": 4}
    main.run(short_cfg)
    assert truncate(long, 4) == json.load(open(path))
