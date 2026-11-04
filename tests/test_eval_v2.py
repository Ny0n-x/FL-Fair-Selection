import json

import numpy as np
import pandas as pd
import pytest

from metrics.compare import bootstrap_ci, latex_table, paired
from metrics.gate1 import fisher_combine, gate1_table, pool
from metrics.loader import load_dir, load_run, run_metrics
from sweep import grid, result_path

CFG = {"selector": "random", "dataset": "cifar10", "alpha": 0.1, "good_def": "count", "seed": 0}


def _rec(acc, counts, sizes, **extra):
    return {"selected": [[0, 1]], "global_acc": [0.1, 0.2], "client_acc": acc,
            "selection_counts": counts, "data_sizes": sizes, **extra}


def _write(path, rec):
    path.write_text(json.dumps(rec))
    return str(path)


# ---- loader: query counts ----

def test_queries_from_queried_per_round(tmp_path):
    r = load_run(_write(tmp_path / "a.json", _rec([0.5, 0.6], [1, 1], [5, 9], config=CFG,
                                                  queried_per_round=[4, 6])))
    assert run_metrics(r)["queries_per_round"] == pytest.approx(5.0)


def test_queries_from_main_py_total(tmp_path):
    r = load_run(_write(tmp_path / "a.json", _rec([0.5, 0.6], [1, 1], [5, 9], cfg=CFG,
                                                  extra_queries=40)))
    assert run_metrics(r)["queries_per_round"] == pytest.approx(20.0)  # 2 rounds


def test_queries_missing_is_nan(tmp_path):
    r = load_run(_write(tmp_path / "a.json", _rec([0.5, 0.6], [1, 1], [5, 9], config=CFG)))
    assert np.isnan(run_metrics(r)["queries_per_round"])


# ---- gate1: pooling over seeds ----

def test_fisher_single_run_matches_spearman_p_roughly():
    # one run, rho=0.3, n=50: FHP z-test p is close to the usual ~0.034
    rho, lo, hi, p = fisher_combine([0.3], 50)
    assert rho == pytest.approx(0.3) and lo < 0.3 < hi and 0.02 < p < 0.06


def test_fisher_more_seeds_narrows_ci():
    _, lo1, hi1, p1 = fisher_combine([0.2], 50)
    _, lo5, hi5, p5 = fisher_combine([0.2] * 5, 50)
    assert hi5 - lo5 < hi1 - lo1 and p5 < p1


def test_fisher_drops_nan():
    assert fisher_combine([0.2, np.nan], 50)[0] == pytest.approx(0.2)


def test_pool_groups_by_alpha(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for a in (0.1, 0.5):
        for s in range(3):
            sizes = rng.integers(10, 500, 20).tolist()
            acc = (np.argsort(np.argsort(sizes)) / 20).tolist()  # acc rank == size rank
            paths.append(_write(tmp_path / f"r{a}_{s}.json", _rec(
                acc, rng.integers(0, 10, 20).tolist(), sizes,
                config={**CFG, "alpha": a, "seed": s})))
    pooled = pool(gate1_table(paths))
    assert list(pooled["alpha"]) == [0.1, 0.5] and (pooled["n_seeds"] == 3).all()
    assert (pooled["rho_size"] > 0.99).all() and (pooled["sig_seeds_size"] == 3).all()


# ---- compare: paired vs Random ----

def _runs(tmp_path, gap):
    """Random and guarantee over 4 seeds; guarantee's worst client is `gap` higher."""
    for s in range(4):
        base = [0.2 + 0.05 * s, 0.5, 0.6, 0.7]
        for sel, shift in (("random", 0.0), ("guarantee", gap)):
            acc = [base[0] + shift] + base[1:]
            _write(tmp_path / f"{sel}_{s}.json", _rec(
                acc, [2, 2, 2, 2], [10, 20, 30, 40], config={**CFG, "selector": sel, "seed": s}))
    return load_dir(str(tmp_path))


def test_paired_difference_is_exact(tmp_path):
    comp = paired(_runs(tmp_path, 0.1))
    row = comp[comp["metric"] == "min_final"].iloc[0]
    assert row["n_pairs"] == 4 and row["diff_mean"] == pytest.approx(0.1)
    assert row["ci_lo"] == pytest.approx(0.1) and row["ci_hi"] == pytest.approx(0.1)


def test_paired_no_difference_gives_nan_p(tmp_path):
    comp = paired(_runs(tmp_path, 0.0))
    assert comp["p_ttest"].isna().all()


def test_paired_uses_count_random_when_both(tmp_path):
    df = _runs(tmp_path, 0.1)
    dup = df[df["selector"] == "random"].copy()
    dup["good"], dup["min_final"] = "weighted", 99.0
    comp = paired(pd.concat([df, dup]))
    row = comp[(comp["selector"] == "guarantee") & (comp["metric"] == "min_final")].iloc[0]
    assert row["diff_mean"] == pytest.approx(0.1)


def test_bootstrap_needs_two():
    assert all(np.isnan(bootstrap_ci([0.3])))


def test_latex_table_bold_and_dagger(tmp_path):
    rng = np.random.default_rng(1)
    for s in range(5):
        for sel, shift in (("random", 0.0), ("guarantee", 0.1 + 0.01 * rng.random())):
            _write(tmp_path / f"{sel}_{s}.json", _rec(
                [0.2 + 0.03 * s + shift, 0.5, 0.6], [2, 2, 2], [10, 20, 30],
                config={**CFG, "selector": sel, "seed": s}))
    df = load_dir(str(tmp_path))
    tex = latex_table(df, paired(df), "cifar10", 0.1, metrics=["min_final"])
    guar = next(line for line in tex.splitlines() if line.startswith("Guarantee"))
    assert "\\textbf" in guar and "$^\\dagger$" in guar
    assert "\\toprule" in tex and tex.splitlines()[-1] == "\\end{tabular}"


# ---- sweep ----

def test_result_path_matches_main_py_float_format():
    assert result_path("results", "random", "cifar10", 1, "count", 0) == \
        "results/random_cifar10_1.0_count_0.json"


def test_grid_runs_good_agnostic_selectors_once():
    jobs = list(grid(["random", "guarantee"], [0.1], ["count", "weighted"], [0, 1]))
    assert sum(j[0] == "random" for j in jobs) == 2
    assert sum(j[0] == "guarantee" for j in jobs) == 4
