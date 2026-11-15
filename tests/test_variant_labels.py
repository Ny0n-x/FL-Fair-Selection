import json

import pandas as pd

from metrics.compare import latex_table
from metrics.plots import _order, display_name, split_variant, style
from metrics.truncate import report


def test_split_variant():
    assert split_variant("subtrunc_subtrunc_b6.0") == ("subtrunc", "subtrunc_b6.0")
    assert split_variant("power_of_choice") == ("power_of_choice", "")
    assert split_variant("power_of_choice_d50") == ("power_of_choice", "d50")
    assert split_variant("prop_fair") == ("prop_fair", "")
    assert split_variant("mystery_x") == ("mystery_x", "")


def test_variant_keeps_base_colour_and_gets_own_label_and_line():
    base, var = style("subtrunc", "count"), style("subtrunc_subtrunc_b6.0", "count")
    assert var["color"] == base["color"] != "#999999"
    assert var["ls"] != base["ls"] and var["label"] == "SUBTRUNC (subtrunc_b6.0)"
    assert style("deficit_eps0.5", "weighted")["label"] == "Deficit (eps0.5) (wtd)"
    assert _order([("subtrunc_subtrunc_b6.0", "count"), ("random", "count"), ("subtrunc", "count")]) \
        == [("random", "count"), ("subtrunc", "count"), ("subtrunc_subtrunc_b6.0", "count")]


def test_latex_escapes_variant_names():
    df = pd.DataFrame([dict(selector=s, good="count", dataset="cifar10", alpha=0.1, seed=i,
                            global_acc_final=0.5 + i / 100)
                       for s in ["random", "subtrunc_subtrunc_b6.0"] for i in range(2)])
    tex = latex_table(df, pd.DataFrame(), "cifar10", 0.1, metrics=["global_acc_final"])
    row = next(l for l in tex.splitlines() if l.startswith("SUBTRUNC"))
    assert row.startswith("SUBTRUNC (subtrunc\\_b6.0) &")
    assert display_name("random") == "Random"


def _run(tmp_path, name, rounds=20, **cfg):
    c = {"selector": "subtrunc", "dataset": "cifar10", "alpha": 0.1, "seed": 0, **cfg}
    (tmp_path / name).write_text(json.dumps({"cfg": c, "global_acc": [0.5] * rounds,
                                             "client_acc_rounds": [[0.5, 0.4]] * rounds}))
    return str(tmp_path / name)


def test_report_keeps_variants_and_datasets_apart_and_short_steps(tmp_path, capsys):
    paths = [_run(tmp_path, "a.json"), _run(tmp_path, "b.json", subtrunc_b=6.0),
             _run(tmp_path, "c.json", dataset="mnist")]
    report(paths, step=5)
    out = capsys.readouterr().out
    assert "cifar10 subtrunc " in out and "cifar10 subtrunc_subtrunc_b6.0" in out
    assert "mnist subtrunc" in out and "nan" not in out
