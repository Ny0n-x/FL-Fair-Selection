"""Paired comparison of every selector against Random.

Runs are paired by (dataset, alpha, seed): the same seed gives the same partition, so the
per-seed difference removes partition-to-partition variation. For each metric we report
the mean difference (method - Random), a percentile bootstrap 95% CI over seeds, and
paired t-test and Wilcoxon signed-rank p-values. With fewer than 6 seeds Wilcoxon cannot
go below p = 0.05, so the table marks significance with the t-test by default.

Usage: python -m metrics.compare [results_dir] [--out compare.csv] [--latex table.tex]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

from metrics.loader import REFERENCE, load_dir
from metrics.plots import NAMES

# metric -> (label, higher is better)
METRICS = {
    "global_acc_final": ("Global acc.", True),
    "worst10_mean_final": ("Worst-10\\%", True),
    "min_final": ("Min", True),
    "jain_final": ("Jain", True),
    "variance_final": ("Var.", False),
    "sel_freq_var": ("Sel. var.", False),
    "rounds_to_target": ("Rounds", False),
}
KEY = ["dataset", "alpha", "seed"]


def bootstrap_ci(d, n_boot=10000, level=0.95, seed=0):
    d = np.asarray(d, dtype=float)
    if d.size < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, d.size, (n_boot, d.size))
    m = d[idx].mean(1)
    return (float(np.quantile(m, (1 - level) / 2)), float(np.quantile(m, (1 + level) / 2)))


def _p(test, d):
    d = np.asarray(d, dtype=float)
    if d.size < 2 or np.ptp(d) < 1e-12:   # zero spread: the tests are undefined
        return float("nan")
    return float(test(d).pvalue)


def reference_rows(df):
    """One Random row per (dataset, alpha, seed). Random ignores good_def, so if it was run
    with both, the count run is used."""
    ref = df[df["selector"] == REFERENCE].sort_values("good", key=lambda g: g != "count")
    return ref.drop_duplicates(KEY).set_index(KEY)


def paired(df, metrics=None):
    """One row per (selector, good, dataset, alpha, metric) with paired stats vs Random."""
    metrics = [m for m in (metrics or METRICS) if m in df.columns]
    ref = reference_rows(df)
    rows = []
    others = df[df["selector"] != REFERENCE]
    for (sel, good, ds, a), g in others.groupby(["selector", "good", "dataset", "alpha"]):
        g = g.set_index(KEY)
        common = g.index.intersection(ref.index)
        for m in metrics:
            x, y = g.loc[common, m].astype(float), ref.loc[common, m].astype(float)
            ok = ~(x.isna() | y.isna())
            d = (x[ok] - y[ok]).to_numpy()
            lo, hi = bootstrap_ci(d)
            rows.append({"selector": sel, "good": good, "dataset": ds, "alpha": a,
                         "metric": m, "n_pairs": int(d.size),
                         "method_mean": float(x[ok].mean()), "random_mean": float(y[ok].mean()),
                         "diff_mean": float(d.mean()) if d.size else float("nan"),
                         "ci_lo": lo, "ci_hi": hi,
                         "p_ttest": _p(lambda v: ttest_1samp(v, 0.0), d),
                         "p_wilcoxon": _p(wilcoxon, d)})
    return pd.DataFrame(rows)


def latex_table(df, comp, dataset, alpha, metrics=None, test="p_ttest", sig=0.05):
    """booktabs table for one (dataset, alpha): mean +- std over seeds, best in bold,
    a dagger where the paired difference to Random has p < sig."""
    metrics = [m for m in (metrics or METRICS) if m in df.columns]
    sub = df[(df["dataset"] == dataset) & (df["alpha"] == alpha)]
    if sub.empty:
        raise ValueError(f"no runs for {dataset} alpha={alpha}")
    grp = sub.groupby(["selector", "good"])
    mean, std = grp[metrics].mean(), grp[metrics].std(ddof=1)
    n = grp["seed"].nunique()
    best = {m: (mean[m].max() if METRICS[m][1] else mean[m].min()) for m in metrics}
    c = comp[(comp["dataset"] == dataset) & (comp["alpha"] == alpha)].set_index(
        ["selector", "good", "metric"]) if not comp.empty else None
    order = sorted(mean.index, key=lambda k: (k[0] != REFERENCE, k[0], k[1]))
    arrows = [("$\\uparrow$" if METRICS[m][1] else "$\\downarrow$") for m in metrics]
    head = " & ".join(f"{METRICS[m][0]} {ar}" for m, ar in zip(metrics, arrows))
    lines = ["\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\toprule",
             f"Method & {head} \\\\", "\\midrule"]
    for sel, good in order:
        name = NAMES.get(sel, sel)
        if sel != REFERENCE and sub[sub["selector"] == sel]["good"].nunique() > 1:
            name += f" ({'wtd' if good == 'weighted' else 'cnt'})"
        cells = []
        for m in metrics:
            mu, sd = mean.loc[(sel, good), m], std.loc[(sel, good), m]
            if np.isnan(mu):
                cells.append("--")
                continue
            digits = 1 if m == "rounds_to_target" else 3
            txt = f"{mu:.{digits}f}"
            if not np.isnan(sd):
                txt += f" $\\pm$ {sd:.{digits}f}"
            if np.isclose(mu, best[m]):
                txt = f"\\textbf{{{txt}}}"
            if c is not None and (sel, good, m) in c.index and c.loc[(sel, good, m), test] < sig:
                txt += "$^\\dagger$"
            cells.append(txt)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    seeds = sorted(set(n))
    note = (f"% {dataset}, alpha={alpha:g}, seeds per method: {', '.join(map(str, seeds))}. "
            f"Mean +- std over seeds. Dagger: paired {test[2:]} vs Random, p < {sig}.")
    return note + "\n" + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--out", default=None, help="write all paired stats as CSV")
    ap.add_argument("--latex", default=None,
                    help="write one LaTeX table per (dataset, alpha); {dataset} and {alpha} "
                         "in the name are filled in, e.g. tables/main_{alpha}.tex")
    ap.add_argument("--test", choices=("ttest", "wilcoxon"), default="ttest")
    a = ap.parse_args()
    df = load_dir(a.results_dir)
    comp = paired(df)
    if comp.empty:
        print("nothing to compare: need Random and at least one other selector")
    else:
        show = comp.copy()
        show["metric"] = [METRICS[m][0].replace("\\", "") for m in show["metric"]]
        show["diff [95% CI]"] = [f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" for d, lo, hi
                                 in zip(show["diff_mean"], show["ci_lo"], show["ci_hi"])]
        cols = ["selector", "good", "alpha", "metric", "n_pairs", "diff [95% CI]",
                "p_ttest", "p_wilcoxon"]
        print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        comp.to_csv(a.out, index=False)
    if a.latex:
        for (ds, al), _ in df.groupby(["dataset", "alpha"]):
            path = a.latex.format(dataset=ds, alpha=f"{al:g}")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(latex_table(df, comp, ds, al, test=f"p_{a.test}"))
            print("wrote", path)


if __name__ == "__main__":
    main()
