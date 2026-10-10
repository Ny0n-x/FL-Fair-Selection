"""Paper figures.

Our rules are coloured (prop_fair blue, guarantee orange, ef1 aqua). Solid lines and filled
markers are count-based, dashed lines and hollow markers weighted. Baselines are grays with
distinct markers. Widths match a single (3.4 in) or double (7 in) column.

Usage: python -m metrics.plots [results_dir] [--out figures] [--alpha 0.1] [--seed 0]
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from metrics.loader import KEYS, load_dir, load_run, size_breakdown, summarize  # noqa: E402
from metrics.metrics import worst_frac_mean  # noqa: E402

COL, FULL = 3.4, 7.0
INK, INK2, GRID = "#1a1a1a", "#555555", "#e5e5e5"
OURS = {"prop_fair": "#2a78d6", "guarantee": "#eb6834", "ef1": "#1baf7a"}
BASELINES = {  # selector -> (gray, marker)
    "random": ("#2b2b2b", "o"),
    "round_robin": ("#7a7a7a", "s"),
    "power_of_choice": ("#2b2b2b", "^"),
    "qffl": ("#7a7a7a", "D"),
    "divfl": ("#2b2b2b", "v"),
    "propfair": ("#7a7a7a", "P"),
}
NAMES = {"random": "Random", "round_robin": "Round-Robin", "power_of_choice": "Power-of-Choice",
         "qffl": "q-FFL", "divfl": "DivFL", "propfair": "PropFair",
         "prop_fair": "Proportional", "guarantee": "Guarantee", "ef1": "EF1-Lag"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.top": True, "ytick.right": True,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def style(selector, good):
    """Colour, marker, fill, line style and label for a (selector, good_def) pair."""
    if selector in OURS:
        c = OURS[selector]
        weighted = good == "weighted"
        return {"color": c, "marker": "o", "face": "white" if weighted else c,
                "ls": "--" if weighted else "-",
                "label": f"{NAMES[selector]} ({'wtd' if weighted else 'cnt'})"}
    c, m = BASELINES.get(selector, ("#999999", "X"))
    return {"color": c, "marker": m, "face": c, "ls": "-",
            "label": NAMES.get(selector, selector)}


def _order(keys):
    """Baselines first, then our rules. Count before weighted."""
    rank = list(BASELINES) + list(OURS)
    return sorted(keys, key=lambda k: (rank.index(k[0]) if k[0] in rank else 99, k[1]))


def _panel(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", pad=3)


def _group(runs, alpha):
    groups = {}
    for r in runs:
        if r[KEYS["alpha"]] == alpha:
            groups.setdefault((r[KEYS["selector"]], r[KEYS["good"]]), []).append(r)
    if not groups:
        raise ValueError(f"no runs at alpha={alpha}")
    return {k: groups[k] for k in _order(groups)}


def load_raw(results_dir):
    return [load_run(p) for p in sorted(glob.glob(os.path.join(results_dir, "*.json")))]


def _band(ax, x, ys, st, every):
    ys = np.asarray(ys, dtype=float)
    m = ys.mean(0)
    ax.plot(x, m, color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["face"],
            mec=st["color"], markevery=every, label=st["label"])
    if len(ys) > 1:
        s = ys.std(0)
        ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.15, lw=0)


def plot_curves(runs, alpha, smooth=5):
    """Test accuracy and worst-10% client accuracy against communication rounds.
    Lines are a `smooth`-round moving average, mean over seeds, bands are +- 1 std."""
    groups = _group(runs, alpha)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    w = np.ones(smooth) / smooth
    for k, rs in groups.items():
        st = style(*k)
        g = [np.convolve(r[KEYS["global_acc"]], w, "valid") for r in rs]
        x = np.arange(1, len(g[0]) + 1) + smooth - 1
        every = max(1, len(x) // 8)
        _band(axes[0], x, g, st, every)
        per = [r.get(KEYS["client_acc_rounds"]) for r in rs]
        if all(per):
            t = [np.convolve([worst_frac_mean(a) for a in p], w, "valid") for p in per]
            _band(axes[1], x, t, st, every)
    _panel(axes[0], "a", "Global test accuracy")
    _panel(axes[1], "b", "Worst-10% client accuracy")
    for ax in axes:
        ax.set_xlabel("Communication round")
        ax.set_xlim(0, x[-1])
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right", ncol=1 if len(groups) < 4 else 2)
    fig.tight_layout(w_pad=1.5)
    return fig


def plot_acc_cdf(runs, alpha):
    """Empirical CDF of final per-client accuracy, pooled over seeds."""
    groups = _group(runs, alpha)
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for k, rs in groups.items():
        st = style(*k)
        acc = np.sort(np.concatenate([r[KEYS["client_acc"]] for r in rs]))
        y = np.arange(1, acc.size + 1) / acc.size
        ax.step(acc, y, where="post", color=st["color"], ls=st["ls"], label=st["label"])
    ax.axhline(0.1, color=INK2, lw=0.6)
    ax.text(0.01, 0.12, "10%", color=INK2, fontsize=6, transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Final client test accuracy")
    ax.set_ylabel("Fraction of clients")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_participation(run):
    """Which clients were selected in which round, clients sorted by data size,
    with total selections per client on the right."""
    sel = run[KEYS["selected"]]
    sizes = np.asarray(run[KEYS["sizes"]], dtype=float)
    counts = np.asarray(run[KEYS["counts"]], dtype=float)
    n, rounds, k = sizes.size, len(sel), len(sel[0])
    rank = np.empty(n, dtype=int)
    rank[np.argsort(sizes)] = np.arange(n)
    st = style(run[KEYS["selector"]], run[KEYS["good"]])
    fig, (ax, side) = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True,
                                   gridspec_kw={"width_ratios": [5, 1], "wspace": 0.04})
    xs = np.repeat(np.arange(1, rounds + 1), k)
    ys = rank[np.concatenate(sel)]
    ax.scatter(xs, ys, s=2.5, marker="s", color=st["color"], lw=0)
    ax.grid(False)
    ax.set_xlim(0.5, rounds + 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Client (sorted by data size)")
    _panel(ax, "a", f"Selected clients per round ({st['label']})")
    side.barh(rank, counts, height=0.8, color=st["color"], lw=0)
    side.axvline(k * rounds / n, color=INK, lw=0.8)
    side.grid(axis="y", visible=False)
    side.set_xlabel("Times selected")
    _panel(side, "b", "Total")
    side.tick_params(labelleft=False)
    return fig


def plot_gate1(run):
    """Final client accuracy against selection count and data size, with Spearman rho."""
    acc = np.asarray(run[KEYS["client_acc"]], dtype=float)
    st = style(run[KEYS["selector"]], run[KEYS["good"]])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    panels = ((axes[0], "counts", "Times selected", "Participation", False),
              (axes[1], "sizes", "Local training samples", "Data size", True))
    for (ax, key, xlabel, name, log), letter in zip(panels, "ab"):
        x = np.asarray(run[KEYS[key]], dtype=float)
        rho, p = spearmanr(x, acc)
        ax.scatter(x, acc, s=12, color=st["color"], lw=0)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        _panel(ax, letter, name)
        ax.text(0.97, 0.05, rf"$\rho_s = {rho:+.2f}$, $p = {p:.2f}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7)
    axes[0].set_ylabel("Final client test accuracy")
    fig.tight_layout(w_pad=1.0)
    return fig


def plot_tradeoff(summary, y="worst10_mean_final", x="global_acc_final", ylabel=None):
    """Global accuracy against a fairness metric, one panel per alpha, mean +- std."""
    alphas = sorted(summary["alpha"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(FULL, 2.3), squeeze=False)
    for ax, a, letter in zip(axes[0], alphas, "abcdef"):
        sub = summary[summary["alpha"] == a]
        for sel, good in _order(list(zip(sub["selector"], sub["good"]))):
            row = sub[(sub["selector"] == sel) & (sub["good"] == good)].iloc[0]
            st = style(sel, good)
            ax.errorbar(row[f"{x}_mean"], row[f"{y}_mean"],
                        xerr=np.nan_to_num(row[f"{x}_std"]), yerr=np.nan_to_num(row[f"{y}_std"]),
                        fmt=st["marker"], ms=5, mfc=st["face"], mec=st["color"], mew=1,
                        ecolor=st["color"], elinewidth=0.8, capsize=1.5, label=st["label"])
        _panel(ax, letter, rf"$\alpha = {a:g}$")
        ax.set_xlabel("Global test accuracy")
    axes[0, 0].set_ylabel(ylabel or y)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def plot_selection_counts(run):
    """Selection count per client sorted by data size, against the fair share
    (K*R/N for count, w_i*K*R for weighted)."""
    sizes = np.asarray(run[KEYS["sizes"]], dtype=float)
    counts = np.asarray(run[KEYS["counts"]], dtype=float)
    n, k, rounds = sizes.size, len(run[KEYS["selected"]][0]), len(run[KEYS["selected"]])
    order = np.argsort(sizes)
    st = style(run[KEYS["selector"]], run[KEYS["good"]])
    fair = (sizes[order] / sizes.sum() * k * rounds if run[KEYS["good"]] == "weighted"
            else np.full(n, k * rounds / n))
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ax.bar(np.arange(n), counts[order], width=0.8, color=st["color"], lw=0, label=st["label"])
    ax.step(np.arange(n), fair, where="mid", color=INK, lw=0.9, label="Fair share")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Clients (sorted by data size)")
    ax.set_ylabel("Times selected")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def plot_size_breakdown(bd):
    """Accuracy (top) and selection count (bottom) per data-size quartile, one column per alpha."""
    alphas = sorted(bd["alpha"].unique())
    fig, axes = plt.subplots(2, len(alphas), figsize=(FULL, 3.6), sharex=True, squeeze=False)
    for j, a in enumerate(alphas):
        sub = bd[bd["alpha"] == a]
        for i, (col, ylab) in enumerate((("acc", "Client test accuracy"),
                                          ("count", "Times selected"))):
            ax = axes[i, j]
            for sel, good in _order(list({(r.selector, r.good) for r in sub.itertuples()})):
                m = sub[(sub["selector"] == sel) & (sub["good"] == good)].sort_values("bucket")
                st = style(sel, good)
                ax.errorbar(m["bucket"] + 1, m[f"{col}_mean"], yerr=m[f"{col}_std"].fillna(0),
                            color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["face"],
                            mec=st["color"], capsize=1.5, elinewidth=0.8, label=st["label"])
            if j == 0:
                ax.set_ylabel(ylab)
        _panel(axes[0, j], "abcdef"[j], rf"$\alpha = {a:g}$")
        axes[1, j].set_xticks(range(1, int(bd["bucket"].max()) + 2))
        axes[1, j].set_xlabel("Data-size quartile (1 = smallest)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    runs = load_raw(a.results_dir)
    tag = f"alpha{a.alpha:g}"
    save(plot_curves(runs, a.alpha), a.out, f"curves_{tag}")
    save(plot_acc_cdf(runs, a.alpha), a.out, f"client_acc_cdf_{tag}")
    for r in runs:
        if r[KEYS["alpha"]] == a.alpha and r[KEYS["seed"]] == a.seed:
            name = f"{r[KEYS['selector']]}_{r[KEYS['good']]}_{tag}_seed{a.seed}"
            save(plot_participation(r), a.out, f"participation_{name}")
            save(plot_selection_counts(r), a.out, f"selection_counts_{name}")
            if r[KEYS["selector"]] == "random":
                save(plot_gate1(r), a.out, f"gate1_{tag}_seed{a.seed}")
    summary = summarize(load_dir(a.results_dir))
    if summary["selector"].nunique() > 1:
        save(plot_tradeoff(summary, ylabel="Worst-10% client accuracy"), a.out, "tradeoff_worst10")
        save(plot_tradeoff(summary, "jain_final", ylabel="Jain's index"), a.out, "tradeoff_jain")
        save(plot_size_breakdown(size_breakdown(runs)), a.out, "size_breakdown")
    print(f"wrote figures to {a.out}")


if __name__ == "__main__":
    main()
