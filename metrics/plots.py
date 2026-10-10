"""Report figures.

Our rules are coloured (prop_fair blue, guarantee orange, ef1 aqua). Filled markers are
count-based, hollow ones weighted. Baselines are gray with distinct markers.

Usage: python -m metrics.plots [results_dir] [--out figures] [--alpha 0.1] [--seed 0]
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from metrics.loader import KEYS, load_dir, load_run, size_breakdown, summarize  # noqa: E402
from metrics.metrics import worst_frac_mean  # noqa: E402

TEXT, TEXT2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
OURS = {"prop_fair": "#2a78d6", "guarantee": "#eb6834", "ef1": "#1baf7a"}
BASELINES = {  # selector -> (gray, marker)
    "random": ("#3d3c39", "o"),
    "round_robin": ("#6b6a66", "s"),
    "power_of_choice": ("#3d3c39", "^"),
    "qffl": ("#6b6a66", "D"),
    "divfl": ("#3d3c39", "v"),
    "propfair": ("#6b6a66", "P"),
}
NAMES = {"random": "Random", "round_robin": "Round-Robin", "power_of_choice": "Power-of-Choice",
         "qffl": "q-FFL", "divfl": "DivFL", "propfair": "PropFair",
         "prop_fair": "Prop. fair", "guarantee": "Guarantee", "ef1": "EF1 lag"}


def style(selector, good):
    """Colour, marker, fill and label for a (selector, good_def) pair."""
    if selector in OURS:
        c = OURS[selector]
        return {"color": c, "marker": "o",
                "face": c if good == "count" else SURFACE,
                "label": f"{NAMES[selector]} ({good})"}
    c, m = BASELINES.get(selector, (MUTED, "X"))
    return {"color": c, "marker": m, "face": c, "label": NAMES.get(selector, selector)}


def _setup_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(nrows, ncols, w, h, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False, facecolor=SURFACE,
                             **kw)
    for ax in axes.flat:
        _setup_ax(ax)
    return fig, axes


def _method_order(keys):
    """Baselines first, then our rules. Count before weighted."""
    rank = list(BASELINES) + list(OURS)
    return sorted(keys, key=lambda k: (rank.index(k[0]) if k[0] in rank else 99, k[1]))


def load_raw(results_dir):
    return [load_run(p) for p in sorted(glob.glob(os.path.join(results_dir, "*.json")))]


def plot_client_acc_hist(runs, alpha, bins=20):
    """Per-client accuracy histogram per method, pooled over seeds.
    Solid line is the mean, dotted line the worst-10% mean."""
    groups = {}
    seeds = {r[KEYS["seed"]] for r in runs if r[KEYS["alpha"]] == alpha}
    for r in runs:
        if r[KEYS["alpha"]] == alpha:
            groups.setdefault((r[KEYS["selector"]], r[KEYS["good"]]), []).extend(r[KEYS["client_acc"]])
    keys = _method_order(groups)
    if not keys:
        raise ValueError(f"no runs at alpha={alpha}")
    edges = np.linspace(0, 1, bins + 1)
    fig, axes = _fig(len(keys), 1, 6, 1.3 * len(keys) + 0.6, sharex=True)
    for ax, k in zip(axes[:, 0], keys):
        acc = np.asarray(groups[k])
        st = style(*k)
        ax.hist(acc, bins=edges, color=st["color"], alpha=0.85, rwidth=0.9)
        ax.axvline(acc.mean(), color=TEXT, linewidth=1.2)
        ax.axvline(worst_frac_mean(acc), color=TEXT, linewidth=1.2, linestyle=":")
        ax.set_title(st["label"], fontsize=9, color=TEXT, loc="left")
        ax.set_title(f"mean {acc.mean():.3f}   worst-10% {worst_frac_mean(acc):.3f}",
                     fontsize=8, color=TEXT2, loc="right")
        ax.set_xlim(0, 1)
        ax.set_ylabel("clients", fontsize=8, color=TEXT2)
    axes[-1, 0].set_xlabel("final per-client test accuracy", fontsize=9, color=TEXT2)
    fig.suptitle(f"Per-client accuracy, alpha = {alpha:g}, {len(seeds)} seed(s)", fontsize=10,
                 color=TEXT, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def _direct_labels(ax, pts, gap_px=11):
    """Label points on the right and push overlapping labels down."""
    fig = ax.figure
    fig.canvas.draw()
    disp = [(ax.transData.transform((x, y)), lab) for x, y, lab in pts]
    disp.sort(key=lambda d: -d[0][1])
    placed = []
    for (px, py), lab in disp:
        ty = py + 4
        for qx, qy in placed:
            if abs(qx - px) < 90 and abs(qy - ty) < gap_px:
                ty = qy - gap_px
        placed.append((px, ty))
        tx, tyd = ax.transData.inverted().transform((px + 7, ty))
        ax.annotate(lab, ax.transData.inverted().transform((px, py)), xytext=(tx, tyd),
                    textcoords="data", fontsize=7, color=TEXT2, va="center",
                    arrowprops=None if abs(ty - py - 4) < 1 else
                    {"arrowstyle": "-", "color": GRID, "linewidth": 0.8})


def plot_tradeoff(summary, y="worst10_mean_final", x="global_acc_final", ylabel=None):
    """Accuracy vs a fairness metric, one panel per alpha, mean +- std over seeds."""
    alphas = sorted(summary["alpha"].unique())
    fig, axes = _fig(1, len(alphas), 4.2 * len(alphas), 3.8)
    for ax, a in zip(axes[0], alphas):
        sub = summary[summary["alpha"] == a]
        keys = _method_order(list(zip(sub["selector"], sub["good"])))
        pts = []
        for sel, good in keys:
            row = sub[(sub["selector"] == sel) & (sub["good"] == good)].iloc[0]
            st = style(sel, good)
            ax.errorbar(row[f"{x}_mean"], row[f"{y}_mean"],
                        xerr=np.nan_to_num(row[f"{x}_std"]), yerr=np.nan_to_num(row[f"{y}_std"]),
                        fmt=st["marker"], ms=8, mfc=st["face"], mec=st["color"], mew=1.5,
                        ecolor=st["color"], elinewidth=1, capsize=0, label=st["label"])
            pts.append((row[f"{x}_mean"], row[f"{y}_mean"], st["label"]))
        _direct_labels(ax, pts)
        ax.set_title(f"alpha = {a:g}", fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel(x.replace("_", " "), fontsize=8, color=TEXT2)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    axes[0, 0].set_ylabel(ylabel or y.replace("_", " "), fontsize=8, color=TEXT2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def plot_selection_counts(runs, alpha, seed):
    """Selection count per client, sorted by data size, with the fair-share line.
    Fair share is K*R/N for count and w_i*K*R for weighted."""
    picked = {(r[KEYS["selector"]], r[KEYS["good"]]): r for r in runs
              if r[KEYS["alpha"]] == alpha and r[KEYS["seed"]] == seed}
    keys = _method_order(picked)
    if not keys:
        raise ValueError(f"no runs at alpha={alpha}, seed={seed}")
    fig, axes = _fig(len(keys), 1, 7, 1.4 * len(keys) + 0.6)
    for ax, k in zip(axes[:, 0], keys):
        r = picked[k]
        sizes = np.asarray(r[KEYS["sizes"]], dtype=float)
        counts = np.asarray(r[KEYS["counts"]], dtype=float)
        n, kk, rounds = len(sizes), len(r[KEYS["selected"]][0]), len(r[KEYS["selected"]])
        order = np.argsort(sizes)
        st = style(*k)
        ax.bar(np.arange(n), counts[order], width=0.8, color=st["color"], alpha=0.85)
        fair = (sizes[order] / sizes.sum() * kk * rounds if k[1] == "weighted"
                else np.full(n, kk * rounds / n))
        ax.step(np.arange(n), fair, where="mid", color=TEXT, linewidth=1.2)
        ax.set_ylim(0, max(counts.max(), fair.max()) * 1.08)
        ax.set_title(st["label"], fontsize=9, color=TEXT, loc="left")
        ax.set_title(f"count variance {counts.var():.1f}", fontsize=8, color=TEXT2, loc="right")
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_xticks([])
        ax.set_ylabel("times selected", fontsize=8, color=TEXT2)
    axes[-1, 0].set_xlabel("clients sorted by data size (small to large), line is fair share",
                           fontsize=9, color=TEXT2)
    fig.suptitle(f"Selection counts, alpha = {alpha:g}, seed {seed}", fontsize=10, color=TEXT,
                 x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plot_size_breakdown(bd):
    """Accuracy (top) and selection count (bottom) per data-size bucket, one column per alpha."""
    alphas = sorted(bd["alpha"].unique())
    nb = int(bd["bucket"].max()) + 1
    fig, axes = _fig(2, len(alphas), 4 * len(alphas), 5.6, sharex=True)
    for j, a in enumerate(alphas):
        sub = bd[bd["alpha"] == a]
        keys = _method_order(list({(r.selector, r.good) for r in sub.itertuples()}))
        for i, (col, ylab) in enumerate((("acc", "mean client accuracy"),
                                          ("count", "mean times selected"))):
            ax = axes[i, j]
            for sel, good in keys:
                m = sub[(sub["selector"] == sel) & (sub["good"] == good)].sort_values("bucket")
                st = style(sel, good)
                ax.errorbar(m["bucket"], m[f"{col}_mean"], yerr=m[f"{col}_std"].fillna(0),
                            color=st["color"], marker=st["marker"], ms=7, mfc=st["face"],
                            mec=st["color"], mew=1.5, linewidth=1.5, elinewidth=1, capsize=0,
                            label=st["label"])
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8, color=TEXT2)
        axes[0, j].set_title(f"alpha = {a:g}", fontsize=9, color=TEXT, loc="left")
        ranges = sub.groupby("bucket")[["size_min", "size_max"]].agg({"size_min": "min",
                                                                      "size_max": "max"})
        axes[1, j].set_xticks(range(nb))
        axes[1, j].set_xticklabels([f"Q{b + 1}\n{int(r.size_min)}-{int(r.size_max)}"
                                    for b, r in ranges.iterrows()], fontsize=7)
        axes[1, j].set_xlabel("data-size bucket (Q1 = smallest), sample range", fontsize=8,
                              color=TEXT2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_gate1(run):
    """Per-client accuracy against selection count and data size for one run."""
    from scipy.stats import spearmanr
    acc = np.asarray(run[KEYS["client_acc"]], dtype=float)
    colour = style(run[KEYS["selector"]], run[KEYS["good"]])["color"]
    fig, axes = _fig(1, 2, 8, 3.2)
    for ax, key, label, log in ((axes[0, 0], "counts", "times selected", False),
                                (axes[0, 1], "sizes", "local data size (log)", True)):
        x = np.asarray(run[KEYS[key]], dtype=float)
        rho, p = spearmanr(x, acc)
        ax.scatter(x, acc, s=28, color=colour, edgecolor=SURFACE, linewidth=1)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label, fontsize=8, color=TEXT2)
        ax.set_title(f"Spearman rho = {rho:+.2f} (p = {p:.2f})", fontsize=9, color=TEXT, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    axes[0, 0].set_ylabel("final client accuracy", fontsize=8, color=TEXT2)
    fig.suptitle(f"Gate 1, {NAMES.get(run[KEYS['selector']], run[KEYS['selector']])}, "
                 f"alpha = {run[KEYS['alpha']]:g}, seed {run[KEYS['seed']]}",
                 fontsize=10, color=TEXT, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    runs = load_raw(a.results_dir)
    summary = summarize(load_dir(a.results_dir))
    save(plot_client_acc_hist(runs, a.alpha), a.out, f"client_acc_hist_alpha{a.alpha}")
    save(plot_tradeoff(summary, "worst10_mean_final", ylabel="worst-10% client accuracy"),
         a.out, "tradeoff_worst10")
    save(plot_tradeoff(summary, "jain_final", ylabel="Jain's index (client accuracy)"),
         a.out, "tradeoff_jain")
    save(plot_selection_counts(runs, a.alpha, a.seed), a.out,
         f"selection_counts_alpha{a.alpha}_seed{a.seed}")
    bd = size_breakdown(runs)
    save(plot_size_breakdown(bd), a.out, "size_breakdown")
    bd.to_csv(os.path.join(a.out, "size_breakdown.csv"), index=False)
    print(f"wrote figures to {a.out}")


if __name__ == "__main__":
    main()
