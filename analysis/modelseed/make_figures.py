#!/usr/bin/env python
"""Figures for the dGbyG vs ModelSEED-thermodynamics comparison.

Drawn in the manuscript's own idiom (MSDB_thermo_paper/analysis/grace_style.py:
closed black frame, inward ticks on all sides, no grid) and palette
(figure_common.py: eQuilibrator BLUE, Group contribution AQUA, dGPredictor
VIOLET; direction segments forward #1c5cab / reversible AQUA / reverse #c2410c /
undetermined NEUTRAL). dGbyG takes ORANGE, the reference palette's slot 2; its
structure-repaired variant is the same orange with hatching, so the two read as
one model under two inputs. The four source hues validate all-pairs (worst CVD
dE 9.2, normal-vision 16.3; dataviz validator, Python port). AQUA sits under
3:1 on white, so every mark is direct-labelled and identity never rests on
colour alone.

Reads results/comparison/*.tsv written by compare.py and the per-reaction
tables; writes results/figures/fig{1..5}_*.{pdf,png}.

  fig1_coverage_sigma      coverage (Fig 2B analogue) + reported sigma (Fig 2C)
  fig2_direction           direction calls per source (Fig 3A) + agreement matrix
  fig3_anchors             predicted vs measured on the 363 live anchors
  fig4_calibration         |error| / reported sigma; dGbyG's own-training CV
  fig5_energy_agreement    dGbyG against each source on shared reactions
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CMP = RES / "comparison"
FIG = RES / "figures"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
YELLOW, NEUTRAL, INK, MUTED = "#eda100", "#c9c8c2", "#000000", "#4d4d4d"
FWD, BACK = "#1c5cab", "#c2410c"
COL = {"eQuilibrator": BLUE, "dGPredictor": VIOLET, "Group contribution": AQUA, "dGbyG": ORANGE,
       "dGbyG (repaired)": ORANGE}
HATCH = {"dGbyG (repaired)": "////"}
SHORT = {"eQuilibrator": "eQ", "dGPredictor": "dGP", "Group contribution": "GC", "dGbyG": "dGbyG",
         "dGbyG (repaired)": "dGbyG rep.", "LLMs": "LLMs"}
KEY = {"eQuilibrator": "eq", "dGPredictor": "dgp", "Group contribution": "gc", "dGbyG": "dgbyg",
       "dGbyG (repaired)": "dgbyr"}
ORDER = ["eQuilibrator", "dGPredictor", "Group contribution", "dGbyG", "dGbyG (repaired)"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.0, "axes.linewidth": 0.8,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.grid": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "legend.frameon": False, "hatch.linewidth": 0.6,
    "hatch.color": "white",
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {FIG / name}.{{pdf,png}}")


def tag(ax, letter, x=0.015, y=0.985, ha="left"):
    ax.annotate(letter, xy=(x, y), xycoords="axes fraction", fontsize=9, fontweight="bold",
                va="top", ha=ha, color=INK, zorder=6)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def live_table():
    rp = pd.read_csv(RES / "reaction_predictions.tsv", sep="\t", keep_default_na=False, dtype=str)
    live = rp[rp["is_obsolete"].astype(int) != 1].copy()
    live["dgbyg_dG_kcal"], live["dgbyg_sd_kcal"], live["dgbyg_op"] = live["dG_kcal"], live["sd_kcal"], live["op"]
    rep = RES / "repaired" / "reaction_predictions.tsv"
    sources = [s for s in ORDER if s != "dGbyG (repaired)" or rep.exists()]
    if rep.exists():
        r2 = pd.read_csv(rep, sep="\t", keep_default_na=False, dtype=str).set_index("rxn")
        for col, src in (("dgbyr_dG_kcal", "dG_kcal"), ("dgbyr_sd_kcal", "sd_kcal"), ("dgbyr_op", "op")):
            live[col] = r2[src].reindex(live["rxn"]).fillna("").to_numpy()
    for s in sources:
        k = KEY[s]
        dg, sd = live[f"{k}_dG_kcal"].map(fnum), live[f"{k}_sd_kcal"].map(fnum).abs()
        live[f"{k}_dg"], live[f"{k}_sd"] = dg, sd
        live[f"{k}_cov"] = dg.notna() & sd.notna() & (dg < 1e7) & (sd < 2500)
    return live, sources


# ---------------------------------------------------------------- figure 1
def fig1(live, sources, cov, numbers):
    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], left=0.115, right=0.985, top=0.97, bottom=0.13,
                          wspace=0.14)
    a = fig.add_subplot(gs[0, 0])
    right = gs[0, 1].subgridspec(len(sources), 1, hspace=0.55)
    tot = numbers["live"]
    c = cov.set_index("source")
    rows = sorted(sources, key=lambda s: c.loc[s, "covered"])
    for i, s in enumerate(rows):
        v = c.loc[s, "covered"]
        a.barh(i, v, height=0.66, color=COL[s], hatch=HATCH.get(s), zorder=3, lw=0)
        a.text(v - 600, i, f"{100 * v / tot:.0f}%", va="center", ha="right", fontsize=6.4, color="white",
               fontweight="bold", bbox=dict(facecolor=COL[s], edgecolor="none", pad=0.6) if s in HATCH else None)
        a.text(v + 500, i, f"{v:,}", va="center", ha="left", fontsize=6.2, color=MUTED)
        a.text(-800, i, SHORT[s], va="center", ha="right", fontsize=7.0, color=INK)
    a.set_yticks([]); a.set_xlim(0, tot * 1.02); a.set_ylim(-0.6, len(rows) - 0.4)
    a.set_xticks([0, 10000, 20000, 30000, 40000, tot])
    a.set_xticklabels(["0", "10k", "20k", "30k", "40k", f"{tot // 1000}k"])
    a.set_xlabel("Live reactions with an energy estimate", labelpad=2)
    a.minorticks_off()
    tag(a, "A")

    band = numbers["silver_sigma_band"]
    for j, s in enumerate(sources):
        ax = fig.add_subplot(right[j])
        v = np.sort(live.loc[live[f"{KEY[s]}_cov"], f"{KEY[s]}_sd"].to_numpy())
        hi = v[int(0.97 * len(v))]
        ax.hist(v[v <= hi], bins=30, range=(0, hi), color=COL[s], hatch=HATCH.get(s), zorder=3, linewidth=0)
        if band.get(s):
            lo, bh, _ = band[s]
            ax.axvspan(lo, min(bh, hi), color=YELLOW, alpha=0.20, lw=0, zorder=2)
        med = v[len(v) // 2]
        ax.axvline(med, color=INK, lw=0.9, zorder=4)
        ax.text(0.30, 0.86, SHORT[s], transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=INK)
        ax.text(0.97, 0.86, f"median {med:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.0,
                color=INK)
        ax.set_yticks([]); ax.set_xlim(0, hi)
        if j == 0:
            tag(ax, "B", x=-0.05, y=1.0, ha="right")
        if j == len(sources) - 1:
            ax.set_xlabel("Reported σ (kcal mol$^{-1}$), own axis per source", labelpad=2)
    save(fig, "fig1_coverage_sigma")


# ---------------------------------------------------------------- figure 2
def fig2(live, sources):
    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], left=0.11, right=0.95, top=0.95, bottom=0.16,
                          wspace=0.32)
    a = fig.add_subplot(gs[0, 0])
    srcs = sources + ["LLMs"]
    opcol = lambda s: "llm_op" if s == "LLMs" else f"{KEY[s]}_op"  # noqa: E731
    counts = []
    for s in srcs:
        ops = live[opcol(s)][live[opcol(s)] != ""]
        counts.append([(ops == o).sum() for o in (">", "=", "<", "?")])
    amax = max(sum(c) for c in counts)
    for i, (s, cs) in enumerate(zip(srcs[::-1], counts[::-1])):
        x = 0
        for v, colr in zip(cs, (FWD, AQUA, BACK, NEUTRAL)):
            a.barh(i, v, left=x, height=0.66, color=colr, zorder=3, edgecolor="white", lw=0.6)
            x += v
        a.text(-amax * 0.02, i, SHORT[s], va="center", ha="right", fontsize=7.0, color=INK)
        a.text(x + amax * 0.01, i, f"{x:,}", va="center", ha="left", fontsize=6.0, color=MUTED)
    a.set_yticks([]); a.set_xlim(0, amax * 1.16); a.set_ylim(-0.6, len(srcs) + 0.2); a.minorticks_off()
    a.set_xticks([0, 10000, 20000, 30000, 40000]); a.set_xticklabels(["0", "10k", "20k", "30k", "40k"])
    a.set_xlabel("Live reactions carrying a direction call", labelpad=2)
    a.legend(handles=[Patch(facecolor=c, label=l) for l, c in (("forward", FWD), ("reversible", AQUA),
                                                               ("reverse", BACK), ("undetermined", NEUTRAL))],
             loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=5.8, ncol=2, handlelength=0.9,
             handleheight=0.8, labelspacing=0.3, labelcolor=INK)
    tag(a, "A")

    b = fig.add_subplot(gs[0, 1])
    n = len(srcs)
    m = np.full((n, n), np.nan)
    nn = np.zeros((n, n), dtype=int)
    for i, s in enumerate(srcs):
        for j, t in enumerate(srcs):
            if i == j:
                continue
            both = live[opcol(s)].isin([">", "<"]) & live[opcol(t)].isin([">", "<"])
            nn[i, j] = both.sum()
            m[i, j] = 100 * (live.loc[both, opcol(s)] == live.loc[both, opcol(t)]).mean() if both.any() else np.nan
    im = b.imshow(m, cmap="Blues", vmin=75, vmax=100)
    for i in range(n):
        for j in range(n):
            if i != j:
                b.text(j, i, f"{m[i, j]:.1f}\n{nn[i, j]:,}", ha="center", va="center", fontsize=4.6,
                       color="white" if m[i, j] > 92 else INK)
    b.set_xticks(range(n)); b.set_yticks(range(n))
    b.set_xticklabels([SHORT[s] for s in srcs], fontsize=5.8, rotation=30, ha="right")
    b.set_yticklabels([SHORT[s] for s in srcs], fontsize=5.8)
    b.minorticks_off(); b.tick_params(top=False, right=False)
    cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.03)
    cb.set_label("% agree where both commit", fontsize=5.8); cb.ax.tick_params(labelsize=5.6)
    tag(b, "B")
    save(fig, "fig2_direction")


# ---------------------------------------------------------------- figure 3
def fig3(an, acc):
    accd = acc[acc["subset"] == "own coverage"].set_index("source")
    row1 = [("eQuilibrator", "eq_dg", "eQ"), ("dGPredictor", "dgp_dg", "dGP"),
            ("Group contribution", "gc_dg", "GC"), ("dGbyG", "dgbyg_dg", "dGbyG, ModelSEED")]
    row2 = [("dGbyG KEGG structures", "dgbyg_kegg_dg", "dGbyG, KEGG structures"),
            ("dGbyG (held-out)", "dgbyg_heldout_dg", "held-out, ModelSEED"),
            ("dGbyG (repaired, held-out)", "dgbyg_heldout_fixed_dg", "held-out, repaired"),
            ("dGbyG KEGG structures (held-out)", "dgbyg_heldout_kegg_dg", "held-out, KEGG structures")]
    row2 = [p for p in row2 if p[1] in an and p[0] in accd.index]
    rows = [row1] + ([row2] if row2 else [])
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 1.95 * len(rows) + 0.1), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.10 if len(rows) > 1 else 0.21,
                        wspace=0.12, hspace=0.42)
    lim = (-25, 25)
    for r, panels in enumerate(rows):
        for c in range(4):
            ax = axes[r][c]
            if c >= len(panels):
                ax.axis("off")
                continue
            src, col, title = panels[c]
            color = COL.get(src.split(" (")[0].replace(" KEGG structures", ""), ORANGE)
            d = an[[col, "dG_exp"]].dropna()
            x, y = d["dG_exp"].clip(*lim), d[col].clip(*lim)
            ax.fill_between(lim, [lim[0] - 2, lim[1] - 2], [lim[0] + 2, lim[1] + 2], color=NEUTRAL, alpha=0.35,
                            lw=0)
            ax.plot(lim, lim, color=INK, lw=0.6)
            ax.scatter(x, y, s=7, color=color, edgecolor="white", lw=0.3, zorder=3,
                       marker="o" if "held-out" not in src else "D")
            ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
            ax.set_title(title, fontsize=6.6, pad=3)
            rr = accd.loc[src]
            ax.text(0.04, 0.96, f"MAE {rr['mae']:.2f}\n≤2: {rr['within_tau_pct']:.0f}%\nn={int(rr['n'])}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=5.6, color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel("Measured (openTECR)", fontsize=6.2, labelpad=1)
            if c == 0:
                ax.set_ylabel("Predicted Δ$_r$G′° (kcal mol$^{-1}$)", fontsize=6.2)
            else:
                ax.set_yticklabels([])
    save(fig, "fig3_anchors")


# ---------------------------------------------------------------- figure 4
def _zcurve(ax, an, curves):
    zs = np.linspace(0, 6, 301)
    from scipy.stats import norm
    ax.plot(zs, 2 * norm.cdf(zs) - 1, color=INK, lw=0.8, ls="--", label="calibrated Gaussian")
    for lab, dc, sc, color, ls in curves:
        if dc not in an:
            continue
        d = an[[dc, sc, "dG_exp"]].dropna()
        d = d[d[sc] > 0]
        z = np.sort(np.abs(d[dc] - d["dG_exp"]) / d[sc])
        ax.plot(zs, np.searchsorted(z, zs, side="right") / len(z), color=color, lw=1.5, ls=ls,
                label=f"{lab} (n={len(z)})")
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.0)
    ax.set_xlabel("|predicted − measured| / reported σ", labelpad=2)
    ax.legend(loc="lower right", fontsize=5.4, handlelength=1.8, labelcolor=INK)


def fig4(an, otc_path):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.18, wspace=0.28)
    _zcurve(axes[0], an, [("eQ", "eq_dg", "eq_sd", BLUE, "-"), ("dGP", "dgp_dg", "dgp_sd", VIOLET, "-"),
                          ("GC", "gc_dg", "gc_sd", AQUA, "-"), ("dGbyG", "dgbyg_dg", "dgbyg_sd", ORANGE, "-"),
                          ("dGbyG KEGG", "dgbyg_kegg_dg", "dgbyg_kegg_sd", ORANGE, ":")])
    axes[0].set_ylabel("Share of anchored reactions", labelpad=2)
    axes[0].set_title("shipped models", fontsize=6.6, pad=3)
    tag(axes[0], "A")
    _zcurve(axes[1], an, [("ModelSEED", "dgbyg_heldout_dg", "dgbyg_heldout_sd", ORANGE, "-"),
                          ("repaired", "dgbyg_heldout_fixed_dg", "dgbyg_heldout_fixed_sd", ORANGE, "--"),
                          ("KEGG", "dgbyg_heldout_kegg_dg", "dgbyg_heldout_kegg_sd", ORANGE, ":")])
    axes[1].set_title("dGbyG held-out (grouped 10-fold CV)", fontsize=6.6, pad=3)
    tag(axes[1], "B")

    b = axes[2]
    if otc_path.exists():
        t = pd.read_csv(otc_path, sep="\t")
        t = t[t["type"] == "reaction dg"]
        e = np.abs(t["pred_kcal"] - t["measured_kcal"])
        b.scatter(t["pred_sd_kcal"], e, s=6, color=ORANGE, edgecolor="white", lw=0.3, zorder=3)
        mx = float(np.quantile(t["pred_sd_kcal"], 0.99))
        b.plot([0, mx], [0, mx], color=INK, lw=0.6)
        b.plot([0, mx], [0, 2 * mx], color=INK, lw=0.6, ls="--")
        b.set_xlim(0, mx); b.set_ylim(0, max(2 * mx, float(np.quantile(e, 0.98))))
        b.text(0.97, 0.95, "lines: |error| = σ, 2σ", transform=b.transAxes, ha="right", va="top", fontsize=5.6)
        b.set_xlabel("held-out σ (kcal mol$^{-1}$)", labelpad=2)
        b.set_ylabel("|held-out error| (kcal mol$^{-1}$)", labelpad=2)
        b.set_title(f"own training reactions, n={len(t)}", fontsize=6.6, pad=3)
    else:
        b.text(0.5, 0.5, "cross-validation pending", ha="center", va="center", transform=b.transAxes)
        b.set_xticks([]); b.set_yticks([])
    tag(b, "C")
    save(fig, "fig4_calibration")


# ---------------------------------------------------------------- figure 5
def fig5(live):
    others = ["eQuilibrator", "dGPredictor", "Group contribution"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.18, wspace=0.08)
    lim = (-60, 60)
    for ax, s in zip(axes, others):
        k = KEY[s]
        m = live["dgbyg_cov"] & live[f"{k}_cov"]
        x, y = live.loc[m, f"{k}_dg"], live.loc[m, "dgbyg_dg"]
        ax.hexbin(x.clip(*lim), y.clip(*lim), gridsize=60, extent=(*lim, *lim), bins="log", cmap="Oranges",
                  mincnt=1, linewidths=0)
        ax.plot(lim, lim, color=INK, lw=0.6)
        d = y - x
        ax.text(0.04, 0.96, f"n={m.sum():,}\nr={np.corrcoef(x, y)[0, 1]:.3f}\nmedian |Δ| {np.median(np.abs(d)):.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=5.8, color=INK)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"{SHORT[s]} Δ$_r$G′° (kcal mol$^{{-1}}$)", labelpad=2)
    axes[0].set_ylabel("dGbyG Δ$_r$G′° (kcal mol$^{-1}$)", labelpad=2)
    save(fig, "fig5_energy_agreement")


def main():
    live, sources = live_table()
    cov = pd.read_csv(CMP / "coverage.tsv", sep="\t")
    sources = [s for s in sources if s in set(cov["source"])]
    numbers = json.load(open(CMP / "numbers.json"))
    an = pd.read_csv(CMP / "anchors.tsv", sep="\t").set_index("rxn")
    acc = pd.read_csv(CMP / "anchor_accuracy.tsv", sep="\t")
    fig1(live, sources, cov, numbers)
    fig2(live, sources)
    fig3(an, acc)
    fig4(an, RES / "cv" / "training_heldout.tsv")
    fig5(live)


if __name__ == "__main__":
    main()
