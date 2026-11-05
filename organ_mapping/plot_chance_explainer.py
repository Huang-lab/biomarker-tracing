"""
Explainer for Figures 3 and 5: how an organ's "chance rank" (the black tick) is computed.

For one organ (e.g. the 35 lung cell types), take every disease that does not have that organ among its own organs (primary
or secondary) and find where the organ's best cell type lands in that disease's cell-type ranking. The median of those
positions is the chance rank: where the organ typically ranks for a disease that is not about it. A disease "beats chance"
when its own best rank for the organ is better (smaller) than that median. Organs that rank high for
every disease (generically associated proteins) get a low (strict) chance rank.

Usage (repository root):  python organ_mapping/plot_chance_explainer.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_organ_hits as E  # noqa: E402

PICKS = ["Asthma, excluding more controls", "Gastric ulcer", "Chronic kidney disease"]


def main():
    d = E.prepare("hpa700"); X, own, names = d["X"], d["own"], list(d["mp"].Disease)
    order = np.argsort(-X, axis=1); n = X.shape[1]
    fig, axes = plt.subplots(len(PICKS), 1, figsize=(10, 2.25 * len(PICKS) + 1.5), sharex=True)
    bins = np.logspace(0, np.log10(n), 40)
    for ax, nm in zip(axes, PICKS):
        i = names.index(nm); M = own[i]
        best = M[order].argmax(1) + 1
        excl = (d["own_any"] & M[None, :]).any(1)              # diseases that have this organ (primary or secondary) are not "other"
        others = best[~excl]; med = np.median(others); mine = best[i]; win = mine < med
        ax.hist(others, bins=bins, color=E.GRAY, edgecolor=E.SURFACE, lw=0.6)
        top = ax.get_ylim()[1] * 1.45; ax.set_ylim(0, top)
        ax.plot([med, med], [0, top * 0.93], color=E.INK, lw=2.2, solid_capstyle="butt")
        left = mine > med                                       # keep the label clear of the disease's own annotation
        ax.text(med / 1.08 if left else med * 1.08, top * 0.93, f"chance rank = median = {med:.0f}", fontsize=8, color=E.INK, va="top",
                ha="right" if left else "left")
        ax.annotate(f"{nm.split(',')[0]} itself: rank {mine}\n({'beats' if win else 'does not beat'} chance)", (mine, 0),
                    xytext=(mine * (1.35 if mine < 3 else 1), top * 0.80), ha="left" if mine < 3 else "center", fontsize=8, color=E.BLUE if win else E.INK2, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=E.BLUE if win else E.INK2, lw=1.4))
        ax.scatter([mine], [0], s=70, color=E.BLUE if win else E.INK2, zorder=5, clip_on=False, edgecolors=E.SURFACE, lw=1.2)
        organs = " + ".join(o.replace(".", " ") for o in d["mp"].organ_dict[i])
        ax.set_title(f"{nm}   ·   own organ: {organs} ({int(M.sum())} of {n} cell types)   ·   "
                     f"{(others <= mine).sum()} of {len(others)} diseases without it rank it at least as high → p = {(1 + (others <= mine).sum()) / (1 + len(others)):.3f}",
                     fontsize=8.8, color=E.INK, loc="left")
        ax.set_xscale("log"); ax.set_xlim(0.8, n * 1.1)
        ax.set_xticks([1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["1", "3", "10", "30", "100", "300"]); ax.minorticks_off()
        E.clean(ax); ax.set_ylabel("diseases without\nthis organ", fontsize=8)
    axes[-1].set_xlabel("rank of the organ's best cell type in a disease's ranking (of 700; left = higher)")
    fig.text(0.01, 0.99, "How the chance rank (black tick) and the p-value in Figures 3 and 5 are computed", fontsize=12.5, fontweight="bold", color=E.INK, va="top")
    fig.text(0.01, 0.955, "Gray bars: take every disease that does NOT have this organ among its own organs, and find where the organ's best cell type lands in that disease's ranking.\n"
             "Black line: the median of those positions = the organ's chance rank. The disease beats chance if its own rank (dot) is left of the black line.\n"
             "p = (1 + gray diseases at or left of the dot) / (1 + all gray diseases): how unusual the disease is against the whole spread, not just the median.",
             fontsize=8, color=E.INK2, va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.08, hspace=0.55)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(E.OUT, f"fig3b_how_chance_works.{ext}"), dpi=200 if ext == "png" else None)


if __name__ == "__main__":
    main()
