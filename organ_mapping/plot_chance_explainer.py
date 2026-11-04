"""
Explainer for Figure 3: how a disease's "chance rank" (the black tick) is computed.

For one disease, take its own-organ cell types (e.g. the 35 lung cell types for asthma). In EVERY OTHER disease's
cell-type ranking, find where the best of those cell types lands. That gives one number per other disease (190 here).
Their median is the chance rank: how high this organ typically ranks for a disease that is not about this organ.
The disease "beats chance" when its own best own-organ rank is better (smaller) than that median. By the definition of
a median, a disease whose ranking carried no organ information would beat it about half the time - hence "~50%
expected". Organs that rank high for every disease (generically associated proteins) get a low (strict) chance rank.

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
        others = np.delete(best, i); med = np.median(others); mine = best[i]; win = mine < med
        ax.hist(others, bins=bins, color=E.GRAY, edgecolor=E.SURFACE, lw=0.6)
        top = ax.get_ylim()[1]
        ax.plot([med, med], [0, top * 0.95], color=E.INK, lw=2.2, solid_capstyle="butt")
        ax.text(med * 1.08, top * 0.92, f"chance rank = median = {med:.0f}", fontsize=8, color=E.INK, va="top")
        ax.annotate(f"{nm.split(',')[0]} itself: rank {mine}\n({'beats' if win else 'does not beat'} chance)", (mine, 0),
                    xytext=(mine * (1.35 if mine < 3 else 1), top * 0.62), ha="left" if mine < 3 else "center", fontsize=8, color=E.BLUE if win else E.INK2, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=E.BLUE if win else E.INK2, lw=1.4))
        ax.scatter([mine], [0], s=70, color=E.BLUE if win else E.INK2, zorder=5, clip_on=False, edgecolors=E.SURFACE, lw=1.2)
        organs = " + ".join(o.replace(".", " ") for o in d["mp"].organ_dict[i])
        ax.set_title(f"{nm}   ·   own organ: {organs} ({int(M.sum())} of {n} cell types)   ·   "
                     f"{(others <= mine).sum()} of {len(others)} other diseases rank it at least as high",
                     fontsize=8.8, color=E.INK, loc="left")
        ax.set_xscale("log"); ax.set_xlim(0.8, n * 1.1)
        ax.set_xticks([1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["1", "3", "10", "30", "100", "300"]); ax.minorticks_off()
        E.clean(ax); ax.set_ylabel("other diseases", fontsize=8)
    axes[-1].set_xlabel("rank of the organ's best cell type in a disease's ranking (of 700; left = higher)")
    fig.text(0.01, 0.99, "How the chance rank (the black tick in Figures 3 and 5) is computed", fontsize=12.5, fontweight="bold", color=E.INK, va="top")
    fig.text(0.01, 0.955, "Gray: for each of the 190 OTHER diseases, where this disease's own-organ cell types first appear in that disease's ranking.\n"
             "Black: the median of those 190 numbers = the chance rank. The disease beats chance if its own rank (dot) is left of the black line.\n"
             "Because it is a median, a disease whose ranking says nothing about its organ would beat it about half the time: that is the '~50% expected'.",
             fontsize=8, color=E.INK2, va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.08, hspace=0.55)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(E.OUT, f"fig3b_how_chance_works.{ext}"), dpi=200 if ext == "png" else None)


if __name__ == "__main__":
    main()
