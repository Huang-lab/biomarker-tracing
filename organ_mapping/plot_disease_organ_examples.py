"""
Figure: the primary organ(s) assigned to a selection of diseases, with the ICD-10 codes they came from.

Reads only disease_organ_map.tsv (no expression data needed). Rows = diseases grouped by organ system;
columns = atlas organs. Filled dot = primary organ (size = share of the disease's organ-bearing ICD-10
categories pointing to it); open ring = secondary organ. Right: the ICD-10 categories used.

Usage (repository root):
    python organ_mapping/plot_disease_organ_examples.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_organ_heatmaps import SYSTEMS, SYS_COLOR, SYS_ORDER, ORGAN_SYSTEM, ORGAN_ORDER, SURFACE, INK, INK2, MUTED, HAIR  # noqa: E402

N_SHOW = 50                # number of diseases drawn
MIN_JACCARD = 0.5          # diseases with the same primary organs and this much ICD-10 overlap are near-duplicates


def pick_representatives(m, n_show=N_SHOW):
    """Group near-duplicate diseases (same primary organs, ICD-10 category sets overlapping by Jaccard >= MIN_JACCARD),
    represent each group by its medoid (highest mean overlap with the group; ties: higher confidence, shorter name),
    then choose n_show groups: first the largest group for every primary organ (so no organ is missed), then groups
    with a not-yet-shown organ combination (largest first), then the largest remaining groups. Returns [(row index, group size)]."""
    cats = [set(c.split(",")) - {""} for c in m.icd10_categories]
    organs = [frozenset(o for o in x.split(";") if o) for x in m.organs_primary]
    jac = lambda i, j: len(cats[i] & cats[j]) / max(1, len(cats[i] | cats[j]))
    parent = list(range(len(m)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(m)):
        for j in range(i):
            if organs[i] == organs[j] and (jac(i, j) >= MIN_JACCARD or (not cats[i] and not cats[j])): parent[find(i)] = find(j)
    groups = {}
    for i in range(len(m)): groups.setdefault(find(i), []).append(i)
    conf_rank = {"high": 0, "medium": 1, "review": 2}
    reps = []
    for g in groups.values():
        rep = min(g, key=lambda i: (-sum(jac(i, j) for j in g) / len(g), conf_rank.get(m.confidence[i], 3), len(m.Disease[i])))
        reps.append((rep, len(g)))
    reps.sort(key=lambda t: (-t[1], conf_rank.get(m.confidence[t[0]], 3), m.Disease[t[0]]))
    chosen, covered = [], set()
    for rep, size in reps:                                   # organ coverage first
        new = organs[rep] - covered
        if new and len(chosen) < n_show: chosen.append((rep, size)); covered |= organs[rep]
    while len(chosen) < n_show:                              # then groups with a not-yet-shown organ combination, largest first
        shown = {organs[r] for r, _ in chosen}
        rest = [t for t in reps if t not in chosen]
        if not rest: break
        chosen.append(min(rest, key=lambda t: (organs[t[0]] in shown, -t[1], conf_rank.get(m.confidence[t[0]], 3), m.Disease[t[0]])))
    print(f"{len(m)} scorable diseases -> {len(reps)} near-duplicate groups; showing {len(chosen)}, covering {len(covered)} primary organs "
          f"and {sum(s for _, s in chosen)} diseases")
    return chosen


def system_of(sup):
    score = {s: 0.0 for s in SYS_ORDER}
    for o, v in sup.items(): score[ORGAN_SYSTEM[o]] += v
    return max(SYS_ORDER, key=lambda s: (round(score[s], 6), -SYS_ORDER.index(s)))


def main():
    m = pd.read_csv(os.path.join(HERE, "disease_organ_map.tsv"), sep="\t", dtype=str).fillna("")
    m = m[m.status == "scorable"].reset_index(drop=True)
    rows = []
    picks = pick_representatives(m)
    total = sum(sz for _, sz in picks)
    for idx, size in picks:
        r = m.iloc[idx]
        sup = {}
        for x in r.organ_support.split(";"):
            o, v = x.split(":"); sup[o] = float(v)
        prim = {o: sup.get(o, 1.0) for o in r.organs_primary.split(";") if o}
        sec = [o for o in r.organs_secondary.split(";") if o]
        cats = r.icd10_categories.split(",")
        rows.append(dict(name=r.Disease, prim=prim, sec=sec, system=system_of(prim), cats=cats, conf=r.confidence, size=size))
    # order: organ-system order, then picks order
    rows.sort(key=lambda d: (SYS_ORDER.index(d["system"]), min(ORGAN_ORDER.index(o) for o in d["prim"]), d["name"]))
    used = {o for d in rows for o in list(d["prim"]) + d["sec"]}
    cols = [o for o in ORGAN_ORDER if o in used]

    n, k = len(rows), len(cols)
    fig = plt.figure(figsize=(13.5, 0.25 * n + 3.3), facecolor=SURFACE)
    L, R, T, B = 0.345, 0.80, 0.855, 0.035
    ax = fig.add_axes([L, B, R - L, T - B]); ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.5, k - 0.5); ax.set_ylim(n - 0.5, -0.5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # group bands, separators and system labels (x in figure coords, y in data coords)
    tr = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    start = 0
    for i in range(1, n + 1):
        if i == n or rows[i]["system"] != rows[start]["system"]:
            sysn = rows[start]["system"]
            ax.axhspan(start - 0.5, i - 0.5, color=SYS_COLOR[sysn], alpha=0.07, lw=0)
            fig.add_artist(plt.Rectangle((0.012, start - 0.4), 0.005, i - start - 0.2, transform=tr, color=SYS_COLOR[sysn], lw=0))
            fig.text(0.022, (start + i - 1) / 2, sysn.replace(", ", ",\n").replace(" & ", " &\n"), fontsize=8, color=INK, va="center",
                     ha="left", fontweight="bold", linespacing=1.15, transform=tr)
            if i < n: ax.axhline(i - 0.5, color=HAIR, lw=0.8)
            start = i

    for j in range(k): ax.axvline(j, color=HAIR, lw=0.4, zorder=0)
    for i, d in enumerate(rows):
        for o, v in d["prim"].items():
            ax.scatter(cols.index(o), i, s=60 + 150 * v, color=SYS_COLOR[ORGAN_SYSTEM[o]], edgecolors=SURFACE, linewidths=1.2, zorder=4)
        for o in d["sec"]:
            if o in cols and o not in d["prim"]:
                ax.scatter(cols.index(o), i, s=50, facecolors="none", edgecolors=MUTED, linewidths=1.1, zorder=3)
        nm = d["name"].split(" (")[0].split(", ")[0] if len(d["name"]) > 40 else d["name"]
        ax.text(-0.75, i, nm + ("*" if d["conf"] != "high" else ""), ha="right", va="center", fontsize=8, color=INK)
        shown = d["cats"][:5]
        ax.text(k - 0.35, i, f"×{d['size']}" if d["size"] > 1 else "", ha="left", va="center", fontsize=7.5, color=INK, fontweight="bold")
        ax.text(k + 1.5, i, ", ".join(shown) + (f" +{len(d['cats']) - 5}" if len(d["cats"]) > 5 else ""), ha="left", va="center",
                fontsize=7.2, color=INK2, family="monospace")
    for j, o in enumerate(cols):
        ax.text(j, -0.9, o.replace(".", " "), rotation=60, ha="left", va="bottom", fontsize=8, color=INK2, rotation_mode="anchor")
        ax.add_patch(plt.Rectangle((j - 0.3, -0.78), 0.6, 0.16, color=SYS_COLOR[ORGAN_SYSTEM[o]], clip_on=False, lw=0))
    ax.text(k - 0.35, -0.9, "×n", ha="left", va="bottom", fontsize=8, color=INK2, fontweight="bold")
    ax.text(k + 1.5, -0.9, "ICD-10 categories", ha="left", va="bottom", fontsize=8, color=INK2, fontweight="bold")

    # legend
    ax.scatter([], [], s=200, color=MUTED, label="primary organ (size = share of the disease's ICD-10 categories)")
    ax.scatter([], [], s=50, facecolors="none", edgecolors=MUTED, linewidths=1.1, label="secondary organ")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.005), ncol=2, frameon=False, fontsize=8, labelcolor=INK2, handletextpad=0.4,
              columnspacing=2.0)

    fig.text(0.012, 0.985, "Which organ is each disease expected to affect?", fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.958, f"{n} representative diseases (covering {total} of the 191 evaluated), grouped by organ system. Near-duplicate diseases (same primary organs, overlapping ICD-10 codes) "
             "are merged; each row is the most typical of its group and '×n' says how many diseases it stands for.\n"
             "Dot colour = the organ's system. An organ is primary when ≥25% of the disease's organ-bearing ICD-10 categories point to it.\n"
             "* = medium confidence: the Atlas chapter cannot corroborate the organ (infections, neoplasms and endocrine/metabolic diseases span many organs; depression is filed under 'Endocrine'), so it rests on the ICD-10 block table alone.",
             fontsize=8, color=INK2, va="top")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figures", f"disease_primary_organs.{ext}"), dpi=200 if ext == "png" else None,
                    facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote figures/disease_primary_organs.png", n, "diseases,", k, "organs")


if __name__ == "__main__":
    main()
