"""
Per disease and organ: does the disease rank its own organ's cell types higher than diseases that are not about that organ do?

For every (disease, organ in the disease's own set) pair - primary organs from disease_organ_map.tsv, plus secondary organs
- cell types are ranked by specificity_z within the disease and the best rank of that organ's cell types is taken.

  * chance rank (black tick in the figures): the MEDIAN of the same best rank over all diseases that do NOT have that organ
    among their primary or secondary organs. It says where the organ typically lands for a disease that is not about it.
  * p-value: (1 + number of those other diseases that rank the organ at least as well) / (1 + number of other diseases).
    The comparison diseases are the null distribution, so organs that rank high for every disease (generic signal) are
    discounted automatically. Ties count against the disease.
  * q-value: Benjamini-Hochberg across every disease-organ pair of the atlas (primary and secondary together).
  * near-duplicate group: diseases with the same primary organs and overlapping ICD-10 codes (see
    plot_disease_organ_examples.near_duplicate_groups) are not independent; groups are counted once in the summary.

No statement is made about "all diseases": the output is a table with one row per disease-organ pair, and the number of pairs
that pass is reported next to the number expected by chance (p < 0.05 on a null distribution is expected for 5% of pairs).

Outputs go to organ_mapping/figures/evaluation/. Usage (repository root):
    python organ_mapping/evaluate_organ_hits.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_organ_heatmaps import load, ORGAN_ORDER, SURFACE, INK, INK2, MUTED, HAIR  # noqa: E402
from plot_disease_organ_examples import near_duplicate_groups  # noqa: E402

BLUE = "#2a78d6"          # categorical slot 1: "own organ" / observed
GRAY = "#c9c8c2"          # other cell types / null
ATLASES = ["hpa700", "pp229"]
ATLAS_LABEL = {"hpa700": "HPA v25.1 (700 cell types)", "pp229": "pp229 (229 cell types)"}
ALPHA = 0.05              # q-value threshold for "significant"
OUT = os.path.join(HERE, "figures", "evaluation")
ROOT = "data"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": HAIR, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def prepare(atlas):
    mp, Z, col_tissue, col_organ = load(atlas, ROOT)
    own = np.array([[o in d for o in col_organ] for d in mp.organ_dict])
    sec = np.array([[o in {x for x in s.split(";") if x} for o in col_organ] for s in mp.organs_secondary])
    keep = own.any(1) & ~own.all(1)                      # organ has cell types in this atlas
    return dict(mp=mp[keep].reset_index(drop=True), Z=Z[keep], X=Z.to_numpy()[keep], own=own[keep], own_any=(own | sec)[keep],
                col_organ=col_organ)


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def organ_pair_table(d):
    """One row per (disease, organ in its own set). See the module docstring for best_rank, chance_rank, p, q."""
    mp, X, col_organ = d["mp"], d["X"], d["col_organ"]
    prim, both = d["own"], d["own_any"]
    R = (-X).argsort(1).argsort(1) + 1                       # rank of every cell type within each disease (1 = highest specificity_z)
    _, gid, *_ = near_duplicate_groups(mp)
    gsize = np.bincount(gid)[gid]
    names = np.array(d["Z"].columns)
    rows, null_u = [], {}
    for o in sorted(set(col_organ)):
        cols = col_organ == o
        best = R[:, cols].min(1)
        in_own = both[:, cols].any(1)
        if not in_own.any(): continue
        null = np.sort(best[~in_own]); chance = np.median(null)
        j = np.where(~in_own)[0]                              # leave-one-out percentile of every comparison disease, same scale as p
        null_u[o] = pd.DataFrame(dict(gid=gid[j], u=np.searchsorted(null, best[j], side="right") / len(null)))
        for i in np.where(in_own)[0]:
            rows.append(dict(disease=mp.Disease[i], organ=o, role="primary" if prim[i, cols].any() else "secondary",
                             n_cell_types=int(cols.sum()), best_rank=int(best[i]), best_cell_type=names[cols][R[i, cols].argmin()],
                             chance_rank=chance, beats_chance=bool(best[i] < chance), n_compared=len(null),
                             p=(1 + np.searchsorted(null, best[i], side="right")) / (1 + len(null)),
                             confidence=mp.confidence[i], dup_group=int(gid[i]), dup_group_size=int(gsize[i]),
                             n_primary_organs=int(prim[i].any() and len(mp.organ_dict[i]))))
    t = pd.DataFrame(rows)
    t["q"] = bh(t.p.to_numpy()); t["significant"] = t.q < ALPHA
    t["nominal"] = t.p < 0.05
    return t, null_u


def organ_table(t, null_u, n_perm=20000, seed=0):
    """Organ-level combination of the pair p-values (exploratory). For one organ and role, the statistic is the mean p of the diseases
    that have the organ, with near-duplicate groups counted once (p averaged within a group). Null: the same statistic for the same
    number of near-duplicate groups drawn at random from the diseases WITHOUT the organ (their leave-one-out percentiles u). Small = the organ's
    diseases rank it higher than random other groups of diseases do. BH across organ x role rows."""
    rng = np.random.default_rng(seed); rows = []
    for (o, role), g in t.groupby(["organ", "role"]):
        gp = g.groupby("dup_group").p.mean(); G = len(gp)
        pool = null_u[o].groupby("gid").u.mean().to_numpy()
        if G > len(pool): continue
        draw = np.array([rng.choice(pool, G, replace=False).mean() for _ in range(n_perm)])
        rows.append(dict(organ=o, role=role, n_pairs=len(g), n_groups=G, mean_p=gp.mean(), organ_p=(1 + (draw <= gp.mean()).sum()) / (n_perm + 1),
                         n_nominal=int(g.nominal.sum()), n_beat_chance=int(g.beats_chance.sum())))
    ot = pd.DataFrame(rows); ot["organ_q"] = bh(ot.organ_p.to_numpy())
    return ot.sort_values("organ_p").reset_index(drop=True)


def summarize(atlas, t):
    n = len(t); tp = t[t.role == "primary"]
    print(f"\n{atlas}: {n} disease-organ pairs ({len(tp)} primary, {n - len(tp)} secondary) over {t.disease.nunique()} diseases")
    print(f"  p < 0.05: {(t.p < 0.05).sum()} pairs (expected by chance: {0.05 * n:.0f})")
    for lab, s in (("all pairs", t), ("primary pairs", tp)):
        sig = s[s.significant]
        print(f"  q < {ALPHA} ({lab}): {len(sig)} pairs, {sig.disease.nunique()} diseases, {sig.dup_group.nunique()} near-duplicate groups, "
              f"organs: {', '.join(sorted(sig.organ.unique()))}")


# ---- figures -----------------------------------------------------------------------------------
def clean(ax, grid_y=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(HAIR); ax.spines["bottom"].set_color(HAIR); ax.tick_params(length=2)
    if grid_y: ax.yaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True)



def mark_nominal(ax, x, y):
    """black ring around the dots of pairs with p < 0.05 (uncorrected)."""
    if len(x): ax.scatter(x, y, s=78, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=5)


def fmt_q(q):
    return "<.001" if q < 0.001 else f"{q:.3f}"


def fig_best_rank(res, otab, n_cols):
    """res[atlas]: pair table restricted to primary pairs of diseases with exactly one primary organ; otab[atlas]: organ_table of those rows."""
    organs = [o for o in ORGAN_ORDER if any((r.organ == o).any() for r in res.values())]
    counts = {o: max(int((res[a].organ == o).sum()) for a in ATLASES) for o in organs}
    ratio = {o: np.nanmedian(np.concatenate([np.log(res[a].loc[res[a].organ == o, "best_rank"].to_numpy() / res[a].loc[res[a].organ == o, "chance_rank"].to_numpy())
                                             for a in ATLASES if (res[a].organ == o).any()])) for o in organs}
    organs = sorted(organs, key=lambda o: ratio[o])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 0.32 * len(organs) + 2.4), sharey=True)
    rng = np.random.default_rng(1)
    for ax, atlas in zip(axes, ATLASES):
        r = res[atlas]; n = n_cols[atlas]; oq = otab[atlas].set_index("organ").organ_q
        for i, o in enumerate(organs):
            s = r[r.organ == o]
            if s.empty: ax.text(1.2, i, "no cell types in this atlas", fontsize=7, color=MUTED, va="center", style="italic"); continue
            v = s.best_rank.to_numpy(); win = s.beats_chance.to_numpy(); nom = s.nominal.to_numpy()
            ax.plot([s.chance_rank.iloc[0]] * 2, [i - 0.36, i + 0.36], color=INK, lw=1.6, zorder=2, solid_capstyle="butt")
            y = i + rng.uniform(-0.2, 0.2, len(v))
            ax.scatter(v[~win], y[~win], s=22, color=GRAY, lw=0.8, edgecolors=SURFACE, zorder=3)
            ax.scatter(v[win], y[win], s=26, color=BLUE, lw=0.8, edgecolors=SURFACE, zorder=4)
            mark_nominal(ax, v[nom], y[nom])
            ax.text(n * 1.55, i, f"{win.sum()}/{len(v)}", fontsize=7.5, va="center", ha="center", color=INK if win.mean() > 0.5 else MUTED)
            q = oq.get(o, np.nan); hit = q < 0.05
            ax.text(n * 3.3, i, fmt_q(q) if q == q else "–", fontsize=7.5, va="center", ha="center", color=INK if hit else MUTED, fontweight="bold" if hit else "normal")
        ax.set_title(f"{ATLAS_LABEL[atlas]}  ·  {len(r)} single-organ diseases", fontsize=8.8, color=INK, loc="left")
        ax.set_xscale("log"); ax.set_xlim(0.8, n * 4.2)
        ticks = [t for t in (1, 3, 10, 30, 100, 300) if t <= n]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        clean(ax, grid_y=False); ax.xaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel(f"rank of the best own-organ cell type (of {n}; left = better)")
        ax.text(n * 1.55, -1, "beat\nchance", fontsize=7, color=INK2, ha="center", va="center")
        ax.text(n * 3.3, -1, "organ q", fontsize=7, color=INK, ha="center", va="center", fontweight="bold")
    axes[0].set_yticks(range(len(organs))); axes[0].set_yticklabels([f"{o.replace('.', ' ')}  (n={counts[o]})" for o in organs])
    axes[0].set_ylim(len(organs) - 0.5, -1.6)
    axes[0].scatter([], [], s=26, color=BLUE, label="ranks its organ better than the chance rank"); axes[0].scatter([], [], s=22, color=GRAY, label="does not")
    axes[0].scatter([], [], s=78, facecolors="none", edgecolors=INK, lw=1.1, label="p < 0.05 for this disease (uncorrected)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.55, 0.0))
    h = fig.get_figheight()
    fig.text(0.01, 0.995, "Per disease: where does its best own-organ cell type rank?", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.995 - 0.33 / h, "One dot per disease with a single primary organ. Black tick = the organ's chance rank: the MEDIAN rank of that organ's best cell type across all diseases that do not have it as an organ.\n"
             "Blue = this disease ranks its own organ better (further left) than that median. Ring = p < 0.05 for that disease alone: p is the share of the other diseases that rank the organ at least as high.\n"
             "No single disease passes a multiple-testing correction (see text). Organ q = organ-level test: do this organ's diseases (near-duplicates counted once) rank it better than random groups\n"
             "of other diseases do? Benjamini-Hochberg across organs; bold = q < 0.05. Organs sorted by median distance from chance.",
             fontsize=7.4, color=INK2, va="top")
    fig.subplots_adjust(left=0.17, right=0.98, top=1 - 1.95 / h, bottom=0.95 / h, wspace=0.08)
    save(fig, "fig3_best_rank_per_disease")


def fig_pairs(res, otab, n_cols):
    """Figure 5: every (disease, organ) pair, primary (upper dots) and secondary (lower dots)."""
    organs = sorted({o for r in res.values() for o in r.organ}, key=lambda o: ORGAN_ORDER.index(o))
    def ratio(o):
        v = [np.log(g.best_rank / g.chance_rank) for r in res.values() for g in [r[(r.organ == o) & (r.role == "primary")]] if len(g)]
        return np.nanmedian(np.concatenate(v)) if v else 9.0              # organs with only secondary rows go last
    organs = sorted(organs, key=ratio)
    cnt = {o: (max(int(((r.organ == o) & (r.role == "primary")).sum()) for r in res.values()),
               max(int(((r.organ == o) & (r.role == "secondary")).sum()) for r in res.values())) for o in organs}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 0.34 * len(organs) + 2.9), sharey=True)
    rng = np.random.default_rng(1)
    cx = {("primary", "n"): 1.07, ("primary", "q"): 1.19, ("secondary", "n"): 1.33, ("secondary", "q"): 1.45}   # axes fraction
    for ax, a in zip(axes, ATLASES):
        r, n = res[a], n_cols[a]; oq = otab[a].set_index(["organ", "role"]).organ_q; tr = ax.get_yaxis_transform()
        for i, o in enumerate(organs):
            g = r[r.organ == o]
            if g.empty: ax.text(1.2, i, "no cell types in this atlas", fontsize=7, color=MUTED, va="center", style="italic"); continue
            ax.plot([g.chance_rank.iloc[0]] * 2, [i - 0.4, i + 0.4], color=INK, lw=1.6, zorder=2, solid_capstyle="butt")
            for role, dy in (("primary", -0.13), ("secondary", 0.17)):
                q = g[g.role == role]
                if q.empty: continue
                y = i + dy + rng.uniform(-0.09, 0.09, len(q)); win = q.beats_chance.to_numpy(); v = q.best_rank.to_numpy()
                if role == "primary":
                    ax.scatter(v[~win], y[~win], s=22, color=GRAY, lw=0.8, edgecolors=SURFACE, zorder=3)
                    ax.scatter(v[win], y[win], s=26, color=BLUE, lw=0.8, edgecolors=SURFACE, zorder=4)
                else:
                    ax.scatter(v[~win], y[~win], s=22, facecolors=SURFACE, edgecolors=MUTED, lw=1.1, zorder=3)
                    ax.scatter(v[win], y[win], s=26, facecolors=SURFACE, edgecolors=BLUE, lw=1.6, zorder=4)
                mark_nominal(ax, v[q.nominal.to_numpy()], y[q.nominal.to_numpy()])
            for role in ("primary", "secondary"):
                q = g[g.role == role]
                ax.text(cx[(role, "n")], i, f"{int(q.beats_chance.sum())}/{len(q)}" if len(q) else "–", fontsize=7.5, va="center", ha="center",
                        color=INK if len(q) and q.beats_chance.mean() > 0.5 else MUTED, transform=tr)
                qv = oq.get((o, role), np.nan); hit = qv < 0.05
                ax.text(cx[(role, "q")], i, fmt_q(qv) if qv == qv else "–", fontsize=7.5, va="center", ha="center",
                        color=INK if hit else MUTED, fontweight="bold" if hit else "normal", transform=tr)
        ax.set_title(f"{ATLAS_LABEL[a]}  ·  {int((r.role == 'primary').sum())} primary + {int((r.role == 'secondary').sum())} secondary pairs", fontsize=8.6, color=INK, loc="left")
        ax.set_xscale("log"); ax.set_xlim(0.8, n * 1.3)
        ticks = [t for t in (1, 3, 10, 30, 100, 300) if t <= n]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        clean(ax, grid_y=False); ax.xaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel(f"rank of the organ's best cell type (of {n}; left = better)", fontsize=8.5)
        for role, lab in (("primary", "primary"), ("secondary", "secondary")):
            ax.text((cx[(role, "n")] + cx[(role, "q")]) / 2, -1.25, lab, fontsize=7.5, color=INK2, ha="center", va="center", fontweight="bold", transform=tr)
            ax.text(cx[(role, "n")], -0.55, "beat\nchance", fontsize=6.5, color=INK2, ha="center", va="center", transform=tr)
            ax.text(cx[(role, "q")], -0.55, "organ\nq", fontsize=6.5, color=INK2, ha="center", va="center", transform=tr)
    axes[0].set_yticks(range(len(organs)))
    axes[0].set_yticklabels([f"{o.replace('.', ' ')}  ({cnt[o][0]}+{cnt[o][1]})" for o in organs])
    axes[0].set_ylim(len(organs) - 0.5, -2.0)
    axes[0].scatter([], [], s=26, color=BLUE, label="primary organ, beats chance"); axes[0].scatter([], [], s=22, color=GRAY, label="primary organ, does not")
    axes[0].scatter([], [], s=26, facecolors=SURFACE, edgecolors=BLUE, lw=1.6, label="secondary organ, beats chance")
    axes[0].scatter([], [], s=22, facecolors=SURFACE, edgecolors=MUTED, lw=1.1, label="secondary organ, does not")
    axes[0].scatter([], [], s=78, facecolors="none", edgecolors=INK, lw=1.1, label="p < 0.05 for this pair (uncorrected)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=7.6, labelcolor=INK2, bbox_to_anchor=(0.55, 0.0))
    h = fig.get_figheight()
    fig.text(0.01, 0.995, "Per disease and organ: where does that organ's best cell type rank?", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.995 - 0.33 / h, "One dot per (disease, organ in its own set): filled = primary organ (upper dots), open ring = secondary organ (lower dots). A disease with several organs has several dots.\n"
             "Black tick = the organ's chance rank: the MEDIAN rank of that organ's best cell type across all diseases that do not have it as an organ (same definition as Figure 3).\n"
             "Blue = this disease ranks the organ better (further left) than that median. Ring = p < 0.05 for that pair alone (the share of other diseases ranking the organ at least as high); none survives correction.\n"
             "Organ q = organ-level test per role (near-duplicate diseases counted once; Benjamini-Hochberg across organs and roles); bold = q < 0.05. Row label: organ (primary pairs + secondary pairs).",
             fontsize=7.4, color=INK2, va="top")
    fig.subplots_adjust(left=0.14, right=0.83, top=1 - 2.0 / h, bottom=0.85 / h, wspace=0.62)
    save(fig, "fig5_best_rank_primary_secondary")


def fig_examples(d, picks):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    xmax = max(d["X"][list(d["mp"].Disease).index(n)].max() for n in picks) * 1.08
    for ax, name in zip(axes.ravel(), picks):
        i = list(d["mp"].Disease).index(name)
        x = d["X"][i]; o = np.argsort(-x)[:15]; own = d["own"][i][o]
        organs = " / ".join(k.replace(".", " ") for k in d["mp"].organ_dict[i])
        ax.barh(range(15), x[o], color=[BLUE if w else GRAY for w in own], height=0.72)
        labels = []
        for r, c in enumerate(d["Z"].columns[o], 1):
            t, ct = c.split("_", 1); ct = " ".join(ct.replace(".", " ").split())
            labels.append(f"{r}. {ct[:26]} · {t.replace('.', ' ')}")
        ax.set_ylim(14.6, -0.6); ax.set_yticks(range(15)); ax.set_yticklabels(labels, fontsize=7)
        for lbl, w in zip(ax.get_yticklabels(), own): lbl.set_color(INK if w else INK2); lbl.set_fontweight("bold" if w else "normal")
        clean(ax, grid_y=False); ax.xaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(0, xmax)
        n_own = int(own.sum())
        ax.set_title(f"{name[:44]}\nown organ: {organs}\n{n_own} of the top 15 in own organ", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("disease-specific z (above a typical disease)", fontsize=8)
    fig.text(0.01, 0.99, "What a hit looks like: the 15 top-ranked cell types of four diseases (HPA atlas)", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.955, "Blue bars (bold labels) = cell types from the disease's own organ; gray = other organs. "
             "Label: cell type · tissue.", fontsize=8, color=INK2, va="top")
    fig.subplots_adjust(left=0.25, right=0.985, top=0.85, bottom=0.07, wspace=1.0, hspace=0.55)
    save(fig, "fig1_examples")



def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {a: prepare(a) for a in ATLASES}
    pairs, organs, nulls = {}, {}, {}
    for a, d in data.items():
        pairs[a], nulls[a] = organ_pair_table(d)
        organs[a] = organ_table(pairs[a], nulls[a])
        pairs[a].sort_values("p").round(5).to_csv(os.path.join(OUT, f"{a}_disease_organ_pairs.tsv"), sep="\t", index=False)
        organs[a].round(5).to_csv(os.path.join(OUT, f"{a}_organ_level_test.tsv"), sep="\t", index=False)
        summarize(a, pairs[a])
        print(organs[a].round(4).to_string(index=False))
    n_cols = {a: data[a]["X"].shape[1] for a in ATLASES}
    single = {a: pairs[a][(pairs[a].role == "primary") & (pairs[a].n_primary_organs == 1)] for a in ATLASES}
    fig_best_rank(single, {a: organ_table(single[a], nulls[a]) for a in ATLASES}, n_cols)
    fig_pairs(pairs, organs, n_cols)
    if len(sys.argv) > 1: picks = sys.argv[1].split("|")
    else: picks = ["Asthma, excluding more controls", "Gastric ulcer", "Heart failure, strict", "Chronic kidney disease"]
    fig_examples(data["hpa700"], [p for p in picks if p in set(data["hpa700"]["mp"].Disease)])


if __name__ == "__main__":
    main()
