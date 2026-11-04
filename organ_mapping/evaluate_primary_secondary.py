"""
Does adding each disease's secondary organs to its "own organ" change the picture?

evaluate_organ_hits.py scores a disease's ranking against its PRIMARY organs only. Here the own-organ set is
widened to primary + secondary organs (disease_organ_map.tsv: organs that fewer than 25% of the disease's
ICD-10 categories point to, plus organs the block table lists as secondary), and the same tests are run:

  * hit rate at depth k (>=1 own-organ cell type among the top k by specificity_z) against a null with the
    organ assignments shuffled across diseases (2000x). Each definition is compared with its OWN null, since a
    wider organ set raises the chance of a hit; the figure therefore plots the excess over the shuffled median;
  * mean-rank AUC of the own-organ cell types against the rest, and the share of the top 10 that is own-organ;
  * "secondary only": for diseases that have secondary organs, do those organs' cell types rank high on their own
    (primary-organ cell types excluded)?

Outputs (organ_mapping/figures/evaluation/): fig4_primary_vs_secondary.png/.pdf, primary_vs_secondary.tsv.
Usage (repository root):
    python organ_mapping/evaluate_primary_secondary.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_organ_hits as E  # noqa: E402
from evaluate_organ_hits import ATLASES, ATLAS_LABEL, KMAX, NPERM, OUT, SURFACE, INK, INK2, MUTED, HAIR, BLUE, GRAY  # noqa: E402

ORANGE = "#eb6834"        # categorical slot 2: primary + secondary
BLUE_BAND, ORANGE_BAND = "#bfd6f3", "#f6cdbb"


def masks(d):
    """own-organ column masks per disease: primary, secondary-only (not primary), primary+secondary."""
    col_organ = d["col_organ"]
    prim = d["own"]
    sec_sets = [set(o for o in s.split(";") if o) - set(p) for s, p in zip(d["mp"].organs_secondary, d["mp"].organ_dict)]
    sec = np.array([[o in s for o in col_organ] for s in sec_sets])
    return prim, sec, prim | sec


def auc_and_share(X, M, k=10, n_perm=2000, seed=0):
    """mean-rank AUC of own columns vs the rest, and top-k own share, with label-shuffling nulls (rows without own/with all dropped)."""
    keep = M.any(1) & ~M.all(1); X, M = X[keep], M[keep]
    n = X.shape[1]; R = np.apply_along_axis(rankdata, 1, X)
    auc = lambda Mm: (((R * Mm).sum(1) - Mm.sum(1) * (Mm.sum(1) + 1) / 2) / (Mm.sum(1) * (n - Mm.sum(1))))
    top = np.argsort(-X, axis=1)[:, :k]
    share = lambda Mm: np.take_along_axis(Mm, top, axis=1).mean()
    rng = np.random.default_rng(seed); obs_a, obs_s = auc(M).mean(), share(M)
    na = np.empty(n_perm); ns = np.empty(n_perm)
    for b in range(n_perm):
        Mp = M[rng.permutation(len(M))]; na[b] = auc(Mp).mean(); ns[b] = share(Mp)
    p = lambda o, nl: (1 + (nl >= o).sum()) / (n_perm + 1)
    return dict(n_diseases=int(keep.sum()), auc=obs_a, auc_null=na.mean(), auc_p=p(obs_a, na),
                top10_share=obs_s, top10_null=ns.mean(), top10_p=p(obs_s, ns))


def organ_pair_table(d):
    """One row per (disease, organ in its own set): the best rank of that organ's cell types in the disease's ranking, and the
    chance rank for that organ (median best rank of the organ over the diseases that do NOT have it in their own set)."""
    prim, sec_only, both = masks(d)
    col_organ, X = d["col_organ"], d["X"]
    R = (-X).argsort(1).argsort(1) + 1                       # rank of every cell type within each disease (1 = highest specificity_z)
    rows = []
    for o in sorted(set(col_organ)):
        cols = col_organ == o
        best = R[:, cols].min(1)
        in_own = both[:, cols].any(1)
        if not in_own.any(): continue
        chance = np.median(best[~in_own])
        for i in np.where(in_own)[0]:
            rows.append(dict(disease=d["mp"].Disease[i], organ=o, role="primary" if prim[i, cols].any() else "secondary",
                             n_cell_types=int(cols.sum()), best_rank=int(best[i]), chance_rank=chance, beats_chance=bool(best[i] < chance),
                             confidence=d["mp"].confidence[i]))
    return pd.DataFrame(rows)


def fig_pairs(res, n_cols):
    sub = {a: res[a] for a in ATLASES}
    organs = sorted({o for r in sub.values() for o in r.organ}, key=lambda o: E.ORGAN_ORDER.index(o))
    def ratio(o):
        v = [np.log(g.best_rank / g.chance_rank) for r in sub.values() for g in [r[(r.organ == o) & (r.role == "primary")]] if len(g)]
        return np.nanmedian(np.concatenate(v)) if v else 9.0              # organs with only secondary rows go last
    organs = sorted(organs, key=ratio)
    cnt = {o: (max(int(((r.organ == o) & (r.role == "primary")).sum()) for r in sub.values()),
               max(int(((r.organ == o) & (r.role == "secondary")).sum()) for r in sub.values())) for o in organs}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.34 * len(organs) + 2.3), sharey=True)
    rng = np.random.default_rng(1)
    for ax, a in zip(axes, ATLASES):
        r, n = sub[a], n_cols[a]
        for i, o in enumerate(organs):
            g = r[r.organ == o]
            if g.empty: ax.text(1.2, i, "no cell types in this atlas", fontsize=7, color=MUTED, va="center", style="italic"); continue
            ch = g.chance_rank.iloc[0]
            ax.plot([ch, ch], [i - 0.4, i + 0.4], color=INK, lw=1.6, zorder=2, solid_capstyle="butt")
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
            for role, x in (("primary", n * 1.55), ("secondary", n * 3.3)):
                q = g[g.role == role]
                ax.text(x, i, f"{int(q.beats_chance.sum())}/{len(q)}" if len(q) else "–", fontsize=7.5, va="center", ha="center",
                        color=INK if len(q) and q.beats_chance.mean() > 0.5 else MUTED)
        wp = r[r.role == "primary"].beats_chance.mean(); ws = r[r.role == "secondary"].beats_chance.mean()
        ax.set_title(f"{ATLAS_LABEL[a]}\nbeat chance: primary pairs {wp:.0%} (n={int((r.role == 'primary').sum())}), "
                     f"secondary pairs {ws:.0%} (n={int((r.role == 'secondary').sum())})", fontsize=8.8, color=INK, loc="left")
        ax.set_xscale("log"); ax.set_xlim(0.8, n * 4.2)
        ticks = [t for t in (1, 3, 10, 30, 100, 300) if t <= n]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        E.clean(ax, grid_y=False); ax.xaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel(f"rank of the organ's best cell type (of {n}; left = better)", fontsize=8.5)
        ax.text(n * 1.55, -0.95, "primary", fontsize=7, color=INK2, ha="center", va="center")
        ax.text(n * 3.3, -0.95, "secondary", fontsize=7, color=INK2, ha="center", va="center")
        ax.text(n * 2.3, -1.45, "beat chance", fontsize=7, color=INK2, ha="center", va="center", fontweight="bold")
    axes[0].set_yticks(range(len(organs)))
    axes[0].set_yticklabels([f"{o.replace('.', ' ')}  ({cnt[o][0]}+{cnt[o][1]})" for o in organs])
    axes[0].set_ylim(len(organs) - 0.5, -1.9)
    axes[0].scatter([], [], s=26, color=BLUE, label="primary organ, beats chance"); axes[0].scatter([], [], s=22, color=GRAY, label="primary organ, does not")
    axes[0].scatter([], [], s=26, facecolors=SURFACE, edgecolors=BLUE, lw=1.6, label="secondary organ, beats chance")
    axes[0].scatter([], [], s=22, facecolors=SURFACE, edgecolors=MUTED, lw=1.1, label="secondary organ, does not")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.55, 0.0))
    h = fig.get_figheight()
    fig.text(0.01, 0.995, "Per disease and organ: where does that organ's best cell type rank?", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.995 - 0.33 / h, "One dot per (disease, organ in its own set): filled = primary organ (upper dots), open ring = secondary organ (lower dots). A disease with several\n"
             "organs has several dots. Black tick = chance for that organ: its median best rank across diseases that do not have it as own organ. Blue = better than chance.\n"
             "Row label: organ (primary pairs + secondary pairs). Any own organ beating chance = any blue dot of the disease. Organs sorted by the primary pairs' distance from chance.",
             fontsize=7.8, color=INK2, va="top")
    fig.subplots_adjust(left=0.17, right=0.985, top=1 - 1.75 / h, bottom=1.0 / h, wspace=0.08)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"fig5_best_rank_primary_secondary.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, curves = [], {}
    for a in ATLASES:
        d = E.prepare(a); prim, sec, both = masks(d)
        n_wider = int((both.sum(1) > prim.sum(1)).sum())
        print(f"{a}: {len(prim)} diseases; {n_wider} get extra cell types from secondary organs "
              f"(own columns per disease: median {np.median(prim.sum(1)):.0f} -> {np.median(both.sum(1)):.0f})")
        for kind, M in [("primary", prim), ("primary+secondary", both), ("secondary only", sec)]:
            ok = M.any(1) & ~M.all(1)
            ks, obs, null, p = E.recall_curve(d["X"][ok], M[ok]); curves[(a, kind)] = (ks, obs, null)
            st = auc_and_share(d["X"], M)
            for k in (1, 5, 10, 20, 50):
                rows.append(dict(atlas=a, own_organ_set=kind, n_diseases=int(ok.sum()), measure=f"≥1 in top {k}", observed=obs[k - 1],
                                 null_median=np.median(null[:, k - 1]), p=p[k]))
            rows.append(dict(atlas=a, own_organ_set=kind, n_diseases=st["n_diseases"], measure="mean-rank AUC", observed=st["auc"],
                             null_median=st["auc_null"], p=st["auc_p"]))
            rows.append(dict(atlas=a, own_organ_set=kind, n_diseases=st["n_diseases"], measure="top-10 own share", observed=st["top10_share"],
                             null_median=st["top10_null"], p=st["top10_p"]))
    t = pd.DataFrame(rows); t.round(4).to_csv(os.path.join(OUT, "primary_vs_secondary.tsv"), sep="\t", index=False)
    print(t.round(3).to_string(index=False))

    res, ncols = {}, {}
    for a in ATLASES:
        d = E.prepare(a); res[a] = organ_pair_table(d); ncols[a] = d["X"].shape[1]
        res[a].round(3).to_csv(os.path.join(OUT, f"{a}_organ_pair_ranks.tsv"), sep="\t", index=False)
        g = res[a].groupby("role").beats_chance.agg(["sum", "size", "mean"])
        print(a, "pairs beating chance:", g.round(3).to_dict("index"))
    fig_pairs(res, ncols)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    fmt = lambda q: "p<0.001" if q < 0.001 else f"p={q:.3f}"
    for ax, a in zip(axes, ATLASES):
        for kind, col, band in (("primary+secondary", ORANGE, ORANGE_BAND), ("primary", BLUE, BLUE_BAND)):
            ks, obs, null = curves[(a, kind)]
            lo, mid, hi = np.percentile(null, [2.5, 50, 97.5], axis=0)
            ax.plot(ks, (hi - mid) * 100, color=col, lw=1.0, ls=(0, (2, 2)), alpha=0.9)
            ax.plot(ks, (obs - mid) * 100, color=col, lw=2.2, label=f"{kind} organs")
        ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (3, 2)))
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(HAIR); ax.spines["bottom"].set_color(HAIR); ax.tick_params(length=2)
        ax.yaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(1, KMAX)
        ax.set_title(ATLAS_LABEL[a], fontsize=9.5, color=INK, loc="left"); ax.set_xlabel("k (depth of the disease's cell-type ranking)")
        g = t[t.atlas == a]
        lines = ["                  top 1       top 10"]
        for kind in ("primary", "primary+secondary"):
            r = g[g.own_organ_set == kind].set_index("measure")
            lines.append(f"{kind:<18}" + "  ".join(f"{r.loc[m, 'observed'] * 100:3.0f}% ({fmt(r.loc[m, 'p'])})" for m in ("≥1 in top 1", "≥1 in top 10")))
        ax.text(0.98, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=7.3, color=INK, family="monospace")
    axes[0].set_ylabel("Percentage points above the shuffled-organ median\n(diseases with ≥1 own-organ cell type in the top k)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
    parts = []
    for a in ATLASES:
        so = t[(t.atlas == a) & (t.own_organ_set == "secondary only")].set_index("measure")
        parts.append(f"{a} ({int(so.n_diseases.iloc[0])} diseases): top 1 {so.loc['≥1 in top 1', 'observed'] * 100:.0f}% vs "
                     f"{so.loc['≥1 in top 1', 'null_median'] * 100:.0f}% shuffled, top 10 {so.loc['≥1 in top 10', 'observed'] * 100:.0f}% vs "
                     f"{so.loc['≥1 in top 10', 'null_median'] * 100:.0f}%")
    fig.text(0.01, 0.015, "Secondary organs on their own (primary-organ cell types excluded; only diseases that have secondary organs), observed vs shuffled:\n"
             + ";  ".join(parts) + "  →  no excess over chance.", fontsize=7.8, color=INK2, va="bottom")
    fig.text(0.01, 0.985, "Does adding secondary organs help? Excess hit rate over the shuffled-organ baseline", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.935, "Lines: observed minus the median of its own shuffled-organ null (each definition has its own null, because a wider organ set raises the chance of a hit).\n"
             "Dashed: upper 95% limit of that null (same colour). A line above its dashed limit = more own-organ hits than chance.\n"
             "Boxes: share of diseases with ≥1 own-organ cell type in the top k, and permutation p.", fontsize=7.8, color=INK2, va="top")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.78, bottom=0.17, wspace=0.06)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig4_primary_vs_secondary.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
