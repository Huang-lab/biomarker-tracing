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
from evaluate_organ_hits import ATLASES, ATLAS_LABEL, KMAX, NPERM, OUT, SURFACE, INK, INK2, MUTED, HAIR, BLUE  # noqa: E402

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


def best_rank_table(d, atlas):
    """Per disease: best rank of its primary+secondary organ cell types, the same for primary only, and the chance rank
    (median over the other diseases of the best rank that disease's own-organ set gets in their rankings)."""
    prim, _, both = masks(d)
    r_both, _ = E.first_hit_rank(d["X"], both)
    r_prim, _ = E.first_hit_rank(d["X"], prim)
    single = np.array([len(x) == 1 for x in d["mp"].organ_dict])
    return pd.DataFrame({"disease": d["mp"].Disease, "organ": [next(iter(x)) if len(x) == 1 else ";".join(x) for x in d["mp"].organ_dict],
                         "n_own_cell_types": both.sum(1), "n_primary_cell_types": prim.sum(1), "best_own_rank": r_both, "primary_only_rank": r_prim,
                         "chance_rank": E.empirical_chance_rank(d["X"], both), "confidence": d["mp"].confidence, "single_organ": single,
                         "has_secondary": both.sum(1) > prim.sum(1)})


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
        d = E.prepare(a); res[a] = best_rank_table(d, a); ncols[a] = d["X"].shape[1]
        res[a].to_csv(os.path.join(OUT, f"{a}_best_rank_primary_secondary.tsv"), sep="\t", index=False)
        r = res[a]; sg = r[r.single_organ]
        print(a, "beat chance rank (primary+secondary):", round((r.best_own_rank < r.chance_rank).mean(), 3),
              "| single-organ diseases:", len(sg), "| best rank improved by secondary organs:", int((sg.primary_only_rank > sg.best_own_rank).sum()))
    sub = ("One dot per single-primary-organ disease (rows = its primary organ), as in Figure 3, but the best rank is taken over its primary + secondary organs' cell types.\n"
           "Hairline = the improvement over the primary-only rank. Black tick = chance for the widened set (median best rank across the other diseases). "
           "Blue = better than chance.")
    E.fig_best_rank({a: res[a][res[a].single_organ] for a in ATLASES}, ncols, name="fig5_best_rank_primary_secondary",
                    title="Per disease: where does its best own-organ cell type rank? (primary + secondary organs)", subtitle=sub)

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
