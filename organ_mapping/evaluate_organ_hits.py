"""
How often does a disease's cell-type ranking point at the disease's own organ?

Complements plot_organ_heatmaps.py (which reports a mean-rank AUC and a fixed top-10 share) with
evaluations that do not depend on one arbitrary cut-off:

  1. Hit rate at every depth k = 1..50 ("recall@k"): the share of diseases with at least one cell type of
     their own organ among their k highest-ranked cell types (ranked by specificity_z), against a null in
     which organ assignments are shuffled across diseases (keeps each disease's ranking and each organ's
     number of cell types). Done twice: counting any own-organ cell type, and counting only the organ's
     defining (parenchymal) cell types - immune, endothelial, stromal and blood cells sit in every tissue,
     so a "kidney macrophage" hit says little about the kidney.
  2. Per disease, how surprising is the rank of its best own-organ cell type? If the organ has m of the n
     cell types and its best one is at rank r, p = P(at least one of m random cell types in the top r)
     (hypergeometric). This adjusts for organs with many cell types (brain: 84 in hpa700) vs few.
  3. Significance-based: of the cell types the pipeline calls significant (max-T p <= 0.05) or
     disease-specific (also specificity FDR <= 0.05), what share are in the disease's own organ?
  4. Worked examples: the top 15 cell types of four diseases, own organ highlighted.

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
from scipy.stats import hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_organ_heatmaps import load, lineage_of, ORGAN_ORDER, SURFACE, INK, INK2, MUTED, HAIR  # noqa: E402

BLUE = "#2a78d6"          # categorical slot 1: "own organ" / observed
GRAY = "#c9c8c2"          # other cell types / null
BAND = "#e1e0d9"
PARENCHYMAL = {"epithelial/other", "neural", "muscle"}
ATLASES = ["hpa700", "pp229"]
ATLAS_LABEL = {"hpa700": "HPA v25.1 (700 cell types)", "pp229": "pp229 (229 cell types)"}
KMAX, NPERM = 50, 2000
OUT = os.path.join(HERE, "figures", "evaluation")
ROOT = "data"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": HAIR, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def prepare(atlas):
    mp, Z, col_tissue, col_organ = load(atlas, ROOT)
    lineage = np.array([lineage_of(c.split("_", 1)[1]) for c in Z.columns])
    own = np.array([[o in d for o in col_organ] for d in mp.organ_dict])
    keep = own.any(1) & ~own.all(1)                      # organ has cell types in this atlas
    return dict(mp=mp[keep].reset_index(drop=True), Z=Z[keep], X=Z.to_numpy()[keep], own=own[keep],
                col_organ=col_organ, lineage=lineage, par=np.isin(lineage, list(PARENCHYMAL)))


def first_hit_rank(X, M):
    """1-based rank of the best-ranked True column of M in each row of X (descending); inf if none."""
    order = np.argsort(-X, axis=1)
    hit = np.take_along_axis(M, order, axis=1)
    r = hit.argmax(1).astype(float) + 1
    r[~hit.any(1)] = np.inf
    return r, order


def recall_curve(X, M, seed=0):
    r, order = first_hit_rank(X, M)
    ks = np.arange(1, KMAX + 1)
    obs = (r[:, None] <= ks).mean(0)
    rng = np.random.default_rng(seed); null = np.empty((NPERM, KMAX))
    for b in range(NPERM):
        hit = np.take_along_axis(M[rng.permutation(len(M))], order[:, :KMAX], axis=1)
        null[b] = (np.cumsum(hit, 1) > 0).mean(0)
    p = {k: (1 + (null[:, k - 1] >= obs[k - 1]).sum()) / (NPERM + 1) for k in (1, 5, 10, 20, 50)}
    return ks, obs, null, p


def best_rank_p(X, M):
    r, _ = first_hit_rank(X, M)
    n, m = M.shape[1], M.sum(1)
    p = np.array([hypergeom.sf(0, n, mi, ri) if np.isfinite(ri) and mi > 0 else np.nan for mi, ri in zip(m, r)])
    return r, m, p


def significance_table(atlas, d):
    h = pd.read_csv(os.path.join(ROOT, f"results_ppa_prevalent_{atlas}", "ppa_prevalent_sumstats", "cross_disease_summary",
                                 "disease_specific_hits.tsv"), sep="\t")
    stem2dis = dict(zip(d["mp"].stem, d["mp"].Disease))
    h = h[h.disease.isin(stem2dis)].assign(Disease=lambda x: x.disease.map(stem2dis))
    cols = list(d["Z"].columns); col_idx = {c: i for i, c in enumerate(cols)}
    own = dict(zip(d["mp"].Disease, d["own"])); par = d["par"]
    rows = []
    for label, sel in [("significant (max-T p ≤ 0.05)", h.sig_within_disease.astype(bool)),
                       ("disease-specific (also specificity FDR ≤ 0.05)", h.hit_class == "disease_specific")]:
        s = h[sel]
        idx = s.cell_tissue.map(col_idx).to_numpy()
        in_own = np.array([own[dz][i] for dz, i in zip(s.Disease, idx)])
        in_own_par = np.array([own[dz][i] and par[i] for dz, i in zip(s.Disease, idx)])
        chance = np.mean([own[dz].mean() for dz in s.Disease]) if len(s) else np.nan
        rows.append(dict(atlas=atlas, hit_set=label, n_hits=len(s), n_diseases_with_hits=s.Disease.nunique(),
                         share_in_own_organ=in_own.mean(), chance=chance, fold=in_own.mean() / chance,
                         share_in_own_organ_parenchymal=in_own_par.mean(),
                         diseases_with_own_organ_hit=pd.Series(in_own).groupby(s.Disease.to_numpy()).any().sum()))
    return pd.DataFrame(rows)


# ---- figures -----------------------------------------------------------------------------------
def clean(ax, grid_y=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(HAIR); ax.spines["bottom"].set_color(HAIR); ax.tick_params(length=2)
    if grid_y: ax.yaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True)


def fig_recall(curves):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True, sharey=True)
    for j, atlas in enumerate(ATLASES):
        for i, kind in enumerate(["all", "parenchymal"]):
            ax = axes[i, j]; ks, obs, null, p = curves[(atlas, kind)]
            lo, mid, hi = np.percentile(null, [2.5, 50, 97.5], axis=0)
            ax.fill_between(ks, lo, hi, color=BAND, lw=0)
            ax.plot(ks, mid, color=MUTED, lw=1.2, ls=(0, (3, 2)))
            ax.plot(ks, obs, color=BLUE, lw=2)
            fmt = lambda q: "p<0.001" if q < 0.001 else f"p={q:.3f}"
            lines = ["         observed  shuffled"]
            for k in (1, 10, 50):
                if k > 1: ax.axvline(k, color=HAIR, lw=0.8, zorder=0)
                ax.plot(k, obs[k - 1], "o", ms=6, color=BLUE, mec=SURFACE, mew=1.5, zorder=5, clip_on=False)
                lines.append(f"top {k:<3}  {obs[k-1]:>6.0%}  {mid[k-1]:>7.0%}   {fmt(p[k])}")
            ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=7.5, color=INK, va="top", family="monospace")
            clean(ax); ax.set_ylim(0, 1); ax.set_xlim(1, KMAX)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            if i == 0: ax.set_title(ATLAS_LABEL[atlas], fontsize=9.5, color=INK, loc="left")
            if j == 0: ax.set_ylabel(("Any own-organ cell type" if kind == "all" else "Organ-defining cell types only\n(no immune, endothelial, stromal, blood)")
                                     + "\n% of diseases with ≥1 in top k")
            if i == 1: ax.set_xlabel("k (depth of the disease's cell-type ranking)")
    fig.text(0.01, 0.985, "How deep do you have to look to find the disease's own organ?", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.950, "Blue line: share of diseases with at least one own-organ cell type among their top k (ranked by disease-specific z).\n"
             f"Gray dashed line and band: the same with organ assignments shuffled across diseases (median, 95% of {NPERM} shuffles). "
             "p = share of shuffles at least as high.", fontsize=8, color=INK2, va="top")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.87, bottom=0.08, hspace=0.12, wspace=0.06)
    save(fig, "fig2_hit_rate_by_depth")


def chance_median_rank(n, m):
    """Median rank of the best of m cell types placed at random among n (smallest r with P(min <= r) >= 0.5)."""
    r = np.arange(1, n + 1)
    return int(r[np.searchsorted(hypergeom.sf(0, n, m, r), 0.5)])


def empirical_chance_rank(X, M):
    """For each disease: median, over the OTHER diseases, of the best rank its own-organ cell set gets in their rankings.
    Unlike chance_median_rank this keeps the correlation between cell types of the same tissue (they rise and fall
    together, so the best of m behaves like fewer than m independent draws)."""
    order = np.argsort(-X, axis=1)
    out = []
    for i in range(len(M)):
        hit = M[i][order]                                    # this organ's cells in every disease's ranking
        r = hit.argmax(1) + 1.0
        out.append(np.median(np.delete(r, i)))
    return np.array(out)


def fig_best_rank(res, n_cols):
    organs = [o for o in ORGAN_ORDER if any((r.organ == o).any() for r in res.values())]
    counts = {o: max(int((res[a].organ == o).sum()) for a in ATLASES) for o in organs}
    ratio = {o: np.nanmedian(np.concatenate([np.log(res[a].loc[res[a].organ == o, "best_own_rank"].to_numpy() / res[a].loc[res[a].organ == o, "chance_rank"].to_numpy())
                                             for a in ATLASES if (res[a].organ == o).any()])) for o in organs}
    organs = sorted(organs, key=lambda o: ratio[o])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.32 * len(organs) + 2.0), sharey=True)
    rng = np.random.default_rng(1)
    for ax, atlas in zip(axes, ATLASES):
        r = res[atlas]; n = n_cols[atlas]
        for i, o in enumerate(organs):
            s = r[r.organ == o]
            if s.empty: ax.text(1.2, i, "no cell types in this atlas", fontsize=7, color=MUTED, va="center", style="italic"); continue
            ch = s.chance_rank.median(); v = s.best_own_rank.to_numpy(); win = v < s.chance_rank.to_numpy()
            ax.plot([ch, ch], [i - 0.36, i + 0.36], color=INK, lw=1.6, zorder=2, solid_capstyle="butt")
            y = i + rng.uniform(-0.2, 0.2, len(v))
            ax.scatter(v[~win], y[~win], s=22, color=GRAY, lw=0.8, edgecolors=SURFACE, zorder=3)
            ax.scatter(v[win], y[win], s=26, color=BLUE, lw=0.8, edgecolors=SURFACE, zorder=4)
            ax.text(n * 1.9, i, f"{win.sum()}/{len(v)}", fontsize=7.5, va="center", ha="center", color=INK if win.mean() > 0.5 else MUTED)
        win_all = (r.best_own_rank < r.chance_rank).mean()
        ax.set_title(f"{ATLAS_LABEL[atlas]}\n{win_all:.0%} of diseases beat the chance rank (≈50% expected)", fontsize=9, color=INK, loc="left")
        ax.set_xscale("log"); ax.set_xlim(0.8, n * 2.6)
        ticks = [t for t in (1, 3, 10, 30, 100, 300) if t <= n]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        clean(ax, grid_y=False); ax.xaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel(f"rank of the best own-organ cell type (of {n}; left = better)")
        ax.text(n * 1.9, -1, "beat\nchance", fontsize=7, color=INK2, ha="center", va="center")
    axes[0].set_yticks(range(len(organs))); axes[0].set_yticklabels([f"{o.replace('.', ' ')}  (n={counts[o]})" for o in organs])
    axes[0].set_ylim(len(organs) - 0.5, -1.6)
    fig.text(0.01, 0.995, "Per disease: where does its best own-organ cell type rank?", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.995 - 0.33 / fig.get_figheight(), "One dot per single-organ disease. Black tick = chance: the organ's median best rank across all other diseases.\n"
             "Blue = better than that. Organs sorted by median distance from chance, both atlases.", fontsize=8, color=INK2, va="top")
    fig.subplots_adjust(left=0.17, right=0.98, top=1 - 1.45 / fig.get_figheight(), bottom=0.75 / fig.get_figheight(), wspace=0.08)
    save(fig, "fig3_best_rank_per_disease")


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
    curves, ranks, sig, summary = {}, {}, [], []
    for a, d in data.items():
        for kind, M in [("all", d["own"]), ("parenchymal", d["own"] & d["par"][None, :])]:
            ok = M.any(1)
            curves[(a, kind)] = recall_curve(d["X"][ok], M[ok])
            ks, obs, null, p = curves[(a, kind)]
            for k in (1, 5, 10, 20, 50):
                summary.append(dict(atlas=a, own_organ_cells=kind, n_diseases=int(ok.sum()), k=k, observed=obs[k - 1],
                                    null_median=np.median(null[:, k - 1]), fold=obs[k - 1] / max(np.median(null[:, k - 1]), 1e-9), p=p[k]))
        r, m, p = best_rank_p(d["X"], d["own"])
        single = np.array([len(x) == 1 for x in d["mp"].organ_dict])
        ranks[a] = pd.DataFrame({"disease": d["mp"].Disease, "organ": [next(iter(x)) if len(x) == 1 else ";".join(x) for x in d["mp"].organ_dict],
                                 "n_own_cell_types": m, "best_own_rank": r,
                                 "chance_rank": empirical_chance_rank(d["X"], d["own"]), "p": p,
                                 "confidence": d["mp"].confidence, "single_organ": single})
        ranks[a].sort_values("p").to_csv(os.path.join(OUT, f"{a}_best_rank_per_disease.tsv"), sep="\t", index=False)
        sig.append(significance_table(a, d))
    pd.DataFrame(summary).round(4).to_csv(os.path.join(OUT, "hit_rate_by_depth.tsv"), sep="\t", index=False)
    pd.concat(sig).round(4).to_csv(os.path.join(OUT, "significant_hits_in_own_organ.tsv"), sep="\t", index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print(pd.concat(sig).round(3).to_string(index=False))
    for a in ATLASES:
        r = ranks[a]; print(a, "diseases p<0.05:", (r.p < 0.05).sum(), "/", r.p.notna().sum(),
                            "| high-confidence:", (r[r.confidence == "high"].p < 0.05).sum(), "/", (r.confidence == "high").sum())
        print(r.sort_values("p").head(12).to_string(index=False))

    fig_recall(curves)
    for a in ATLASES:
        r = ranks[a]; print(a, "beat chance rank:", (r.best_own_rank < r.chance_rank).mean().round(3),
                            "| best own cell type is #1:", (r.best_own_rank == 1).sum(), "/", len(r))
    fig_best_rank({a: ranks[a][ranks[a].single_organ] for a in ATLASES}, {a: data[a]["X"].shape[1] for a in ATLASES})
    if len(sys.argv) > 1: picks = sys.argv[1].split("|")
    else: picks = ["Asthma, excluding more controls", "Gastric ulcer", "Heart failure, strict", "Chronic kidney disease"]
    fig_examples(data["hpa700"], [p for p in picks if p in set(data["hpa700"]["mp"].Disease)])


if __name__ == "__main__":
    main()
