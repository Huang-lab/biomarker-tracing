"""
Global view of the results: do cell types and diseases organise by organ?

For every disease with a primary organ (organ_mapping/disease_organ_map.tsv, status "scorable"),
takes its disease-specific cell-type association from the cross-disease summary
(`specificity_z`: how much more a cell type is associated with this disease than with a typical
disease; 0 = typical) and draws two figures per atlas:

  1. <atlas>_clustermap.png/.pdf
     diseases x cell types, both axes clustered (average linkage, correlation distance). Left bar:
     the disease's primary organ system. Top bar: the cell type's tissue's organ system (same colours).
     If cell types cluster by organ, the top bar forms contiguous blocks.
  2. <atlas>_organ_by_tissue.png/.pdf
     the same data collapsed to organ x tissue: mean specificity z of the diseases whose primary organ is
     the row, over the cell types of the column's tissue. Organs share the order on both axes, and
     row = column cells are outlined, so organ concordance shows as an outlined diagonal.

Also writes <atlas>_stats.tsv: (a) organ concordance, the AUC of a disease's own-organ cell types against
all others, averaged over diseases, with a null from shuffling organ labels across diseases; (b) whether
cell types cluster more by tissue or by (approximate) cell lineage, and diseases by organ system, as
silhouette scores against label-shuffling nulls.

Colours: categorical slots 1-8 of the reference palette for the eight organ systems (fixed order);
diverging blue-red with a gray midpoint, equal lightness steps per arm, for the signed z.

Usage (repository root):
    python organ_mapping/plot_organ_heatmaps.py --atlas both
"""
import argparse
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- reference palette (see the dataviz skill's palette.md), light surface --------------------
SURFACE, INK, INK2, MUTED, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MID_GRAY = "#f0efec"
# eight organ systems, in the palette's fixed slot order; organs per system
SYSTEMS = OrderedDict([
    ("nervous & sensory",      ("#2a78d6", ["brain", "eye"])),
    ("digestive",              ("#eb6834", ["esophagus", "stomach", "small.intestine", "colon", "rectum", "liver", "pancreas", "salivary.gland", "tongue"])),
    ("respiratory",            ("#1baf7a", ["lung"])),
    ("skin",                   ("#eda100", ["skin"])),
    ("urinary & reproductive", ("#e87ba4", ["kidney", "urinary.bladder", "prostate", "testis", "epididymis", "ovary", "fallopian.tube", "endometrium", "placenta", "breast"])),
    ("endocrine, fat & muscle", ("#008300", ["adrenal.gland", "pituitary.gland", "adipose.tissue", "skeletal.muscle"])),
    ("blood & immune",         ("#6250d6", ["blood", "bone.marrow", "spleen", "lymph.node", "thymus"])),
    ("cardiovascular",         ("#e34948", ["heart.muscle", "vasculature"])),
])
ORGAN_SYSTEM = {o: s for s, (_, organs) in SYSTEMS.items() for o in organs}
ORGAN_ORDER = [o for _, (_, organs) in SYSTEMS.items() for o in organs]
SYS_COLOR = {s: c for s, (c, _) in SYSTEMS.items()}
SYS_ORDER = list(SYSTEMS)

# approximate cell lineage from the cell-type name, to ask whether cell types follow organ or lineage
LINEAGE_RULES = [
    ("erythroid/platelet", r"erythro|reticulocyte|platelet|megakaryo"),
    ("immune", r"macrophage|monocyte|t\.cell|b\.cell|plasma|nk\.|natural\.killer|dendritic|mast|neutrophil|granulocyte|lymphocyte|microglia|langerhans|kupffer|hofbauer|basophil|eosinophil|leukocyte|lymphoid|myeloid|treg|thymocyte|\.t\.|^t\.|^b\.|immune"),
    ("endothelial", r"endothelial|lymphatic"),
    ("stromal", r"fibroblast|stromal|pericyte|smooth\.muscle|mesenchym|mesothelial|myofibro|fibro\.|adventitial"),
    ("neural", r"neuron|astrocyte|oligodendro|opc|glia|schwann|ependym|neural|glial"),
    ("muscle", r"cardiomyocyte|myocyte|myoblast|satellite|skeletal\.muscle|muscle\.cell"),
]


def lineage_of(celltype):
    for name, rx in LINEAGE_RULES:
        if re.search(rx, celltype.lower()): return name
    return "epithelial/other"


# ---- diverging colour map: gray midpoint, equal OKLab lightness on both arms -----------------
def _lin(c): c = np.asarray(c, float); return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def _gam(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
def hex2rgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
def rgb2hex(r): return "#" + "".join(f"{int(round(x * 255)):02x}" for x in np.clip(r, 0, 1))


def to_oklab(rgb):
    r, g, b = _lin(rgb)
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])


def from_oklab(lab):
    L, a, b = lab
    l, m, s = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return _gam(np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                          -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                          -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s]))


def diverging_cmap():
    """Blue arm = palette blue steps 200/400/600; red arm = slot-8 red hue at the same lightness and chroma."""
    blue = ["#9ec5f4", "#3987e5", "#184f95"]
    red_h = np.arctan2(*to_oklab(hex2rgb(SYS_COLOR["cardiovascular"]))[:0:-1])
    red = []
    for b in blue:
        L, a, bb = to_oklab(hex2rgb(b)); C = np.hypot(a, bb)
        red.append(rgb2hex(from_oklab((L, C * np.cos(red_h), C * np.sin(red_h)))))
    stops = [(0.0, blue[2]), (1 / 6, blue[1]), (2 / 6, blue[0]), (0.5, MID_GRAY), (4 / 6, red[0]), (5 / 6, red[1]), (1.0, red[2])]
    return LinearSegmentedColormap.from_list("div_blue_red", stops), blue, red


# ---- data ------------------------------------------------------------------------------------
def load(atlas, root):
    mp = pd.read_csv(os.path.join(HERE, "disease_organ_map.tsv"), sep="\t", dtype=str).fillna("")
    man = pd.read_csv(os.path.join(root, "ppa_prevalent_sumstats", "manifest.tsv"), sep="\t", dtype=str)
    man = man[man.status == "ok"].assign(stem=lambda d: d.file.str.replace(".csv", "", regex=False))
    mp = mp.merge(man[["Disease_code", "stem"]], on="Disease_code")
    mp = mp[mp.status == "scorable"].copy()
    h = pd.read_csv(os.path.join(root, f"results_ppa_prevalent_{atlas}", "ppa_prevalent_sumstats", "cross_disease_summary",
                                 "disease_specific_hits.tsv"), sep="\t", usecols=["disease", "cell_tissue", "specificity_z"])
    z = h.pivot(index="disease", columns="cell_tissue", values="specificity_z")
    mp = mp[mp.stem.isin(z.index)].reset_index(drop=True)
    Z = z.loc[mp.stem]
    Z.index = mp.Disease.values

    vocab = pd.read_csv(os.path.join(HERE, "organ_vocabulary.tsv"), sep="\t", dtype=str)
    tissue2organ = dict(zip(vocab.atlas_tissue, vocab.organ))
    col_tissue = [c.split("_")[0] for c in Z.columns]
    col_organ = [tissue2organ[t] for t in col_tissue]

    def organs_with_support(r):
        d = {}
        for x in r.organ_support.split(";"):
            o, s = x.split(":"); d[o] = float(s)
        return {o: d.get(o, 1.0) for o in r.organs_primary.split(";") if o}

    mp["organ_dict"] = mp.apply(organs_with_support, axis=1)

    def system_of(d):                      # system with the most support; ties go to the earlier palette slot
        score = {s: 0.0 for s in SYS_ORDER}
        for o, sup in d.items(): score[ORGAN_SYSTEM[o]] += sup
        return max(SYS_ORDER, key=lambda s: (round(score[s], 6), -SYS_ORDER.index(s)))

    mp["system"] = mp.organ_dict.apply(system_of)
    return mp, Z, np.array(col_tissue), np.array(col_organ)


# ---- statistics ------------------------------------------------------------------------------
def concordance(mp, Z, col_organ, n_perm=5000, seed=0):
    """AUC of own-organ cell types vs the rest, per disease; null = organ sets shuffled across diseases."""
    rng = np.random.default_rng(seed)
    R = np.apply_along_axis(rankdata, 1, Z.to_numpy())
    mask = np.array([[o in d for o in col_organ] for d in mp.organ_dict])
    keep = mask.any(1) & ~mask.all(1)
    R, mask, sub = R[keep], mask[keep], mp[keep].reset_index(drop=True)
    n = R.shape[1]

    def auc(Rm, M):
        nt = M.sum(1); return ((Rm * M).sum(1) - nt * (nt + 1) / 2) / (nt * (n - nt))

    obs = auc(R, mask)
    null = np.array([auc(R, mask[rng.permutation(len(mask))]).mean() for _ in range(n_perm)])
    p = (1 + (null >= obs.mean()).sum()) / (n_perm + 1)
    return obs, null, p, sub


def topk_enrichment(mp, Z, col_organ, k=10, n_perm=2000, seed=0):
    """Focal test: the share of a disease's top-k cell types (by specificity z) that belong to its own organ(s),
    against (a) the share expected from how many columns the organ has and (b) organ labels shuffled across
    diseases. The mean-rank AUC is diluted when only a few of an organ's many cell types matter; this is not."""
    X = Z.to_numpy(); mask = np.array([[o in d for o in col_organ] for d in mp.organ_dict])
    keep = mask.any(1) & ~mask.all(1)
    Mk, tk = mask[keep], np.argsort(-X, axis=1)[keep][:, :k]
    obs = np.take_along_axis(Mk, tk, axis=1).mean(1)
    rng = np.random.default_rng(seed)
    null = np.array([np.take_along_axis(Mk[rng.permutation(len(Mk))], tk, axis=1).mean() for _ in range(n_perm)])
    p = (1 + (null >= obs.mean()).sum()) / (n_perm + 1)
    single = np.array([len(d) == 1 for d in mp.organ_dict])[keep]
    organ = np.array([next(iter(d)) if len(d) == 1 else None for d in mp.organ_dict])[keep]
    per = (pd.DataFrame({"organ": organ, "share_in_own_organ": obs, "chance": Mk.mean(1)})[single].groupby("organ")
           .agg(n_diseases=("share_in_own_organ", "size"), share_in_own_organ=("share_in_own_organ", "mean"), chance=("chance", "mean")))
    per["fold_over_chance"] = per.share_in_own_organ / per.chance
    return obs, Mk.mean(1), null, p, per.sort_values("fold_over_chance", ascending=False), k


def organ_auc(mp, Z, col_organ):
    """For each disease and each of its primary organs: AUC of that organ's cell types against the cell types of
    organs that are not among the disease's primary organs (1 = the organ's cell types rank highest)."""
    X = Z.to_numpy(); out = []
    for i, d in enumerate(mp.organ_dict):
        rest = ~np.isin(col_organ, list(d))
        for o in d:
            tgt = col_organ == o
            if not tgt.any() or not rest.any(): continue
            sel = tgt | rest
            r = rankdata(X[i, sel]); nt = tgt.sum()
            out.append((mp.Disease[i], o, (r[tgt[sel]].sum() - nt * (nt + 1) / 2) / (nt * rest.sum()), nt))
    return pd.DataFrame(out, columns=["disease", "organ", "auc", "n_cell_types"])


def silhouette_vs_null(D, labels, n_perm=500, seed=0):
    labels = np.asarray(labels); rng = np.random.default_rng(seed)
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels): return np.nan, np.nan, np.nan
    obs = silhouette_score(D, labels, metric="precomputed")
    null = np.array([silhouette_score(D, rng.permutation(labels), metric="precomputed") for _ in range(n_perm)])
    return obs, null.mean(), (1 + (null >= obs).sum()) / (n_perm + 1)


# ---- drawing helpers ---------------------------------------------------------------------------
def style_axes(ax):
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def cluster_order(X, axis):
    M = X if axis == 0 else X.T
    d = pdist(M, "correlation")
    Z = optimal_leaf_ordering(linkage(d, "average"), d)
    return Z, leaves_list(Z)


def runs(labels, min_len):
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if i - start >= min_len: out.append((start, i, labels[start]))
            start = i
    return out


def clustermap(atlas, mp, Z, col_tissue, col_organ, stats_line, out_dir, cmap, args_matrix_dir, vmax=4.0):
    X = Z.to_numpy()
    zr, ro = cluster_order(X, 0)
    zc, co = cluster_order(X, 1)
    Xo = X[np.ix_(ro, co)]
    sys_rows = mp.system.to_numpy()[ro]
    sys_cols = np.array([ORGAN_SYSTEM[o] for o in col_organ])[co]
    names = Z.index.to_numpy()[ro]
    organs_txt = mp.organ_dict.apply(lambda d: "/".join(o.replace(".", " ") for o in d)).to_numpy()[ro]

    fig = plt.figure(figsize=(12, 17), facecolor=SURFACE)
    L, W, B, H = 0.105, 0.60, 0.100, 0.760
    ax_h = fig.add_axes([L, B, W, H])
    ax_h.imshow(Xo, aspect="auto", cmap=cmap, norm=Normalize(-vmax, vmax), interpolation="nearest")
    style_axes(ax_h)

    ax_rd = fig.add_axes([0.012, B, 0.075, H]); ax_rd.set_facecolor(SURFACE)
    dendrogram(zr, orientation="left", no_labels=True, color_threshold=0, above_threshold_color=MUTED, ax=ax_rd, link_color_func=lambda k: MUTED)
    ax_rd.invert_yaxis(); style_axes(ax_rd)
    for c in ax_rd.collections: c.set_linewidth(0.5)
    ax_cd = fig.add_axes([L, B + H + 0.026, W, 0.04]); ax_cd.set_facecolor(SURFACE)
    dendrogram(zc, orientation="top", no_labels=True, color_threshold=0, above_threshold_color=MUTED, ax=ax_cd, link_color_func=lambda k: MUTED)
    style_axes(ax_cd)
    for c in ax_cd.collections: c.set_linewidth(0.5)

    ax_rb = fig.add_axes([0.090, B, 0.011, H]); ax_cb = fig.add_axes([L, B + H + 0.005, W, 0.016])
    ax_rb.imshow(np.array([[matplotlib.colors.to_rgb(SYS_COLOR[s])] for s in sys_rows]), aspect="auto", interpolation="nearest"); style_axes(ax_rb)
    ax_cb.imshow(np.array([[matplotlib.colors.to_rgb(SYS_COLOR[s]) for s in sys_cols]]), aspect="auto", interpolation="nearest"); style_axes(ax_cb)

    # disease labels on the right: name (primary organs)
    for i, (n, o) in enumerate(zip(names, organs_txt)):
        ax_h.text(Xo.shape[1] + 0.4, i, f"{n[:46]}  ({o[:30]})", fontsize=3.7, va="center", ha="left", color=INK2, clip_on=False)
    # tissue labels under the heatmap for runs of the same tissue
    min_run = 6 if Xo.shape[1] > 400 else 3
    for a, b, t in runs(col_tissue[co], min_run):
        ax_h.text((a + b - 1) / 2, Xo.shape[0] + 1.0, t.replace(".", " "), rotation=90, fontsize=5.2, ha="center", va="top", color=INK2, clip_on=False)

    # legend (systems) and colour scale
    handles = [Patch(facecolor=SYS_COLOR[s], label=s, edgecolor="none") for s in SYS_ORDER]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.095, 0.012), ncol=4, frameon=False, fontsize=7.5,
               labelcolor=INK2, title="Organ system (bars: disease primary site, left; cell-type tissue, top)", title_fontsize=7.5, alignment="left", columnspacing=1.6, handlelength=1.2)
    cax = fig.add_axes([0.76, 0.040, 0.14, 0.009])
    sm = matplotlib.cm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-vmax, -vmax / 2, 0, vmax / 2, vmax]); cb.outline.set_visible(False)
    cb.ax.set_xticklabels([f"≤{-vmax:g}", f"{-vmax / 2:g}", "0", f"{vmax / 2:g}", f"≥{vmax:g}"])
    cb.ax.tick_params(labelsize=7, colors=INK2, length=2)
    fig.text(0.76, 0.058, "Disease-specific association (z)\nred = more than a typical disease", fontsize=7.5, color=INK2, va="bottom")

    nR, nC = Xo.shape
    fig.text(0.012, 0.985, f"Cell types and diseases by organ: {nR} diseases × {nC} cell types", fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.966, f"Atlas: {atlas}. Rows and columns clustered (average linkage, correlation distance). Disease-specific z removes each cell type's baseline across all 268 diseases.",
             fontsize=8, color=INK2, va="top")
    fig.text(0.012, 0.952, stats_line, fontsize=8, color=INK2, va="top")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{atlas}_clustermap.{ext}"), dpi=220 if ext == "png" else None, facecolor=SURFACE)
    plt.close(fig)
    pd.DataFrame(Xo, index=names, columns=Z.columns.to_numpy()[co]).round(3).to_csv(os.path.join(args_matrix_dir, f"{atlas}_clustermap_matrix.tsv"), sep="\t")


def organ_by_tissue(atlas, mp, Z, col_organ, stats_line, out_dir, cmap, auc_df):
    X = Z.to_numpy()
    cols = [o for o in ORGAN_ORDER if (col_organ == o).any()]
    rows = [o for o in ORGAN_ORDER if any(o in d for d in mp.organ_dict)]
    per_dis = np.column_stack([X[:, col_organ == c].mean(1) for c in cols])          # each disease: mean z over the tissue's cell types
    M = np.full((len(rows), len(cols)), np.nan); n_dis = []
    for i, r in enumerate(rows):
        sel = np.array([r in d for d in mp.organ_dict]); n_dis.append(int(sel.sum()))
        M[i] = per_dis[sel].mean(0)
    pd.DataFrame(M, index=[f"{r} (n={n})" for r, n in zip(rows, n_dis)], columns=cols).round(3).to_csv(os.path.join(out_dir, f"{atlas}_organ_by_tissue_mean_z.tsv"), sep="\t")

    vm = max(0.5, np.ceil(np.nanmax(np.abs(M)) * 10) / 10)
    fig = plt.figure(figsize=(12, 9.4), facecolor=SURFACE)
    ax = fig.add_axes([0.205, 0.215, 0.495, 0.64])
    ax.pcolormesh(M, cmap=cmap, norm=Normalize(-vm, vm), edgecolors=SURFACE, linewidth=1.0)
    ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels([c.replace(".", " ") for c in cols], rotation=90, fontsize=8, color=INK2)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([f"{r.replace('.', ' ')}  (n={n})" for r, n in zip(rows, n_dis)], fontsize=8)
    for lbl, n in zip(ax.get_yticklabels(), n_dis): lbl.set_color(INK2 if n >= 3 else MUTED)   # few diseases: muted
    ax.tick_params(length=0, pad=14)
    for s in ax.spines.values(): s.set_visible(False)
    # outline the cells where the disease organ and the tissue are the same organ
    for i, r in enumerate(rows):
        if r in cols: ax.add_patch(Rectangle((cols.index(r), i), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    # system bars next to the labels
    for i, r in enumerate(rows): ax.add_patch(Rectangle((-0.62, i + 0.04), 0.44, 0.92, color=SYS_COLOR[ORGAN_SYSTEM[r]], clip_on=False, lw=0))
    for j, c in enumerate(cols): ax.add_patch(Rectangle((j + 0.04, -0.62), 0.92, 0.44, color=SYS_COLOR[ORGAN_SYSTEM[c]], clip_on=False, lw=0))
    # right panel: AUC per organ (dots = diseases, bar = mean), rows aligned with the matrix
    ax2 = fig.add_axes([0.745, 0.215, 0.15, 0.64]); rng = np.random.default_rng(1)
    ax2.set_ylim(len(rows), 0); ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color=MUTED, lw=0.8)
    for i, r in enumerate(rows):
        a = auc_df.loc[auc_df.organ == r, "auc"].to_numpy()
        if not len(a):                      # organ has no cell types in this atlas, so nothing to rank
            ax2.text(0.5, i + 0.5, "organ not in this atlas", fontsize=7, va="center", ha="center", color=MUTED, style="italic"); continue
        ax2.scatter(a, i + 0.5 + rng.uniform(-0.27, 0.27, len(a)), s=7, color=MUTED, alpha=0.75, linewidths=0)
        ax2.plot([a.mean()] * 2, [i + 0.12, i + 0.88], color=INK, lw=2.2, solid_capstyle="butt")
        ax2.text(1.04, i + 0.5, f"{a.mean():.2f}", fontsize=7.5, va="center", color=INK2 if len(a) >= 3 else MUTED)
    for sp in ("top", "right", "left"): ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(HAIR); ax2.set_yticks([]); ax2.set_xticks([0, 0.5, 1])
    ax2.tick_params(axis="x", labelsize=7.5, colors=INK2, length=2)
    ax2.set_xlabel("AUC: own-organ cell types vs other organs\n(0.5 = no difference, 1 = own organ ranks highest)", fontsize=7.5, color=INK2, labelpad=6)
    ax.set_xlabel("Tissue of the cell types", fontsize=9, color=INK2, labelpad=8)
    ax.set_ylabel("Primary organ of the disease", fontsize=9, color=INK2, labelpad=8)

    handles = [Patch(facecolor=SYS_COLOR[s], label=s, edgecolor="none") for s in SYS_ORDER]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.205, 0.015), ncol=4, frameon=False, fontsize=8, labelcolor=INK2, columnspacing=1.6, handlelength=1.2)
    cax = fig.add_axes([0.70, 0.045, 0.2, 0.012])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(-vm, vm), cmap=cmap), cax=cax, orientation="horizontal", ticks=[-vm, 0, vm])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, colors=INK2, length=2)
    fig.text(0.70, 0.07, "mean disease-specific z (red = above a typical disease)", fontsize=8, color=INK2, va="bottom")
    fig.text(0.012, 0.985, "Do a disease's strongest cell types sit in its own organ?", fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.962, f"Atlas: {atlas}. Rows: diseases by primary organ (a disease counts in each of its primary organs). Columns: tissue of the cell type. "
             "Outlined cells = same organ.", fontsize=8, color=INK2, va="top")
    fig.text(0.012, 0.945, stats_line, fontsize=8, color=INK2, va="top")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{atlas}_organ_by_tissue.{ext}"), dpi=220 if ext == "png" else None, facecolor=SURFACE)
    plt.close(fig)


def main(args):
    cmap, blue, red = diverging_cmap()
    out_dir = args.out_dir; os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.root), exist_ok=True)
    for atlas in (["hpa700", "pp229"] if args.atlas == "both" else [args.atlas]):
        mp, Z, col_tissue, col_organ = load(atlas, args.root)
        print(f"\n=== {atlas}: {Z.shape[0]} diseases x {Z.shape[1]} cell types")

        obs, null, p, sub = concordance(mp, Z, col_organ)
        # cell-type clustering: by tissue vs by lineage; diseases: by organ system
        Dc = squareform(pdist(Z.to_numpy().T, "correlation")); Dr = squareform(pdist(Z.to_numpy(), "correlation"))
        lineage = [lineage_of(c.split("_", 1)[1]) for c in Z.columns]
        s_t = silhouette_vs_null(Dc, col_tissue); s_l = silhouette_vs_null(Dc, lineage); s_d = silhouette_vs_null(Dr, mp.system.to_numpy())
        tko, tkexp, tknull, tkp, tk_per, K = topk_enrichment(mp, Z, col_organ)
        tk_per.round(4).to_csv(os.path.join(out_dir, f"{atlas}_organ_topk.tsv"), sep="\t")
        stats = pd.DataFrame([
            ["organ concordance: mean AUC (own-organ cell types vs the rest)", obs.mean(), null.mean(), p, f"{len(obs)} diseases; {(obs > 0.5).mean():.0%} have AUC > 0.5"],
            ["cell types: silhouette by tissue", *s_t, "correlation distance across diseases"],
            ["cell types: silhouette by cell lineage (approximate)", *s_l, "; ".join(f"{k}: {v}" for k, v in pd.Series(lineage).value_counts().items())],
            ["diseases: silhouette by primary organ system", *s_d, "correlation distance across cell types"],
            [f"top-{K} cell types in the disease's own organ: mean share", tko.mean(), tknull.mean(), tkp, f"{tko.size} diseases; chance from column share {tkexp.mean():.3f}; {(tko > 0).mean():.0%} have at least one"],
        ], columns=["measure", "observed", "null_mean", "p_value", "note"])
        stats.to_csv(os.path.join(out_dir, f"{atlas}_stats.tsv"), sep="\t", index=False)
        print(stats.round(4).to_string(index=False))
        auc_df = organ_auc(mp, Z, col_organ)
        auc_df.round(4).to_csv(os.path.join(out_dir, f"{atlas}_organ_auc_per_disease.tsv"), sep="\t", index=False)
        line1 = (f"Silhouette of the groupings (higher = tighter; shuffled labels in brackets): cell lineage {s_l[0]:.2f} ({s_l[1]:.2f}), "
                 f"tissue {s_t[0]:.2f} ({s_t[1]:.2f}); disease organ system {s_d[0]:.2f} ({s_d[1]:.2f}).")
        line2 = (f"Mean-rank test: a disease's own-organ cell types rank at AUC {obs.mean():.2f} vs {null.mean():.2f} for shuffled organ labels (p = {p:.2f}, {len(obs)} diseases); per organ at right.\n"
                 f"Focal test: {tko.mean():.1%} of each disease's top {K} cell types are in its own organ vs {tknull.mean():.1%} with shuffled organ labels (p = {tkp:.4f}).")
        os.makedirs(args.matrix_dir, exist_ok=True)
        clustermap(atlas, mp, Z, col_tissue, col_organ, line1, out_dir, cmap, args.matrix_dir)
        organ_by_tissue(atlas, mp, Z, col_organ, line2, out_dir, cmap, auc_df)
    print("\ndiverging arms:", blue, red)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", choices=["hpa700", "pp229", "both"], default="both")
    p.add_argument("--root", default="data", help="folder holding the downloaded sumstats and results_ppa_prevalent_<atlas>")
    p.add_argument("--out-dir", default=os.path.join("organ_mapping", "figures"))
    p.add_argument("--matrix-dir", default=os.path.join("data", "organ_heatmaps"), help="where the large clustered matrices (table twins) go")
    main(p.parse_args())
