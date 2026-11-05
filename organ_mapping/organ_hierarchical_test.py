"""
Formal per-disease and per-organ test of organ concordance (complements the best-rank view of evaluate_organ_hits.py).

1. Organ score. For disease d and organ o: S(d, o) = mean specificity_z over all of o's cell types in d, minus the mean
   over all cell types in d (removes disease-wide shifts). Uses the whole organ, not only its best cell type.
2. Pair z and p. For a disease-organ pair (o among d's primary or secondary organs), z = (S(d, o) - mean) / SD, with mean
   and SD over the comparison diseases (diseases WITHOUT o as a primary or secondary organ). p_norm = upper normal tail.
   p_cal = p_norm calibrated against label shuffles: organ assignments are permuted across diseases, every pair p is
   recomputed, and p_cal is the share of those null p-values at or below the observed one. Half of the N_SHUFFLE shuffles
   are the reference; the other half is held out to check the calibration.
3. Hierarchical FDR (Benjamini & Bogomolov 2014). Families = organ x role (primary / secondary). Family p = the
   mixed-model organ p (step 4; one-sided) for primary families, Simes combination of the pairs' p_cal for secondary
   families (too few diseases for the model). Families are selected by BH at Q; inside each of the R selected families (of m),
   pairs are tested by BH at Q * R / m. This mirrors how results are reported: organ first, then diseases in it.
4. Organ effect sizes. Mixed model on the primary pairs' z: z ~ 1 + (1 | organ) + (1 | near-duplicate group within organ),
   fitted by REML (statsmodels). Organ effect = fixed intercept + organ BLUP, with an approximate 95% interval
   (variance of the intercept + conditional variance of the BLUP). Near-duplicate diseases share a group effect, so they
   do not count as independent evidence. Per-pair "model_effect" = the shrunken estimate for that disease's group.
5. Calibration: under label shuffles, share of pair p-values below 0.05 / 0.01 / 0.001, share of shuffles with any
   hierarchical-FDR discovery, share of organ intervals excluding 0.
6. Power: a shift of delta (in comparison-SD units) is added to the score of disease-organ pairs that are NOT real
   (null pairs); power = share then passing p_cal < 0.05 and the Bonferroni threshold 0.05 / number of pairs.
   Also reported per organ: the mean shift in specificity_z across the organ's cell types needed for 80% power.

Outputs (figures/evaluation/): <atlas>_organ_scores.tsv, <atlas>_organ_effects.tsv, calibration.tsv,
fig6_organ_effects.png/pdf, fig7_calibration_power.png/pdf.
Usage (repository root):  python organ_mapping/organ_hierarchical_test.py
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_organ_hits as E  # noqa: E402
from evaluate_organ_hits import ATLASES, ATLAS_LABEL, OUT, SURFACE, INK, INK2, MUTED, HAIR, BLUE, GRAY  # noqa: E402
from plot_disease_organ_examples import near_duplicate_groups  # noqa: E402

Q = 0.05
N_SHUFFLE = 500           # label shuffles for p calibration
N_SHUFFLE_MODEL = 100     # label shuffles for the mixed-model / hierarchical-FDR calibration
ORANGE = "#eb6834"
DELTAS = np.round(np.arange(0, 10.01, 0.25), 2)


# ---- scores and pair p-values -------------------------------------------------------------------
def organ_scores(d):
    X, co = d["X"], d["col_organ"]
    organs = np.array(sorted(set(co)))
    S = np.stack([X[:, co == o].mean(1) for o in organs], 1) - X.mean(1, keepdims=True)
    own = np.stack([d["own"][:, co == o].any(1) for o in organs], 1)
    anyo = np.stack([d["own_any"][:, co == o].any(1) for o in organs], 1)
    return organs, S, own, anyo


def pair_z(S, anyo):
    """z of every disease for every organ against that organ's comparison diseases; also the comparison SD. NaN where no pair."""
    Z = np.full(S.shape, np.nan); sd = np.full(S.shape[1], np.nan)
    for k in range(S.shape[1]):
        nul = S[~anyo[:, k], k]
        if not anyo[:, k].any() or len(nul) < 10: continue
        sd[k] = nul.std(ddof=1); Z[:, k] = (S[:, k] - nul.mean()) / sd[k]
    return Z, sd


def null_pvalues(S, anyo, n, rng):
    """pair p_norm values under label shuffles (organ sets permuted across diseases)."""
    out = []
    for _ in range(n):
        A = anyo[rng.permutation(len(S))]
        Z, _ = pair_z(S, A)
        out.append(norm.sf(Z[A & ~np.isnan(Z)]))
    return out


def calibrate(p, null_sorted):
    return (1 + np.searchsorted(null_sorted, p, side="right")) / (1 + len(null_sorted))


# ---- hierarchical FDR -------------------------------------------------------------------------
def simes(p):
    p = np.sort(p); n = len(p)
    return min(1.0, np.min(p * n / np.arange(1, n + 1)))


def bb_fdr(t, model_p, pcol="p_cal", q=Q):
    """Benjamini-Bogomolov: select organ x role families by BH on their family p, then BH inside selected families at q*R/m.
    Family p = the mixed-model organ p (one-sided, effect > 0) for primary families; Simes of the pairs' p for secondary
    families, which are too small for the model."""
    fam = t.groupby(["organ", "role"])[pcol].apply(lambda s: simes(s.to_numpy())).rename("family_p").reset_index()
    prim = fam.role == "primary"
    fam.loc[prim, "family_p"] = fam.loc[prim, "organ"].map(model_p).fillna(1.0).to_numpy()
    fam["family_q"] = E.bh(fam.family_p.to_numpy()); fam["selected"] = fam.family_q <= q
    R, m = int(fam.selected.sum()), len(fam)
    t = t.merge(fam, on=["organ", "role"], how="left")
    t["q_within"] = np.nan; t["significant"] = False
    for (o, r), idx in t.groupby(["organ", "role"]).groups.items():
        if not fam.set_index(["organ", "role"]).selected.get((o, r), False): continue
        qq = E.bh(t.loc[idx, pcol].to_numpy())
        t.loc[idx, "q_within"] = qq; t.loc[idx, "significant"] = qq <= q * R / m
    return t, fam, R, m


# ---- mixed model ------------------------------------------------------------------------------
def fit_model(t):
    """z ~ 1 + (1|organ) + (1|group within organ) on primary pairs. Returns organ table and per-row shrunken effect."""
    p = t[t.role == "primary"].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("z ~ 1", data=p, groups="organ", re_formula="1", vc_formula={"grp": "0 + C(dup_group)"}).fit(reml=True)
        b0, v0 = m.fe_params["Intercept"], m.bse_fe["Intercept"] ** 2
        re_all, cov_all = m.random_effects, m.random_effects_cov
    rows, eff = [], np.full(len(p), np.nan)
    for o, re in re_all.items():
        cov = cov_all[o]
        est = b0 + re.iloc[0]; se = np.sqrt(v0 + cov.iloc[0, 0])
        g = p[p.organ == o]
        rows.append(dict(organ=o, n_diseases=len(g), n_groups=g.dup_group.nunique(), mean_z=g.z.mean(), effect=est,
                         ci_low=est - 1.96 * se, ci_high=est + 1.96 * se, model_p=norm.sf(est / se)))
        for i in g.index:                                        # disease's group effect (organ + its near-duplicate group)
            key = f"grp[C(dup_group)[{p.dup_group[i]}]]"
            eff[i] = est + (re[key] if key in re.index else 0.0)
    ot = pd.DataFrame(rows).sort_values("effect", ascending=False).reset_index(drop=True)
    return ot, pd.Series(eff, index=p.set_index(["disease", "organ"]).index), m


# ---- main analysis per atlas ------------------------------------------------------------------
def analyse(atlas, rng):
    d = E.prepare(atlas); mp = d["mp"]
    organs, S, own, anyo = organ_scores(d)
    Z, sd = pair_z(S, anyo)
    _, gid, *_ = near_duplicate_groups(mp)
    nulls = null_pvalues(S, anyo, N_SHUFFLE, rng)
    null_sorted = np.sort(np.concatenate(nulls[: N_SHUFFLE // 2]))        # reference half; the other half checks calibration
    held_out = np.concatenate(nulls[N_SHUFFLE // 2:])
    ii, kk = np.where(anyo)
    t = pd.DataFrame(dict(disease=mp.Disease.to_numpy(dtype=object)[ii], organ=organs[kk],
                          role=np.where(own[ii, kk], "primary", "secondary"),
                          n_cell_types=[int((d["col_organ"] == organs[k]).sum()) for k in kk],
                          organ_score=S[ii, kk], z=Z[ii, kk], dup_group=gid[ii], confidence=mp.confidence.to_numpy(dtype=object)[ii]))
    t = t[~t.z.isna()].reset_index(drop=True)
    t["p_norm"] = norm.sf(t.z); t["p_cal"] = calibrate(t.p_norm.to_numpy(), null_sorted)
    ot, eff, _ = fit_model(t)
    t, fam, R, m = bb_fdr(t, dict(zip(ot.organ, ot.model_p)))
    t["model_effect"] = [eff.get((a, b), np.nan) for a, b in zip(t.disease, t.organ)]
    ot = ot.merge(fam[fam.role == "primary"][["organ", "family_p", "family_q", "selected"]], on="organ", how="left")
    ot = ot.merge(t[t.role == "primary"].groupby("organ").significant.sum().rename("n_significant").reset_index(), on="organ", how="left")
    ot["comparison_sd"] = ot.organ.map(dict(zip(organs, sd)))

    # calibration under label shuffles: pair p rates, hierarchical-FDR discoveries, organ intervals excluding 0
    cal = dict(atlas=atlas)
    held_cal = calibrate(held_out, null_sorted)
    for thr in (0.05, 0.01, 0.001):
        cal[f"p_norm<{thr}"] = (held_out < thr).mean()
        cal[f"p_cal<{thr}"] = (held_cal < thr).mean()
    any_disc, excl = [], []
    for _ in range(N_SHUFFLE_MODEL):
        pi = rng.permutation(len(S)); A, O = anyo[pi], own[pi]
        Zb, _ = pair_z(S, A); ib, kb = np.where(A & ~np.isnan(Zb))
        tb = pd.DataFrame(dict(disease=ib, organ=organs[kb], role=np.where(O[ib, kb], "primary", "secondary"), z=Zb[ib, kb], dup_group=gid[pi][ib]))
        tb["p_cal"] = calibrate(norm.sf(tb.z.to_numpy()), null_sorted)
        try:
            ob, *_ = fit_model(tb)
        except Exception:
            continue
        excl.append(((ob.ci_low > 0) | (ob.ci_high < 0)).mean())
        tb, *_ = bb_fdr(tb, dict(zip(ob.organ, ob.model_p))); any_disc.append(tb.significant.any())
    cal["shuffles_with_any_discovery"] = np.mean(any_disc)
    cal["organ_intervals_excluding_0"] = np.mean(excl) if excl else np.nan

    # power: shift null pairs (disease does not have the organ) by delta comparison SDs
    zn = Z[~anyo & ~np.isnan(Z)]
    bonf = Q / len(t)
    pw = pd.DataFrame(dict(delta=DELTAS, power_p05=[(calibrate(norm.sf(zn + dl), null_sorted) < 0.05).mean() for dl in DELTAS],
                           power_bonf=[(calibrate(norm.sf(zn + dl), null_sorted) < bonf).mean() for dl in DELTAS]))
    d80 = np.interp(0.8, pw.power_bonf, pw.delta)
    ot["shift_for_80pct_power"] = d80 * ot.comparison_sd           # mean specificity_z shift across the organ's cell types
    print(f"\n{atlas}: {len(t)} pairs | p_cal<0.05 {(t.p_cal < 0.05).sum()} (expected {0.05 * len(t):.0f}) | "
          f"families selected {R}/{m} | significant pairs {t.significant.sum()} ({t[t.significant].disease.nunique()} diseases, "
          f"{t[t.significant].dup_group.nunique()} near-duplicate groups)")
    print("  calibration:", {k: round(v, 4) for k, v in cal.items() if k != "atlas"})
    print(f"  80% power at Bonferroni needs a shift of {d80:.2f} comparison SDs")
    return t, ot, cal, pw, d80, zn


# ---- figures ----------------------------------------------------------------------------------
def fig_effects(res):
    ot_all = pd.concat([r[1].assign(atlas=a) for a, r in res.items()])
    order = (ot_all.groupby("organ").effect.mean().sort_values(ascending=False)).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 0.3 * len(order) + 2.6), sharey=True)
    rng = np.random.default_rng(1)
    for ax, a in zip(axes, ATLASES):
        t, ot = res[a][0], res[a][1].set_index("organ")
        tp = t[t.role == "primary"]
        ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
        for i, o in enumerate(order):
            if o not in ot.index:
                ax.text(-2.8, i, "not in this atlas", fontsize=7, color=MUTED, va="center", style="italic"); continue
            g = tp[tp.organ == o]; y = i + rng.uniform(-0.22, 0.22, len(g)); sig = g.significant.to_numpy()
            ax.scatter(g.z[~sig], y[~sig], s=14, color=GRAY, lw=0.6, edgecolors=SURFACE, zorder=2)
            ax.scatter(g.z[sig], y[sig], s=24, color=BLUE, lw=0.6, edgecolors=SURFACE, zorder=4)
            r = ot.loc[o]; strong = bool(r.selected)
            ax.plot([r.ci_low, r.ci_high], [i, i], color=INK if strong else MUTED, lw=2.0, zorder=3, solid_capstyle="butt")
            ax.scatter([r.effect], [i], marker="D", s=30, color=INK if strong else SURFACE, edgecolors=INK if strong else MUTED, lw=1.2, zorder=5)
            ax.text(1.03, i, f"{int(r.n_significant)}/{int(r.n_diseases)}", transform=ax.get_yaxis_transform(), fontsize=7.5, va="center",
                    ha="center", color=INK if r.n_significant else MUTED, fontweight="bold" if r.n_significant else "normal")
        ax.text(1.03, -1.0, "significant\ndiseases", transform=ax.get_yaxis_transform(), fontsize=6.8, color=INK2, ha="center", va="center")
        ax.set_xlim(-3, 8); ax.set_title(ATLAS_LABEL[a], fontsize=9, color=INK, loc="left")
        E.clean(ax, grid_y=False); ax.xaxis.grid(True, color=HAIR, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("organ score of the disease, in SDs above diseases without that organ (z)", fontsize=8.2)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([o.replace(".", " ") for o in order]); axes[0].set_ylim(len(order) - 0.5, -1.6)
    axes[0].scatter([], [], s=24, color=BLUE, label="disease significant (hierarchical FDR 5%)")
    axes[0].scatter([], [], s=14, color=GRAY, label="disease not significant")
    axes[0].scatter([], [], marker="D", s=30, color=INK, label="organ selected (hierarchical FDR 5%)")
    axes[0].scatter([], [], marker="D", s=30, color=SURFACE, edgecolors=MUTED, lw=1.2, label="organ not selected")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=7.8, labelcolor=INK2)
    h = fig.get_figheight()
    fig.text(0.01, 0.995, "Organ score per disease, with the organ-level effect", fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.995 - 0.32 / h,
             "Dots: one per disease with that primary organ. Organ score = mean disease-specific z over ALL the organ's cell types (minus the disease's mean), "
             "expressed in SDs\nabove the diseases that do not have that organ. Blue = significant after hierarchical FDR (organs first, then diseases within selected organs), "
             "with p calibrated by label shuffles.\nDiamond and bar: organ effect from a mixed model (near-duplicate diseases share a group effect, small organs are pulled "
             "toward the average), approximate 95% interval,\nnot corrected for the number of organs. Filled diamond = organ selected by the FDR step.", fontsize=7.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.12, right=0.95, top=1 - 1.7 / h, bottom=0.85 / h, wspace=0.16)
    E.save(fig, "fig6_organ_effects")


def fig_cal_power(res, cal):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    ax = axes[0]
    for a, col in zip(ATLASES, (BLUE, ORANGE)):
        c = cal[cal.atlas == a].iloc[0]
        x = [0.05, 0.01, 0.001]
        ax.plot(x, [c[f"p_norm<{v}"] for v in x], "o--", color=col, lw=1.2, ms=4, label=f"{a}: normal p")
        ax.plot(x, [c[f"p_cal<{v}"] for v in x], "o-", color=col, lw=2.0, ms=5, label=f"{a}: calibrated p")
    ax.plot([0.0008, 0.06], [0.0008, 0.06], color=MUTED, lw=1, ls=(0, (3, 2)))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("threshold"); ax.set_ylabel("share of shuffled-label pairs below it")
    ax.set_title("A. Calibration under shuffled organ labels\n(on the diagonal = correct)", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7, labelcolor=INK2); E.clean(ax)
    ax = axes[1]
    for a, col in zip(ATLASES, (BLUE, ORANGE)):
        pw = res[a][3]
        ax.plot(pw.delta, pw.power_p05, color=col, lw=1.2, ls=(0, (2, 2)))
        ax.plot(pw.delta, pw.power_bonf, color=col, lw=2.0, label=a)
    ax.axhline(0.8, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("added shift (SDs of diseases without the organ)"); ax.set_ylabel("power")
    ax.set_title("B. Power for one disease\nsolid: Bonferroni over all pairs; dashed: p < 0.05", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2); E.clean(ax)
    ax = axes[2]
    bins = np.arange(-3, 8.01, 0.5)
    for a, col in zip(ATLASES, (BLUE, ORANGE)):
        tp = res[a][0][res[a][0].role == "primary"]
        ax.hist(tp.z, bins=bins, histtype="step", color=col, lw=1.8, label=f"{a}: real primary pairs", density=True)
        ax.hist(res[a][5], bins=bins, histtype="stepfilled", color=col, alpha=0.15, lw=0, density=True)
        ax.axvline(res[a][4], color=col, lw=1.0, ls=(0, (3, 2)))
    ax.set_xlabel("z"); ax.set_ylabel("density")
    ax.set_title("C. z of real pairs (lines) vs non-pairs (shaded)\ndashed: z needed for 80% power (Bonferroni)", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2); E.clean(ax)
    fig.suptitle("Is the test calibrated, and how large an organ signal can it detect?", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.14, wspace=0.3)
    E.save(fig, "fig7_calibration_power")


def main():
    rng = np.random.default_rng(0)
    res, cals = {}, []
    for a in ATLASES:
        res[a] = analyse(a, rng); cals.append(res[a][2])
        t, ot = res[a][0], res[a][1]
        t.sort_values("p_cal").round(5).to_csv(os.path.join(OUT, f"{a}_organ_scores.tsv"), sep="\t", index=False)
        ot.round(4).to_csv(os.path.join(OUT, f"{a}_organ_effects.tsv"), sep="\t", index=False)
        print(ot[["organ", "n_diseases", "n_groups", "effect", "ci_low", "ci_high", "family_q", "n_significant", "shift_for_80pct_power"]].round(3).to_string(index=False))
        print(t[t.significant].sort_values("p_cal")[["disease", "organ", "role", "z", "p_cal", "q_within", "dup_group"]].round(4).to_string(index=False))
    cal = pd.DataFrame(cals); cal.round(5).to_csv(os.path.join(OUT, "calibration.tsv"), sep="\t", index=False)
    fig_effects(res); fig_cal_power(res, cal)


if __name__ == "__main__":
    main()
