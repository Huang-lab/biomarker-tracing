"""
Cross-disease summary of univariate results: which cell-type hits are disease-specific?

When the univariate test is run on many diseases, a handful of cell types (typically
macrophage / monocyte populations that express highly pleiotropic plasma proteins)
come out significant for most diseases. Their within-disease p-values are valid, but
they say little about *which* disease a cell type belongs to.

This script collects <results_dir>/<disease>/univar_regression_results.tsv across all
diseases into a diseases x cell-types t-statistic matrix and, for each cell type,
uses its own distribution of t across diseases as an empirical baseline:

    specificity_z(d, c) = ( t(d, c) - median_d' t(d', c) ) / ( 1.4826 * MAD_d' t(d', c) )

A large specificity_z means cell type c is far more associated with disease d than it
is with a typical disease, i.e. the association is not explained by c expressing
generically disease-associated proteins. Cell types significant in more than
--generic_frac of diseases are flagged as generic.

Outputs (in --save_path):
    univar_tstat_matrix.tsv     diseases x cell types, `--stat_col`
    univar_sig_matrix.tsv       diseases x cell types, 1 if `--sig_col` <= --thres
    cell_type_recurrence.tsv    per cell type: fraction of diseases where it is significant, median/MAD t
    disease_specific_hits.tsv   long table of every (disease, cell type) pair with specificity_z,
                                specificity_fdr and hit_class in {disease_specific, generic, significant, none}

Usage:
    python pipelines/cross_disease_summary.py \
        --results_dir <save_path>/<disease_folder_name>/univar_association_testing \
        --save_path   <save_path>/<disease_folder_name>/cross_disease_summary
"""
import argparse, glob, logging, os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def collect(results_dir, stat_col, sig_col):
    tstat, sig, extra = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "univar_regression_results.tsv"))):
        disease = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path, sep="\t").set_index("cell_tissue")
        if stat_col not in df.columns or sig_col not in df.columns:
            logging.warning(f"{disease}: missing {stat_col} or {sig_col}; skipped")
            continue
        tstat[disease] = df[stat_col]
        sig[disease] = df[sig_col]
        if "pval_maxT_one_side" in df.columns:
            extra[disease] = df["pval_maxT_one_side"]
    T = pd.DataFrame(tstat).T          # diseases x cell types
    S = pd.DataFrame(sig).T
    M = pd.DataFrame(extra).T if extra else None
    return T, S, M


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    T, S, M = collect(args.results_dir, args.stat_col, args.sig_col)
    n_dis, n_ct = T.shape
    logging.info(f"Collected {n_dis} diseases x {n_ct} cell types")
    if n_dis < args.min_diseases:
        logging.warning(f"Only {n_dis} diseases: the per-cell-type baseline across diseases is unreliable "
                        f"(--min_diseases={args.min_diseases}); specificity_z is reported but should not be trusted")

    T.rename_axis("disease").to_csv(f"{args.save_path}/univar_tstat_matrix.tsv", sep="\t")
    (S <= args.thres).astype(int).rename_axis("disease").to_csv(f"{args.save_path}/univar_sig_matrix.tsv", sep="\t")

    # Per-cell-type recurrence and robust baseline across diseases
    sig_mask = S <= args.thres
    med = T.median(axis=0)
    mad = (T - med).abs().median(axis=0) * 1.4826
    mad = mad.replace(0, np.nan).fillna(T.std(axis=0))
    rec = pd.DataFrame({
        "n_diseases": T.notna().sum(axis=0),
        "n_sig": sig_mask.sum(axis=0),
        "frac_sig": sig_mask.mean(axis=0),
        "median_t": med,
        "mad_t": mad,
        "mean_t": T.mean(axis=0),
    }).sort_values("frac_sig", ascending=False)
    rec["generic"] = rec["frac_sig"] > args.generic_frac
    rec.rename_axis("cell_tissue").to_csv(f"{args.save_path}/cell_type_recurrence.tsv", sep="\t")
    logging.info("Most recurrent cell types:\n" + rec.head(10).round(3).to_string())

    # Disease-specific hits: one row per (disease, cell type) pair
    def melt(df, name):
        return (df.rename_axis(index="disease", columns="cell_tissue")
                  .reset_index().melt(id_vars="disease", var_name="cell_tissue", value_name=name))
    long = melt(T, "t").merge(melt(S, args.sig_col), on=["disease", "cell_tissue"])
    if M is not None:
        long = long.merge(melt(M.reindex(index=T.index, columns=T.columns), "pval_maxT_one_side"),
                          on=["disease", "cell_tissue"], how="left")
    long = long.dropna(subset=["t"])
    long["specificity_z"] = (long["t"] - long["cell_tissue"].map(med)) / long["cell_tissue"].map(mad)
    long["specificity_p"] = stats.norm.sf(long["specificity_z"])
    long["specificity_fdr"] = multipletests(long["specificity_p"], method="fdr_bh")[1]
    long["sig_within_disease"] = long[args.sig_col] <= args.thres
    long["generic_cell_type"] = long["cell_tissue"].map(rec["generic"]).astype(bool)

    sig = long["sig_within_disease"]
    long["hit_class"] = np.select(
        [sig & (long["specificity_fdr"] <= args.thres), sig & long["generic_cell_type"], sig],
        ["disease_specific", "generic", "significant"], default="none")
    long = long.sort_values(["disease", "specificity_fdr"])
    long.to_csv(f"{args.save_path}/disease_specific_hits.tsv", sep="\t", index=False)
    logging.info("hit_class counts:\n" + long["hit_class"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=str,
                        help="Folder containing one <disease>/univar_regression_results.tsv per disease")
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--stat_col", type=str, default="tval_predictor")
    parser.add_argument("--sig_col", type=str, default="fdr_one_side_predictor",
                        help="Within-disease significance column (e.g. fdr_one_side_predictor, fdr_perm_one_side, pval_maxT_one_side)")
    parser.add_argument("--thres", type=float, default=0.05)
    parser.add_argument("--generic_frac", type=float, default=0.5,
                        help="Cell types significant in more than this fraction of diseases are flagged generic")
    parser.add_argument("--min_diseases", type=int, default=20)
    args = parser.parse_args()
    main(args)
