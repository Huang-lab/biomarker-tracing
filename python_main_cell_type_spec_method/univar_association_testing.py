"""
Univariate association between cell-type specificity and protein disease statistics.

For one disease, fits one OLS per cell type:

    disease_effect ~ specificity_of_that_cell_type ( + optional covariates )

and reports, per cell type, the specificity coefficient (beta), its t-value,
two-sided and one-sided p-values, BH-FDR-adjusted p-values, model R^2, and a
Shapiro residual-normality p-value. Optional covariates include a Gini
specificity score (--covar_gini) and any columns in an external --covar_df.

This is the direct analogue of seismicGWAS's per-cell-type test; the tutorial
parameters are chosen so the ranking matches the seismicGWAS output.

Optionally (--n_perm > 0) a Freedman-Lane permutation test is run on top of the
OLS fits. The outcome residuals (after regressing out the intercept and any
covariates) are permuted across proteins, refit against every cell type at once,
and the observed t-statistics are compared with the permuted ones. This yields
  * pval_perm_one_side  -- per-cell-type permutation p (slope > 0),
  * pval_maxT_one_side  -- Westfall-Young max-T p, i.e. family-wise error control
                           across all cell types that accounts for their correlation.
With --perm_strata_col / --perm_n_strata the permutation is stratified: proteins are
only shuffled within quantile bins of a covariate (e.g. a pleiotropy score from
pleiotropy_score.py), so highly pleiotropic "hub" proteins only swap with other hubs
and the null preserves the pleiotropy structure of the data.

Inputs : a genes x cell-tissues specificity matrix and one disease's summary statistics.
Outputs: univar_regression_results.tsv (ranked by fdr_one_side_predictor),
         model_summary.txt, prot_spec_final*.tsv, and cmd_args.json in --save_path.
"""
import argparse, logging
import os, json
import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro, t
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import StandardScaler
from utils import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _gini_coeff(df, g):
    arr = df.loc[g, :].to_numpy()
    if np.all(arr == 0):
        return np.nan

    arr = arr[arr > 0.0]
    
    x = np.sort(arr)
    n = len(x)
    cumx = np.cumsum(x)
    
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return gini


def compute_covariates(args, atlas_smal):
    """
    atlas_smal has shape (num genes, num cell tissues)
    with the index being the gene names
    """
    gene_l = None

    # Calculate Gini coefficient
    if args.covar_gini == 1:
        logging.info("Calculating Gini scores:...")
        gene_l = atlas_smal.index.tolist()
        gini_l = [_gini_coeff(atlas_smal, g) for g in gene_l]
    
    if gene_l is not None:
        covar_df = pd.DataFrame({
            "gene": gene_l,
            "gini": gini_l
        }).set_index("gene")
        return covar_df
    else:
        return None


def _residualize(M, C):
    """Residuals of every column of M after OLS on the covariate matrix C."""
    return M - C @ np.linalg.lstsq(C, M, rcond=None)[0]


def permutation_testing(X, y, C, strata=None, n_perm=1000, seed=0, chunk=500):
    """
    Freedman-Lane permutation test of y ~ x_k + C for every column k of X at once.

    X      : (n genes, K cell types) specificity matrix (complete cases only)
    y      : (n,) outcome
    C      : (n, p) covariate matrix including the intercept column
    strata : optional (n,) integer labels; permutations happen within each label

    Uses the Frisch-Waugh-Lovell decomposition: after residualising X and y on C,
    the slope t-statistic for cell type k is c_k / sqrt((|y~|^2 - c_k^2) / df) where
    c_k = x~_k' y~ with unit-norm x~_k. Each permutation is then one matrix product,
    so all K cell types x `chunk` permutations are evaluated together.

    Returns t_obs (K,), one-sided p-values (slope > 0), two-sided p-values and
    Westfall-Young max-T one-sided p-values (family-wise across cell types).
    """
    rng = np.random.default_rng(seed)
    n, K = X.shape
    df = n - C.shape[1] - 1

    Xr = _residualize(X, C)
    Xr = Xr / np.linalg.norm(Xr, axis=0, keepdims=True)
    yr = _residualize(y[:, None], C)[:, 0]

    def tstats(E):
        # E: (n, B) matrix of (re-residualised) permuted outcomes -> (K, B) t-statistics
        c = Xr.T @ E
        rss = np.sum(E**2, axis=0, keepdims=True) - c**2
        return c / np.sqrt(np.clip(rss, 1e-300, None) / df)

    t_obs = tstats(yr[:, None])[:, 0]

    if strata is None:
        strata = np.zeros(n, dtype=int)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]

    ge_one = np.zeros(K)          # #perms with t_null >= t_obs
    ge_two = np.zeros(K)          # #perms with |t_null| >= |t_obs|
    ge_max = np.zeros(K)          # #perms with max_k t_null >= t_obs  (max-T)
    done = 0
    while done < n_perm:
        B = min(chunk, n_perm - done)
        E = np.empty((n, B))
        for idx in groups:
            # independent within-stratum shuffles for each of the B permutations
            keys = rng.random((len(idx), B))
            order = np.argsort(keys, axis=0)
            E[idx, :] = yr[idx][order]
        E = _residualize(E, C)        # Freedman-Lane: permuted residuals must stay orthogonal to C
        T = tstats(E)
        ge_one += (T >= t_obs[:, None]).sum(axis=1)
        ge_two += (np.abs(T) >= np.abs(t_obs)[:, None]).sum(axis=1)
        ge_max += (T.max(axis=0)[None, :] >= t_obs[:, None]).sum(axis=1)
        done += B

    p_one = (1 + ge_one) / (n_perm + 1)
    p_two = (1 + ge_two) / (n_perm + 1)
    p_max = (1 + ge_max) / (n_perm + 1)
    return t_obs, p_one, p_two, p_max


def univariate_testing(args, atlas_smal, prot_spec_final, covar_df=None):
    """
    Make sure atlas_smal, prot_spec_final, and covar_df both have gene as index
    """
    with open(f"{args.save_path}/model_summary.txt", "a") as f:
        f.write("Linear model summary: \n")

    # do some prep
    col = "HR"
    if col not in prot_spec_final.columns: col = "OR"
    prot_spec_final, _, _ = prep_data(args, prot_spec_final[[col, f"log{col}", "P_value"]], col)

    # atlas_smal has shape = (num genes, num cell types)
    obj = StandardScaler(with_std=False)
    if (args.ztransform_type == 1): scaled = obj.fit_transform(atlas_smal)
    elif (args.ztransform_type == 2): scaled = obj.fit_transform(atlas_smal.T).T
    else: scaled = atlas_smal.to_numpy()
    result_df = pd.DataFrame(scaled, columns=atlas_smal.columns, index=atlas_smal.index)

    # Because of python string format, replace all . with _
    old_col_names = result_df.columns.tolist()
    new_col_names = [i.replace(".", "_") for i in result_df.columns.tolist()]
    result_df.columns = new_col_names
    cell_tis = new_col_names

    # Add covariates if any
    covar_cols = []
    if covar_df is not None:
        result_df = result_df.merge(covar_df, right_index=True, left_index=True)
        covar_cols = covar_df.columns.tolist()
    prot_df_sub = prot_spec_final[[args.output_label]].copy()  # .copy() avoids SettingWithCopyWarning
    if args.abs_hr == 1:
        prot_df_sub[args.output_label] = np.abs(prot_df_sub[args.output_label])
    result_df = result_df.merge(prot_df_sub, right_index=True, left_index=True)

    logging.info(covar_cols)
    logging.info(covar_df)

    # Loop over each cell-tissue in the dataset
    final_df = []
    for old_ct, ct in zip(old_col_names, cell_tis):

        indep_var = ct
        all_cols = covar_cols + [indep_var, args.output_label]
        logging.info(f"Processing {ct}")
        result_df_ct = result_df[all_cols].dropna()

        covariates = covar_cols
        formula = args.output_label + " ~ " + " + ".join([indep_var] + covariates)

        # Linear regression
        model = smf.ols(formula, data=result_df_ct).fit()
        with open(f"{args.save_path}/model_summary.txt", "a") as f:
            f.write(f"Linear model for {old_ct}\n")
            f.write(model.summary().as_text())
            f.write("\n===============================\n")

        # Residual normality test
        shapiro_p = shapiro(model.resid)[1]

        # Extract coefficients
        coeffs = model.params
        tvals = model.tvalues
        pvals = model.pvalues

        beta_indep = coeffs[indep_var]
        tval_indep = tvals[indep_var]
        pval_indep = pvals[indep_var]

        intc = coeffs["Intercept"]
        tval_intc = tvals["Intercept"]
        pval_intc = pvals["Intercept"]

        r2 = model.rsquared

        # One-sided test
        df_resid = model.df_resid
        # pval_one_sided = 1 - t.cdf(tval_indep, df=df_resid)       # This is the original implementation
        p_two_sided = 2 * (1 - t.cdf(abs(tval_indep), df=df_resid))         # # This is the implementation to match the results from seismic
        if beta_indep > 0: pval_one_sided = p_two_sided / 2
        else: pval_one_sided = 1 - p_two_sided / 2

        # Append
        final_df.append([old_ct, beta_indep, pval_indep, tval_indep, intc, pval_intc, tval_intc, r2, shapiro_p, pval_one_sided])

    ### Save results
    final_df = pd.DataFrame(final_df, columns=[
        "cell_tissue", "beta_predictor", "pval_predictor", "tval_predictor",
        "intercept", "pval_intercept", "tval_intercept", "r2_score",
        "residual_pval", "pval_predictor_one_side"]
    )

    # FDR correction
    final_df["fdr_predictor"] = multipletests(final_df["pval_predictor"], method="fdr_bh")[1]

    # FDR correction one-side
    final_df.loc[:, "fdr_one_side_predictor"] = multipletests(final_df["pval_predictor_one_side"], method="fdr_bh")[1]

    ### Permutation test (optional)
    if args.n_perm > 0:
        logging.info(f"Running {args.n_perm} Freedman-Lane permutations "
                     f"(strata: {args.perm_strata_col}, n_strata={args.perm_n_strata})...")
        # Complete cases across all cell types so that one permutation serves every cell type.
        # (With no NAs in the atlas this is exactly the gene set used by each OLS above.)
        perm_df = result_df[cell_tis + covar_cols + [args.output_label]].dropna()
        if len(perm_df) < len(result_df):
            logging.warning(f"Permutation uses {len(perm_df)} complete-case genes vs "
                            f"{len(result_df)} in the per-cell-type OLS fits")
        X = perm_df[cell_tis].to_numpy(float)
        y = perm_df[args.output_label].to_numpy(float)
        C = np.column_stack([np.ones(len(perm_df))] + [perm_df[c].to_numpy(float) for c in covar_cols])

        strata = None
        if args.perm_strata_col not in (None, "None") and args.perm_n_strata > 1:
            if args.perm_strata_col not in perm_df.columns:
                raise ValueError(f"--perm_strata_col '{args.perm_strata_col}' is not a covariate column "
                                 f"(available: {covar_cols})")
            strata = pd.qcut(perm_df[args.perm_strata_col].rank(method="first"),
                             args.perm_n_strata, labels=False).to_numpy()

        t_obs, p_one, p_two, p_max = permutation_testing(
            X, y, C, strata=strata, n_perm=args.n_perm, seed=args.perm_seed
        )
        # Sanity check: the vectorised t must reproduce the statsmodels t
        t_ols = final_df.set_index("cell_tissue").loc[old_col_names, "tval_predictor"].to_numpy()
        if not np.allclose(t_obs, t_ols, rtol=1e-6, atol=1e-6):
            logging.warning("Vectorised permutation t-statistics differ from OLS t-statistics "
                            f"(max abs diff {np.max(np.abs(t_obs - t_ols)):.3g}); check for NA genes")
        perm_res = pd.DataFrame({
            "cell_tissue": old_col_names,
            "pval_perm_one_side": p_one,
            "pval_perm_two_side": p_two,
            "pval_maxT_one_side": p_max,
        })
        perm_res["fdr_perm_one_side"] = multipletests(perm_res["pval_perm_one_side"], method="fdr_bh")[1]
        final_df = final_df.merge(perm_res, on="cell_tissue", how="left")

    final_df = final_df.sort_values("fdr_one_side_predictor")

    # Write output
    final_df.to_csv(
        os.path.join(args.save_path, f"univar_regression_results.tsv"),
        sep="\t", index=False
    )
    prot_df_sub.to_csv(
        os.path.join(args.save_path, "prot_spec_final_sub.tsv"),
        sep="\t"
    )


def main(args):

    if args.covar_df == "None": args.covar_df = None

    # Save the arguments
    os.makedirs(args.save_path, exist_ok=True)
    args_dict = vars(args)
    with open(f"{args.save_path}/cmd_args.json", "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

    # Load in the atlas data
    atlas_smal = pd.read_csv(args.atlas_smal_path, sep="\t").set_index("gene")

    # Load in prot data
    # Dispatches on the file's own columns: a `gene` column means pre-mapped
    # Ensembl IDs and is read directly; otherwise it is UK Biobank Phenome-Proteome
    # and goes through load_prot_data. A failure in either is raised, not swallowed.
    prot_spec_final = load_sumstats(args.prot_data_path, args.disease, atlas_smal)

    if "gene" in prot_spec_final.columns: prot_spec_final = prot_spec_final.set_index("gene")
    prot_spec_final.to_csv(f"{args.save_path}/prot_spec_final.tsv", sep="\t", index=False)

    # Load in covariate data
    covar_df = None
    if args.covar_df is not None: 
        covar_df = pd.read_csv(args.covar_df, sep="\t")
        if "gene" in covar_df.columns: covar_df = covar_df.set_index("gene")
    covar_df_more = compute_covariates(args, atlas_smal)
    if covar_df is None and covar_df_more is None: pass
    else: covar_df = pd.concat([covar_df, covar_df_more], axis=1).fillna(0.0)

    # Convert some argument values to bool
    args.abs_hr = args.abs_hr == 1

    # Run univariate testing
    univariate_testing(args, atlas_smal, prot_spec_final, covar_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--atlas_smal_path", type=str, help="Atlas smal path")
    parser.add_argument("--prot_data_path", type=str, help="Prot path")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--disease", type=str)
    parser.add_argument("--output_label", type=str, default="HR")
    parser.add_argument("--abs_hr", type=int, default=0,
                        help="Whether to take the absolute value of the output")
    parser.add_argument("--gene_weight_minmax", type=int, default=0, help="Whether to minmax gene weight")

    parser.add_argument("--covar_df", type=str, default=None)
    parser.add_argument("--covar_gini", type=int, default=0)
    parser.add_argument("--ztransform_type", type=int, default=1, help="Whether to z transform on each cell type (1) or each gene (2) or no zscore (-1)")

    parser.add_argument("--n_perm", type=int, default=0,
                        help="Number of Freedman-Lane permutations (0 = skip). Adds pval_perm_*, fdr_perm_one_side and pval_maxT_one_side columns")
    parser.add_argument("--perm_strata_col", type=str, default="None",
                        help="Covariate column to stratify permutations by (e.g. 'pleiotropy' from pleiotropy_score.py); 'None' = unstratified")
    parser.add_argument("--perm_n_strata", type=int, default=1,
                        help="Number of quantile bins of --perm_strata_col within which proteins are shuffled")
    parser.add_argument("--perm_seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
