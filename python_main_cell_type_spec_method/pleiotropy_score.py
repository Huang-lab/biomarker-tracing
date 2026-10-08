"""
Per-protein pleiotropy scores from a folder of disease summary statistics.

Many plasma proteins (e.g. GDF15, NEFL, WFDC2, PLAUR, EDA2R) are associated with
hundreds of diseases. Any cell type that happens to express these "hub" proteins
therefore looks disease-relevant for almost every disease. This script quantifies
how pleiotropic each protein is so that the univariate test can condition on it
(--covar_df) and/or stratify its permutation null by it (--perm_strata_col).

For every disease d in --prot_data_path it writes <save_path>/<d>.tsv with columns
`gene` and `pleiotropy`, where the score is computed from the OTHER diseases only:
  * by default all diseases in the same `Disease_category` (ICD chapter, present in
    the UK Biobank Proteome-Phenome Atlas format) as d are excluded as well, so that
    e.g. the other liver diseases do not leak d's own signal into its covariate;
  * --score mean_z uses the mean signed z-score (default); mean_abs_z uses |z|.

Also written: z_matrix.tsv (proteins x diseases), disease_categories.tsv and
hub_proteins.tsv (per protein: number of diseases with P < --hub_pval, mean |z|).

Usage:
    python python_main_cell_type_spec_method/pleiotropy_score.py \
        --prot_data_path <folder of <disease>.csv> \
        --atlas_smal_path <specificity matrix .tsv> \
        --save_path <output folder>
"""
import argparse, logging, os, json
import numpy as np
import pandas as pd
from scipy import stats

from utils import load_prot_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_one_disease(prot_data_path, disease, atlas):
    """Mirror the loading logic of the method scripts and return (z Series indexed by gene, category)."""
    try:
        df = load_prot_data(prot_data_path, disease, atlas)
    except Exception:
        # Pre-mapped custom format (may still carry a Disease_category column)
        df = pd.read_csv(f"{os.path.join(prot_data_path, disease)}.csv")
    if "gene" not in df.columns:
        raise ValueError(f"{disease}: no `gene` column after loading")
    category = str(df["Disease_category"].iloc[0]) if "Disease_category" in df.columns else "unknown"
    df = df.drop_duplicates(subset="gene").set_index("gene")

    col = "HR" if "HR" in df.columns else "OR"
    p = df["P_value"].clip(lower=1e-300)
    sign = np.where(df[col] > 1, 1.0, -1.0)
    z = pd.Series(sign * stats.norm.isf(p / 2), index=df.index, name=disease)
    return z, category


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    with open(f"{args.save_path}/cmd_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    atlas = pd.read_csv(args.atlas_smal_path, sep="\t").set_index("gene")

    diseases = sorted(f[:-4] for f in os.listdir(args.prot_data_path) if f.endswith(".csv"))
    logging.info(f"Found {len(diseases)} diseases")
    if len(diseases) < 20:
        logging.warning("Fewer than 20 diseases: pleiotropy scores will be noisy and conditioning "
                        "on them mostly removes shared signal between related diseases")

    zs, cats = [], {}
    for d in diseases:
        try:
            z, cat = load_one_disease(args.prot_data_path, d, atlas)
        except Exception as e:
            logging.warning(f"Skipping {d}: {e}")
            continue
        zs.append(z)
        cats[d] = cat
    Z = pd.concat(zs, axis=1)
    Z = Z.loc[Z.index.intersection(atlas.index)]
    logging.info(f"z matrix: {Z.shape[0]} proteins x {Z.shape[1]} diseases")

    Z.to_csv(f"{args.save_path}/z_matrix.tsv", sep="\t")
    pd.Series(cats, name="Disease_category").rename_axis("disease").to_csv(
        f"{args.save_path}/disease_categories.tsv", sep="\t")

    # Hub-protein table: how many diseases each protein is associated with
    P = 2 * stats.norm.sf(Z.abs())
    hubs = pd.DataFrame({
        "n_diseases_tested": Z.notna().sum(axis=1),
        f"n_diseases_p_lt_{args.hub_pval:g}": (P < args.hub_pval).sum(axis=1),
        "mean_abs_z": Z.abs().mean(axis=1),
        "mean_z": Z.mean(axis=1),
    }, index=Z.index).sort_values(f"n_diseases_p_lt_{args.hub_pval:g}", ascending=False)
    hubs.rename_axis("gene").to_csv(f"{args.save_path}/hub_proteins.tsv", sep="\t")
    logging.info("Top hub proteins:\n" + hubs.head(10).to_string())

    # Per-disease leave-one-(category)-out score
    M = Z.abs() if args.score == "mean_abs_z" else Z
    for d in Z.columns:
        if args.exclude_same_category == 1 and cats[d] != "unknown":
            others = [o for o in Z.columns if cats[o] != cats[d]]
        else:
            others = [o for o in Z.columns if o != d]
        if len(others) == 0:
            logging.warning(f"{d}: no other diseases left to compute a pleiotropy score; skipping")
            continue
        score = M[others].mean(axis=1, skipna=True)
        out = pd.DataFrame({"gene": score.index, "pleiotropy": score.values}).dropna()
        out.to_csv(f"{args.save_path}/{d}.tsv", sep="\t", index=False)
    logging.info(f"Wrote per-disease covariate files to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prot_data_path", required=True, type=str, help="Folder with one <disease>.csv per disease")
    parser.add_argument("--atlas_smal_path", required=True, type=str, help="Specificity matrix; used to restrict to atlas genes")
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--score", type=str, default="mean_z", choices=["mean_z", "mean_abs_z"])
    parser.add_argument("--exclude_same_category", type=int, default=1,
                        help="1 = exclude all diseases sharing the target's Disease_category (ICD chapter), not just the target")
    parser.add_argument("--hub_pval", type=float, default=1e-5, help="P threshold used to count diseases per protein in hub_proteins.tsv")
    args = parser.parse_args()
    main(args)
