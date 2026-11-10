"""Expected-output tests for the four association methods.

Each method is run on the tutorial disease (Alcoholic_liver_disease, 229 cell types) with fixed, small
settings and its main output is compared with a saved expected output in tests/expected_outputs/.
Univariate, elastic net and random forest are seeded, so their outputs must match to a tight tolerance.

These are the baseline for later changes: a refactor or speed-up must reproduce the expected outputs,
and a change that is meant to alter the results must regenerate them on purpose:

    UPDATE_EXPECTED=1 python -m pytest tests/test_method_outputs.py

then review the diff of tests/expected_outputs/ before committing it.

Marked "slow" (about a minute in all): skip them with `pytest -m "not slow"`.
"""
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS_DIR = os.path.join(REPO, "python_main_cell_type_spec_method")
EXPECTED_DIR = os.path.join(REPO, "tests", "expected_outputs")
ATLAS = os.path.join(REPO, "tutorial", "human_protein_atlas_seismic_cell_spec_matrix.tsv")
SUMSTATS_DIR = os.path.join(REPO, "tutorial", "sample_disease")
DISEASE = "Alcoholic_liver_disease"
UPDATE = os.environ.get("UPDATE_EXPECTED") == "1"

pytestmark = pytest.mark.slow

# The settings each method is run with. Changing them changes the expected outputs.
METHOD_ARGS = {
    "univariate": ("univar_association_testing.py",
                   ["--covar_df", "None", "--covar_gini", "0", "--n_perm", "200", "--perm_seed", "0"]),
    "elastic_net": ("elastic_kfold_ver2.py",
                    ["--num_alphas", "10", "--num_folds", "3", "--gene_weight", "0", "--pos_coef", "0"]),
    "random_forest": ("tree_based_methods.py",
                      ["--param_search", "0", "--num_trees", "50", "--min_samples_split", "10",
                       "--min_samples_leaf", "2", "--max_samples", "0.7", "--kfold_n", "3",
                       "--n_permute_repeat", "3"]),
    # Specificity z-scored per cell type (1, the script's default): the fixed penalty grid (alpha 0.1 to 1) is
    # sized for that and selects nothing on the unscaled values the other methods use here (-1)
    "stability_selection": ("stability_analyses.py", ["--thres", "0.6", "--ztransform_type", "1"]),
}


def run_method(method, save_path):
    """Run one method script on the tutorial disease, as the pipeline does in local mode."""
    script, extra_args = METHOD_ARGS[method]
    command = [
        sys.executable, os.path.join(METHODS_DIR, script),
        "--atlas_smal_path", ATLAS,
        "--prot_data_path", SUMSTATS_DIR,
        "--save_path", str(save_path),
        "--disease", DISEASE,
        "--output_label", "z_score",
        "--abs_hr", "0",
        "--ztransform_type", "-1",
        *extra_args,
    ]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, f"{script} failed:\n{result.stderr[-3000:]}"
    return save_path


def check_against_expected(method, save_path, output_file, sort_by, rtol=1e-6, atol=1e-10):
    """Compare one output table with its expected version (or write it, with UPDATE_EXPECTED=1)."""
    path = os.path.join(save_path, output_file)
    assert os.path.exists(path), f"{method} did not write {output_file}"
    new = sort_rows(pd.read_csv(path, sep="\t"), sort_by)

    expected_path = os.path.join(EXPECTED_DIR, method, output_file)
    if UPDATE:
        os.makedirs(os.path.dirname(expected_path), exist_ok=True)
        new.to_csv(expected_path, sep="\t", index=False)
        return

    assert os.path.exists(expected_path), f"no expected output; create it with UPDATE_EXPECTED=1"
    expected = sort_rows(pd.read_csv(expected_path, sep="\t"), sort_by)
    pd.testing.assert_frame_equal(new, expected, rtol=rtol, atol=atol, check_exact=False)


def sort_rows(df, sort_by):
    """Put rows in a fixed order so that row order alone never fails a comparison."""
    return (df.sort_values(sort_by) if sort_by else df).reset_index(drop=True)


def significant_cell_types(method, save_path):
    """The cell types a method calls significant, by the rule the analysis uses; returns (selected, tested)."""
    out = lambda name: pd.read_csv(os.path.join(save_path, name), sep="\t")
    if method == "univariate":      # FDR-adjusted one-sided p < 0.05, the pipeline's gate for the multivariate steps
        res = out("univar_regression_results.tsv")
        return set(res.cell_tissue[res.fdr_one_side_predictor < 0.05]), set(res.cell_tissue)
    if method == "elastic_net":     # non-zero coefficient in the model refit on all proteins (best by R^2)
        coef = out("coef_best_model_full_data.tsv").drop(columns="disease").iloc[0]
        return set(coef.index[coef != 0]), set(coef.index)
    if method == "random_forest":   # permutation importance more than 2 SD above zero across folds
        scores = out("permute_importance_scores.tsv").groupby("cell_tissue").score.agg(["mean", "std"])
        return set(scores.index[scores["mean"] - 2 * scores["std"] > 0]), set(scores.index)
    if method == "stability_selection":     # stability score at or above --thres
        return set(out("selected_features_thres_0.6.tsv").selected_features), set(out("feature_scores.tsv").feature_label)


@pytest.fixture
def check_significant(request):
    """Assert a method found some significant cell types, and record how many for the end-of-run summary."""
    def check(method, save_path):
        selected, tested = significant_cell_types(method, save_path)
        assert selected <= tested
        assert selected, f"{method} called no cell type significant, so these tests would pass even if it broke"
        request.config.significance_rows = getattr(request.config, "significance_rows", []) + [
            (method, len(selected), len(tested), sorted(selected))]
    return check


def test_univariate(tmp_path, check_significant):
    run_method("univariate", tmp_path)
    check_significant("univariate", tmp_path)
    check_against_expected("univariate", tmp_path, "univar_regression_results.tsv", sort_by="cell_tissue")


def test_elastic_net(tmp_path, check_significant):
    run_method("elastic_net", tmp_path)
    check_significant("elastic_net", tmp_path)
    # Coefficients of the models refitted on all proteins (best by R^2, by Pearson r and by MSE),
    # and which alpha and l1_ratio each of them used
    check_against_expected("elastic_net", tmp_path, "best_model_full_data.tsv", sort_by="model_run")
    check_against_expected("elastic_net", tmp_path, "coef_best_model_full_data.tsv", sort_by=None)


def test_random_forest(tmp_path, check_significant):
    run_method("random_forest", tmp_path)
    check_significant("random_forest", tmp_path)
    check_against_expected("random_forest", tmp_path, "permute_importance_scores.tsv",
                           sort_by=["fold", "cell_tissue"])


def test_stability_selection(tmp_path, check_significant):
    run_method("stability_selection", tmp_path)
    check_significant("stability_selection", tmp_path)
    # Fraction of subsamples selecting each cell type at each penalty, then the cell types kept at the threshold
    check_against_expected("stability_selection", tmp_path, "feature_scores.tsv", sort_by="feature_label")
    check_against_expected("stability_selection", tmp_path, "selected_features_thres_0.6.tsv",
                           sort_by="selected_features")
