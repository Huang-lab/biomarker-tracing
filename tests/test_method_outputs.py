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
    "stability_selection": ("stability_analyses.py", ["--thres", "0.6"]),
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
    # Fixed hash seed: elastic_kfold_ver2.py orders proteins with list(set(...)), whose order changes
    # between Python processes, so the seeded k-fold split (and the chosen best model) otherwise varies run to run
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=env)
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


def test_univariate(tmp_path):
    run_method("univariate", tmp_path)
    check_against_expected("univariate", tmp_path, "univar_regression_results.tsv", sort_by="cell_tissue")


def test_elastic_net(tmp_path):
    run_method("elastic_net", tmp_path)
    # Coefficients of the models refitted on all proteins (best by R^2, by Pearson r and by MSE),
    # and which alpha and l1_ratio each of them used
    check_against_expected("elastic_net", tmp_path, "best_model_full_data.tsv", sort_by="model_run")
    check_against_expected("elastic_net", tmp_path, "coef_best_model_full_data.tsv", sort_by=None)


def test_random_forest(tmp_path):
    run_method("random_forest", tmp_path)
    check_against_expected("random_forest", tmp_path, "permute_importance_scores.tsv",
                           sort_by=["fold", "cell_tissue"])


@pytest.mark.xfail(strict=True, reason="Known bugs on master: stability-selection needs a sklearn.linear_model.base "
                   "shim that is missing, and once that is added the fixed penalty grid (alpha 0.1 to 1) selects no "
                   "cell type. The expected output is added together with the fix.")
def test_stability_selection(tmp_path):
    run_method("stability_selection", tmp_path)
    check_against_expected("stability_selection", tmp_path, "selected_features_thres_0.6.tsv",
                           sort_by="selected_features")
