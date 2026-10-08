"""
Univariate -> multivariate association pipeline (local or LSF).

For each disease listed in the YAML config this script:
  1. runs univariate regression (blocking) and waits for it,
  2. selects the cell types passing a threshold on a chosen column
     (default: fdr_one_side_predictor <= 0.05) and subsets the specificity matrix,
  3. launches the enabled multivariate methods (ElasticNet, LASSO stability
     selection, random forest) on that reduced set of cell types.

Execution mode is picked via the YAML config's `execution.mode`:
  * "local": calls each base method script
    (python_main_cell_type_spec_method/<method>.py) directly as a local
    subprocess -- no job scheduler involved.
  * "lsf" (original behavior): calls each method's `*_auto-script.py`
    launcher (config['constants']), which itself submits a `bsub` job via
    bash_scripts/<method>.sh. The univariate step blocks on its job via
    submit_job_and_wait; the multivariate steps are fire-and-forget,
    matching the original design.

Usage:
    python pipelines/pipeline_univar_to_multivar.py --yml_file pipeline_yml/univar_multivar_sample.yml
"""
import argparse, yaml, os, subprocess, logging
import pandas as pd
from utils import submit_job_and_wait

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Method save-path subfolder names, matching what each *_auto-script.py constructs.
METHOD_SUBDIR = {
    "univariate": "univar_association_testing",
    "elasticnet_kfold": "elastic_kfold_ver2",
    "stability_selection": "stability_analyses",
    "random_forest": "tree_based_methods_random_forest",
}


def method_save_path(base_save_path: str, disease_folder_name: str, method: str, dis_name: str) -> str:
    return f"{base_save_path}/{disease_folder_name}/{METHOD_SUBDIR[method]}/{dis_name}"


def univar_top_features(config, base_save_path: str, dis_name: str):
    """
    Select top univariate features
    """
    # Select the features based on some threshold
    univar_path = f"{base_save_path}/{config['inputs']['disease_folder_name']}/univar_association_testing/{dis_name}"
    if not os.path.exists(f"{univar_path}/univar_regression_results.tsv"):
        return None

    res_df = pd.read_csv(f"{univar_path}/univar_regression_results.tsv", sep="\t")
    col = config["univar_to_multivar"]["column_to_choose"]
    thres = config["univar_to_multivar"]["thres"]
    selected = res_df[res_df[col] <= thres].copy()["cell_tissue"].tolist()

    # If there is no feature left, return None
    if len(selected) == 0:
        return None

    # Create a new atlas_smal based on the selected features
    atlas_smal_path = pd.read_csv(config["inputs"]["atlas_smal_path"], sep="\t")
    logger.debug(f"Loaded atlas matrix with shape {atlas_smal_path.shape}")
    atlas_df_sub = atlas_smal_path[selected + ["gene"]]
    atlas_df_sub.to_csv(
        f"{univar_path}/atlas_smal_path_sig_cel_tis_filtered.tsv",
        sep="\t", index=False
    )
    return f"{univar_path}/atlas_smal_path_sig_cel_tis_filtered.tsv"


def main(args):
    """
    The pipeline from univariate regression to multivariate regression
    """
    # Load the YAML file
    logger.info(f"Loading yml file...")
    with open(args.yml_file, "r") as f:
        config = yaml.safe_load(f)

    exec_mode = config.get("execution", {}).get("mode", "local")
    logger.info(f"Execution mode: {exec_mode}")

    # Some preprocessing and parsing
    logger.info(f"Preprocessing and parsing paths...")
    base_path = f"{config['inputs']['disease_prot_dir']}/{config['inputs']['disease_folder_name']}"
    diseases = os.listdir(base_path)
    if config['inputs']["save_path_suffix"] != "": save_path_suffix = f"_{config['inputs']['save_path_suffix']}"
    else: save_path_suffix = config['inputs']["save_path_suffix"]
    base_save_path = f"{config['inputs']['save_path']}{save_path_suffix}"
    os.makedirs(base_save_path, exist_ok=True)

    # Optional: per-protein pleiotropy scores computed once from ALL diseases in the sumstats
    # folder (leave-one-chapter-out). They are then passed to the univariate step as a
    # per-disease covariate and/or as the stratification variable of the permutation null.
    univ_cfg = config["univariate"]
    pleio_cfg = univ_cfg.get("pleiotropy", {}) or {}
    pleio_dir = None
    if pleio_cfg.get("run", 0) == 1:
        pleio_dir = pleio_cfg.get("save_dir") or f"{base_save_path}/{config['inputs']['disease_folder_name']}/pleiotropy_scores"
        logger.info(f">>> Computing pleiotropy scores into {pleio_dir}")
        command = ["python", "python_main_cell_type_spec_method/pleiotropy_score.py",
                   "--prot_data_path", base_path,
                   "--atlas_smal_path", config['inputs']['atlas_smal_path'],
                   "--save_path", pleio_dir,
                   "--score", str(pleio_cfg.get("score", "mean_z")),
                   "--exclude_same_category", str(pleio_cfg.get("exclude_same_category", 1))]
        result = subprocess.run(command)
        if result.returncode != 0:
            logger.error(f">>> pleiotropy_score.py failed (exit code {result.returncode}); continuing without it")
            pleio_dir = None
    elif pleio_cfg.get("save_dir"):
        pleio_dir = pleio_cfg["save_dir"]   # precomputed scores

    n_perm = str(univ_cfg.get("n_perm", 0))
    perm_n_strata = str(univ_cfg.get("perm_n_strata", 1))
    perm_seed = str(univ_cfg.get("perm_seed", 0))
    perm_strata_col = str(univ_cfg.get("perm_strata_col", "pleiotropy" if pleio_dir else "None"))

    # Loop through the diseases
    for disease in sorted(diseases):

        # Check if disease is in yml file
        if len(disease.split(".")) == 2:
            dis_name = disease.split(".")[0]
        else:
            dis_name = ".".join(disease.split(".")[:-1])

        # Check if running on correct disease - if keyword "all", then run all diseases
        if config['inputs']["disease_name"][0] == "all": pass
        elif dis_name not in config['inputs']["disease_name"] : continue
        else: pass

        logger.info(f"Start running on {dis_name}...")

        # Run univariate regression (if run is not set, then by default it is run)
        if (config["univariate"].get("run", 1) == 1):
            logger.info(">>> Start running univariate regression")
            logger.info(f">>> Univariate params: {config['univariate']}")

            # Per-disease pleiotropy covariate takes precedence over a single shared covar_df
            covar_df = config['univariate']['covar_df']
            if pleio_dir is not None:
                pleio_file = f"{pleio_dir}/{dis_name}.tsv"
                if os.path.exists(pleio_file): covar_df = pleio_file
                else: logger.warning(f">>> No pleiotropy score file for {dis_name}; falling back to covar_df={covar_df}")
            perm_args = [
                "--n_perm", n_perm,
                "--perm_strata_col", perm_strata_col if covar_df != "None" else "None",
                "--perm_n_strata", perm_n_strata,
                "--perm_seed", perm_seed,
            ]

            if exec_mode == "local":
                save_path = method_save_path(base_save_path, config['inputs']['disease_folder_name'], "univariate", dis_name)
                os.makedirs(save_path, exist_ok=True)
                sub_args = [
                    "--atlas_smal_path", config['inputs']['atlas_smal_path'],
                    "--prot_data_path", base_path,
                    "--save_path", save_path,
                    "--disease", dis_name,
                    "--output_label", config['univariate']['output_label'],
                    "--abs_hr", str(config['univariate']['abs_hr']),
                    "--covar_df", covar_df,
                    "--covar_gini", str(config['univariate']['covar_gini']),
                    "--ztransform_type", str(config['univariate']['ztransform_type'])
                ] + perm_args
                command = ["python", "python_main_cell_type_spec_method/univar_association_testing.py"] + sub_args
                result = subprocess.run(command)
                if result.returncode != 0:
                    logger.error(f">>> Something went wrong for univariate regression! (exit code {result.returncode})")
            else:
                sub_args = [
                    "--atlas_smal_path", config['inputs']['atlas_smal_path'],
                    "--disease_prot_dir", config['inputs']['disease_prot_dir'],
                    "--disease_folder_name", config['inputs']['disease_folder_name'],
                    "--save_path", base_save_path,
                    "--save_path_suffix", "",
                    "--disease_name", dis_name,
                    "--output_label", config['univariate']['output_label'],
                    "--abs_hr", str(config['univariate']['abs_hr']),
                    "--covar_df", covar_df,
                    "--covar_gini", str(config['univariate']['covar_gini']),
                    "--ztransform_type", str(config['univariate']['ztransform_type'])
                ] + perm_args
                command = ["python", config['constants']["univar_script_path"]] + sub_args
                try:
                    submit_job_and_wait(command, wait_time=10)
                except Exception as e:
                    # Something wrong with the run, then continue
                    logger.error(f">>> Something went wrong for univariate regression! Error log: {e}")

            # Extract the atlas_smal from the significant features found by univariate
            logger.info(">>> Extract significant features from univariate regression...")
            new_atlas_smal = univar_top_features(config, base_save_path, dis_name)
        else:
            logger.info(">>> Univariate regression skipped, using the full dataset")
            new_atlas_smal = config['inputs']['atlas_smal_path']

        # If new_atlas_smal is None, then there is no significant feature
        if new_atlas_smal is None:
            logger.warning(f">>> No significant features given the current univariate threshold, or univariate did not run successfully!")
            continue

        # Run elasticnet
        if (config["elasticnet_kfold"]["run"] == 1):
            logger.info(">>> Start running elasticnet...")
            logger.info(f">>> Elasticnet params: {config['elasticnet_kfold']}")

            if exec_mode == "local":
                save_path = method_save_path(base_save_path, config['inputs']['disease_folder_name'], "elasticnet_kfold", dis_name)
                os.makedirs(save_path, exist_ok=True)
                sub_args = [
                    "--atlas_smal_path", new_atlas_smal,
                    "--prot_data_path", base_path,
                    "--save_path", save_path,
                    "--disease", dis_name,
                    "--output_label", config['elasticnet_kfold']['output_label'],
                    "--abs_hr", str(config['elasticnet_kfold']['abs_hr']),
                    "--num_alphas", str(config['elasticnet_kfold']['num_alpha']),
                    "--num_folds", str(config['elasticnet_kfold']['num_folds']),
                    "--gene_weight", str(config['elasticnet_kfold']['gene_weight']),
                    "--ztransform_type", str(config['elasticnet_kfold']['ztransform_type']),
                    "--pos_coef", str(config['elasticnet_kfold']['pos_coef'])
                ]
                command = ["python", "python_main_cell_type_spec_method/elastic_kfold_ver2.py"] + sub_args
                subprocess.run(command)
            else:
                sub_args = [
                    "--atlas_smal_path", new_atlas_smal,
                    "--disease_prot_dir", config['inputs']['disease_prot_dir'],
                    "--disease_folder_name", config['inputs']['disease_folder_name'],
                    "--save_path", base_save_path,
                    "--save_path_suffix", "",
                    "--disease_name", dis_name,
                    "--output_label", config['elasticnet_kfold']['output_label'],
                    "--abs_hr", str(config['elasticnet_kfold']['abs_hr']),
                    "--num_alpha", str(config['elasticnet_kfold']['num_alpha']),
                    "--num_folds", str(config['elasticnet_kfold']['num_folds']),
                    "--gene_weight", str(config['elasticnet_kfold']['gene_weight']),
                    "--ztransform_type", str(config['elasticnet_kfold']['ztransform_type']),
                    "--pos_coef", str(config['elasticnet_kfold']['pos_coef'])
                ]
                command = ["python", config['constants']["enet_script_path"]] + sub_args
                subprocess.run(command)

        # Run Lasso stability selection
        if (config["stability_selection"]["run"] == 1):
            logger.info(">>> Start running stability selection...")
            logger.info(f">>> Stability selection params: {config['stability_selection']}")

            if exec_mode == "local":
                save_path = method_save_path(base_save_path, config['inputs']['disease_folder_name'], "stability_selection", dis_name)
                os.makedirs(save_path, exist_ok=True)
                sub_args = [
                    "--atlas_smal_path", new_atlas_smal,
                    "--prot_data_path", base_path,
                    "--save_path", save_path,
                    "--disease", dis_name,
                    "--output_label", config['stability_selection']['output_label'],
                    "--abs_hr", str(config['stability_selection']['abs_hr']),
                    "--thres", str(config['stability_selection']['thres']),
                    "--ztransform_type", str(config['stability_selection']['ztransform_type'])
                ]
                command = ["python", "python_main_cell_type_spec_method/stability_analyses.py"] + sub_args
                subprocess.run(command)
            else:
                sub_args = [
                    "--atlas_smal_path", new_atlas_smal,
                    "--disease_prot_dir", config['inputs']['disease_prot_dir'],
                    "--disease_folder_name", config['inputs']['disease_folder_name'],
                    "--save_path", base_save_path,
                    "--save_path_suffix", "",
                    "--disease_name", dis_name,
                    "--output_label", config['stability_selection']['output_label'],
                    "--abs_hr", str(config['stability_selection']['abs_hr']),
                    "--thres", str(config['stability_selection']['thres']),
                    "--ztransform_type", str(config['stability_selection']['ztransform_type'])
                ]
                command = ["python", config['constants']["stab_sele_script_path"]] + sub_args
                subprocess.run(command)

        # Run random forests
        if (config["random_forest"]["run"] == 1):
            logger.info(">>> Start running random_forest...")
            logger.info(f">>> Random_forest params: {config['random_forest']}")

            if exec_mode == "local":
                save_path = method_save_path(base_save_path, config['inputs']['disease_folder_name'], "random_forest", dis_name)
                os.makedirs(save_path, exist_ok=True)
                sub_args = [
                    "--atlas_smal_path", new_atlas_smal,
                    "--prot_data_path", base_path,
                    "--save_path", save_path,
                    "--disease", dis_name,
                    "--output_label", config['random_forest']['output_label'],
                    "--abs_hr", str(config['random_forest']['abs_hr']),
                    "--param_search", str(config['random_forest']['param_search']),
                    "--num_trees", str(config['random_forest']['num_trees']),
                    "--min_samples_split", str(config['random_forest']['min_samples_split']),
                    "--min_samples_leaf", str(config['random_forest']['min_samples_leaf']),
                    "--max_samples", str(config['random_forest']['max_samples']),
                    "--kfold_n", str(config['random_forest']['kfold_n']),
                    "--n_permute_repeat", str(config['random_forest']['n_permute_repeat']),
                    "--ztransform_type", str(config['random_forest']['ztransform_type'])
                ]
                command = ["python", "python_main_cell_type_spec_method/tree_based_methods.py"] + sub_args
                subprocess.run(command)
            else:
                sub_args = [
                    "--atlas_smal_path", new_atlas_smal,
                    "--disease_prot_dir", config['inputs']['disease_prot_dir'],
                    "--disease_folder_name", config['inputs']['disease_folder_name'],
                    "--save_path", base_save_path,
                    "--save_path_suffix", "",
                    "--disease_name", dis_name,
                    "--output_label", config['random_forest']['output_label'],
                    "--abs_hr", str(config['random_forest']['abs_hr']),
                    "--param_search", str(config['random_forest']['param_search']),
                    "--num_trees", str(config['random_forest']['num_trees']),
                    "--min_samples_split", str(config['random_forest']['min_samples_split']),
                    "--min_samples_leaf", str(config['random_forest']['min_samples_leaf']),
                    "--max_samples", str(config['random_forest']['max_samples']),
                    "--kfold_n", str(config['random_forest']['kfold_n']),
                    "--n_permute_repeat", str(config['random_forest']['n_permute_repeat']),
                    "--ztransform_type", str(config['random_forest']['ztransform_type'])
                ]
                command = ["python", config['constants']["rf_script_path"]] + sub_args
                subprocess.run(command)

    # Cross-disease summary: separate disease-specific cell-type hits from cell types that are
    # significant for most diseases. Only meaningful when many diseases were run.
    cd_cfg = config.get("cross_disease", {}) or {}
    if cd_cfg.get("run", 0) == 1:
        results_dir = f"{base_save_path}/{config['inputs']['disease_folder_name']}/{METHOD_SUBDIR['univariate']}"
        cd_save = f"{base_save_path}/{config['inputs']['disease_folder_name']}/cross_disease_summary"
        logger.info(f">>> Running cross-disease summary into {cd_save}")
        command = ["python", "pipelines/cross_disease_summary.py",
                   "--results_dir", results_dir,
                   "--save_path", cd_save,
                   "--sig_col", str(cd_cfg.get("sig_col", config["univar_to_multivar"]["column_to_choose"])),
                   "--thres", str(cd_cfg.get("thres", config["univar_to_multivar"]["thres"])),
                   "--generic_frac", str(cd_cfg.get("generic_frac", 0.5)),
                   "--min_diseases", str(cd_cfg.get("min_diseases", 20))]
        result = subprocess.run(command)
        if result.returncode != 0:
            logger.error(f">>> cross_disease_summary.py failed (exit code {result.returncode})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_file", required=True, type=str, help="Path to yml file")

    args = parser.parse_args()
    main(args)
