# Tutorial: from a single-cell atlas to disease-relevant cell types

This is a small, self-contained walkthrough of the full `biomarker-tracing` workflow using a tiny slice of the Human Protein Atlas (HPA) single-cell data and one disease (alcoholic liver disease). It has two parts:

1. **Part 1 (R):** compute a `seismic` cell-type specificity matrix from a single-cell Seurat object.
2. **Part 2 (Python):** run the association pipeline that links each protein's disease statistics to that specificity matrix.

Everything you need is in this `tutorial/` folder, and a complete set of reference outputs is provided under `tutorial/pipeline_univar_multivar_sample_run/` so you can check your results.

> ⚠️ **Platform note.** Part 1 runs anywhere R and its packages are installed. **Part 2 currently runs only on an LSF HPC cluster** (it submits `bsub` jobs). To run Part 2 elsewhere, call the base method scripts directly — see "Running without LSF" in the [root README](../README.md).

## Prerequisites

- **Part 1:** R with the packages in [`../r_packages.txt`](../r_packages.txt) (notably `seismicGWAS`, `Seurat`, `qs`, `dplyr`, `data.table`).
- **Part 2:** the conda environment in [`../python_packages.yml`](../python_packages.yml):
  ```bash
  conda env create -f ../python_packages.yml   # creates "scanpy-env"
  conda activate scanpy-env
  ```

Install these before starting. To use your own summary statistics instead of the bundled `sample_disease/Alcoholic_liver_disease.csv`, format the file like `../sample_data/sample_sumstats/AD_meta_sumstats_all.csv` (columns `gene`, `P_value`, `OR`/`HR`, `logOR`/`logHR`, with `gene` in the same identifier space as the atlas).

## Part 1 — cell-type specificity matrix (R)

Run [`../cell_tissue_specificity/hpa_seismic_cell_type_specificity.R`](../cell_tissue_specificity/hpa_seismic_cell_type_specificity.R). It reads the bundled example Seurat object `human_protein_atlas_single_cell_small.rds` (a tiny subset of the full HPA single-cell RNA-seq atlas), runs `seismicGWAS::calc_specificity`, and writes a `genes × cell-tissues` specificity matrix.

```bash
Rscript ../cell_tissue_specificity/hpa_seismic_cell_type_specificity.R
```

Expected output: a TSV that looks like the provided `human_protein_atlas_seismic_cell_spec_matrix.tsv` (rows = genes, columns = cell-tissue types, plus a `gene` column).

> `../cell_tissue_specificity/hpa_data_preprocessing.R` shows how the tiny `.rds` was derived from the full atlas (downsampling, variable-gene selection). You do not need to run it for this tutorial.

## Part 2 — the association pipeline (Python, LSF)

Two steps:

**1. Configure.** Use the provided [`univar_multivar_sample.yml`](univar_multivar_sample.yml) to reproduce the reference results.

> The `univariate` hyperparameters are set so the univariate output matches the original `seismicGWAS` package. The multivariate sections (ElasticNet, stability selection, random forest) are less constrained — feel free to experiment. In our experience the univariate ranking is already biologically meaningful on its own.

**2. Run.** Submit the pipeline with [`run_pipeline_part2.sh`](run_pipeline_part2.sh):

```bash
bsub < run_pipeline_part2.sh
```

> Edit the `module load anaconda3/latest; conda activate scanpy-env` line in that script to match your cluster's module system and environment name.

The pipeline runs univariate regression, selects FDR-significant cell types, and then runs the enabled multivariate methods on the reduced set. Results appear under `pipeline_univar_multivar_sample_run/sample_disease/<method>/Alcoholic_liver_disease/`.

## Checking your results

Compare against the reference outputs already in this folder:

```
pipeline_univar_multivar_sample_run/sample_disease/
├── univar_association_testing/…/univar_regression_results.tsv   # per cell-type stats, ranked by FDR
├── elastic_kfold_ver2/…/coef_best_model_full_data.tsv           # ElasticNet coefficients
├── stability_analyses/…/selected_features_thres_0.6.tsv         # stably selected cell types
└── tree_based_methods_random_forest/…/permute_importance_scores.tsv
```

The top-ranked cell types in `univar_regression_results.tsv` (lowest `fdr_one_side_predictor`) are the tutorial's headline result: the cell types whose specificity most strongly tracks alcoholic-liver-disease protein associations.

> **Note on reference outputs.** The bundled `random_forest` and `stability_analyses` outputs were produced before two correctness fixes: the random-forest permutation-importance step now trains only on the training fold (previously it leaked the test fold), and stability selection now honors the configured `ztransform_type` (previously it was silently ignored). Re-running these two methods will therefore give slightly different numbers than the committed reference files. The univariate and ElasticNet reference outputs are unaffected.

## License

This project is released under the [MIT License](../LICENSE).
