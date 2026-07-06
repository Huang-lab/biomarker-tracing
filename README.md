# biomarker-tracing

**Tracing disease-associated proteins back to the cell types and tissues that express them.**

This codebase links two kinds of data:

1. **Cell/tissue specificity of genes** — how selectively each gene is expressed across cell types and tissues, derived from a single-cell (or pseudobulk) expression atlas.
2. **Disease association of proteins** — how strongly each protein is associated with a disease, taken from proteomic summary statistics (e.g. a proteome-wide association study reporting a hazard ratio / odds ratio and p-value per protein).

By regressing the disease-association signal of proteins onto the cell-type specificity of their genes, we ask: **which cell types and tissues are enriched for the proteins that drive a given disease?** A cell type whose specificity scores predict which proteins are disease-associated is a candidate "cell type of action" for that disease.

The approach generalizes the univariate logic of [`seismicGWAS`](https://github.com/ylaboratory/seismic) — which links GWAS genes to cell types — to proteomic summary statistics, and adds several multivariate models that jointly consider all cell types.

---

## Biological rationale

For a disease, proteins with strong effects (large |logHR|, small p-value) tend to be expressed in the tissues and cell types where the disease acts. If we treat each protein's disease-association statistic as an outcome `y` and each cell type's specificity score for that protein's gene as a predictor, then:

- A **positive, significant** association between a cell type's specificity and protein disease-effect means proteins that are specific to that cell type tend to be more disease-associated → the cell type is implicated in the disease.
- Ranking cell types by the strength of this association produces a prioritized list of disease-relevant cell types / tissues.

The univariate model does this one cell type at a time (like `seismicGWAS`); the multivariate models (ElasticNet, LASSO stability selection, random forest) do it jointly, accounting for correlation between cell types.

---

## How it works (data flow)

```
                       ┌─────────────────────────────────────┐
  single-cell atlas    │  Part 1 (R): cell-type specificity   │
  (Seurat .rds)  ─────▶│  seismicGWAS::calc_specificity       │
                       └───────────────┬─────────────────────┘
                                       │  genes × cell-tissues
                                       ▼  specificity matrix (.tsv)
  proteomic sumstats   ┌─────────────────────────────────────┐
  (per-disease .csv) ─▶│  Part 2 (Python): association models  │
                       │                                       │
                       │   1. Univariate OLS per cell type     │
                       │        │ select FDR-significant       │
                       │        ▼ cell types                   │
                       │   2. Multivariate on selected set:    │
                       │        • ElasticNet (k-fold CV)       │
                       │        • LASSO stability selection    │
                       │        • Random forest + permutation  │
                       └───────────────┬─────────────────────┘
                                       ▼
                          ranked disease-relevant cell types
```

**Part 1** turns a single-cell expression atlas into a `genes × cell-tissues` specificity matrix.
**Part 2** regresses proteomic disease statistics onto that matrix, first univariately (to select significant cell types), then multivariately (to rank them jointly). The whole of Part 2 is orchestrated by a single YAML-configured pipeline.

---

## Repository structure

| Path | Purpose |
|------|---------|
| `cell_tissue_specificity/` | **Part 1 (R).** Preprocess a single-cell atlas and compute the `seismic` specificity matrix. |
| `python_main_cell_type_spec_method/` | **Part 2 (Python).** The four association methods, each in a base script (the statistics) + an `*_auto-script.py` (LSF job launcher). Shared helpers in `utils.py`. |
| `bash_scripts/` | LSF worker scripts that activate the conda env and invoke each base method script. |
| `pipelines/` | The end-to-end pipeline (`pipeline_univar_to_multivar.py`) and its helpers. |
| `pipeline_yml/` | Example pipeline configuration (`univar_multivar_sample.yml`). |
| `tutorial/` | A runnable, self-contained example with a tiny atlas, one disease, and reference outputs. **Start here.** |
| `sample_data/` | Example atlas and summary-statistics files illustrating the required input formats. |
| `python_packages.yml` | Exact conda environment for Part 2. |
| `r_packages.txt` | `sessionInfo()` snapshot of the R environment used for Part 1. |

### The three-layer launcher pattern (Part 2)

Each method exists as three files so that a single method can be run standalone, or fanned out across many diseases on the cluster:

```
pipeline_univar_to_multivar.py          # orchestrator: reads YAML, loops over diseases
   └─ <method>_auto-script.py           # submits one bsub job per disease
        └─ bash_scripts/<method>.sh     # activates conda env inside the job
             └─ <method>.py             # the actual statistics
```

When editing the *science*, edit the base `<method>.py`. When editing *how jobs are launched* (resources, paths), edit the `*_auto-script.py` or the `.sh`.

---

## Requirements

> ⚠️ **Part 2 currently targets an LSF (IBM Spectrum) HPC cluster.** The `*_auto-script.py` launchers and `bash_scripts/*.sh` call `bsub`, `bjobs`, and `module load`, so they will not run unmodified on a local machine or a Slurm cluster. Part 1 (R) runs anywhere R and its packages are installed. Porting Part 2 to run locally requires replacing the `bsub` launch layer — see [Running without LSF](#running-without-lsf).

**Part 1 (R):** install the packages listed in `r_packages.txt` (key ones: `seismicGWAS`, `Seurat`, `qs`, `scRNAseq`, `dplyr`, `data.table`).

**Part 2 (Python):** create the conda environment from the pinned spec:

```bash
conda env create -f python_packages.yml   # creates the "scanpy-env" environment
conda activate scanpy-env
```

Key Python dependencies: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`, `kneed`, and [`stability-selection`](https://github.com/scikit-learn-contrib/stability-selection).

---

## Input data formats

### 1. Cell-type specificity / atlas matrix

A TSV where **rows are genes** and **columns are cell types or tissues**:

- One column per cell type / tissue, holding the specificity score for each gene.
- A final column named `gene` holding the gene identifier (Ensembl ID or symbol) for each row.
- Cell-tissue column names use `.` and `_` conventionally (e.g. `adipose.tissue_adipocytes`); the code internally replaces `.` with `_` where needed for formula safety.

Examples: `sample_data/sample_atlas_data/human_protein_atlas_all_tissues.tsv` (tissue-level) and `tutorial/human_protein_atlas_seismic_cell_spec_matrix.tsv` (cell-tissue level, produced by Part 1).

### 2. Proteomic summary statistics

One CSV **per disease**, named `<disease_name>.csv`, all stored in the same folder. Two accepted layouts:

**a) UK Biobank Proteome-Phenome Atlas format** (see [proteome-phenome-atlas.com](https://proteome-phenome-atlas.com/)) — as in `sample_data/sample_sumstats/Alcoholic_liver_disease.csv`. The loader parses the `Protein`, `HR[95%CI]` (or `OR[95%CI]`), and `P_value` columns and maps protein names to Ensembl IDs.

**b) Custom pre-formatted** — as in `sample_data/sample_sumstats/AD_meta_sumstats_all.csv`. Must contain at least:

| Column | Meaning |
|--------|---------|
| `gene` | Gene identifier, **same format as the atlas `gene` column** (Ensembl ID or symbol) |
| `P_value` | Association p-value |
| `OR` or `HR` | Odds/hazard ratio |
| `logOR` or `logHR` | Its natural log |

Extra columns are ignored. The loader tries format (a) first and falls back to (b); with format (b) no protein-name→gene-ID mapping is attempted, so pre-map your genes to match the atlas.

> Note: the built-in name→Ensembl mapping for format (a) relies on lookup tables under a lab-specific absolute path. External users should use format (b) with genes already mapped to the atlas identifier space.

---

## Quick start

The fastest way to understand the codebase is the fully worked example in [`tutorial/`](tutorial/README.md), which ships a tiny Human Protein Atlas object, one disease, and reference outputs. In brief:

```bash
# Part 1 (R) — compute the specificity matrix (runs anywhere)
Rscript cell_tissue_specificity/hpa_seismic_cell_type_specificity.R

# Part 2 (Python, LSF) — run the association pipeline for all diseases in the config
conda activate scanpy-env
python pipelines/pipeline_univar_to_multivar.py \
    --yml_file pipeline_yml/univar_multivar_sample.yml
```

Launch the pipeline from a lightweight interactive/head job: it submits and manages one cluster job per disease per method.

---

## The four association methods

All four live in `python_main_cell_type_spec_method/` and share the same inputs (specificity matrix + one disease's sumstats) and the same standardization options (`--ztransform_type`: `1` = z-score per cell type, `2` = z-score per gene, `-1` = none).

| Method | Script | What it does | Key outputs |
|--------|--------|--------------|-------------|
| **Univariate regression** | `univar_association_testing.py` | One OLS per cell type: `disease_effect ~ specificity (+ covariates)`. Reports beta, t, p, FDR, one-sided p, residual-normality test. Optional covariates: mean expression, Gini specificity, PCs. Tuned to reproduce `seismicGWAS`. | `univar_regression_results.tsv` |
| **ElasticNet (k-fold CV)** | `elastic_kfold_ver2.py` | Grid search over `alpha` × `l1_ratio` with k-fold CV; picks best models by R², Pearson r, and MSE; refits on full data and reports coefficients. | `perf_df.tsv`, `coef_df.tsv`, `best_model*.tsv`, coefficient plots |
| **LASSO stability selection** | `stability_analyses.py` | Repeated subsampled LASSO fits over a λ grid; retains cell types selected above a stability threshold. | `feature_scores.tsv`, `selected_features_thres_*.tsv`, `regularization_path.pdf` |
| **Random forest** | `tree_based_methods.py` | Random-forest regression with optional OOB hyperparameter search; reports impurity importances and (more reliable) held-out permutation importances. | `coef_random_forest.tsv`, `permute_importance_scores.tsv/.png` |

Each run also writes `cmd_args.json` (the exact arguments used) and `prot_spec_final.tsv` (the harmonized protein table) for provenance.

### Running a single method standalone

The base scripts take a specificity matrix, a sumstats folder, and a disease name directly. For example:

```bash
python python_main_cell_type_spec_method/univar_association_testing.py \
    --atlas_smal_path tutorial/human_protein_atlas_seismic_cell_spec_matrix.tsv \
    --prot_data_path  tutorial/sample_disease \
    --disease         Alcoholic_liver_disease \
    --save_path       /tmp/univar_out \
    --output_label    z_score --ztransform_type -1
```

Full argument lists are in each script's `argparse` block and in the matching `bash_scripts/*.sh`.

---

## The pipeline

`pipelines/pipeline_univar_to_multivar.py` chains the methods into one workflow driven by a single YAML file:

1. For each disease, run **univariate** regression.
2. **Select** cell types passing a threshold on a chosen column (default: `fdr_one_side_predictor <= 0.05`) and subset the specificity matrix to those cell types.
3. Run any enabled **multivariate** methods (ElasticNet / stability selection / random forest) on that reduced set.

Run it with:

```bash
python pipelines/pipeline_univar_to_multivar.py --yml_file pipeline_yml/univar_multivar_sample.yml
```

### Configuration (`univar_multivar_sample.yml`)

The YAML has these sections:

- `constants` — paths to the four `*_auto-script.py` launchers.
- `inputs` — specificity-matrix path, sumstats directory (`disease_prot_dir/disease_folder_name`), output path, and `disease_name` (a list of diseases, or `["all"]`).
- `univariate` — univariate parameters (covariates, output label, standardization).
- `univar_to_multivar` — `column_to_choose` and `thres` used for feature selection between the univariate and multivariate stages.
- `elasticnet_kfold`, `stability_selection`, `random_forest` — each with a `run: 0/1` toggle and method-specific hyperparameters.

To fan a configuration out into one YAML per disease, use `pipelines/utils.py` (`create_yml_files`).

---

## Outputs

Results are written under `<save_path>[_<suffix>]/<disease_folder_name>/<method>/<disease>/`. The `tutorial/pipeline_univar_multivar_sample_run/` directory contains a complete reference set you can diff against. Each method directory contains its result tables/plots plus `cmd_args.json` and per-job `.stdout`/`.stderr` logs.

---

## Running without LSF

To run Part 2 off-cluster, bypass the `bsub` launcher layer and call the **base** scripts directly (as in [Running a single method standalone](#running-a-single-method-standalone)) — they are plain scikit-learn/statsmodels and have no cluster dependency. The `*_auto-script.py` files and `bash_scripts/*.sh` are only the LSF submission wrappers; a Slurm or local port would replace those two layers while leaving the base scripts unchanged.

---

## Notes and caveats

- Hardware/scheduler: Part 2 as shipped assumes LSF + a conda env named `scanpy-env` loaded via `module load anaconda3/latest`. Adjust `bash_scripts/*.sh` for your site.
- The univariate parameters in the tutorial YAML are set so results match the original `seismicGWAS` output; multivariate hyperparameters are exploratory. In practice the univariate ranking is often already the most interpretable.
- The format-(a) protein-name→gene-ID mapping is tailored to a specific proteomics dataset and lab file paths; prefer the pre-mapped format (b) for new data.

---

## License

Released under the [MIT License](LICENSE).
