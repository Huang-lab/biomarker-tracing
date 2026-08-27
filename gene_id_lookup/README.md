# Gene-ID lookup tables

`python_main_cell_type_spec_method/utils.py` reads two files from this directory when it
loads **UK Biobank Phenome-Proteome** sumstats, to map protein names to Ensembl gene IDs:

| File | Columns used |
|---|---|
| `gene_id_symbol_df.tsv` | `gene_ids`, `gene_symbols` |
| `gene_id_symbol_hgnc.tsv` | `Approved_symbol`, `Ensembl_gene_ID` (plus HGNC columns that are dropped on load) |

**Neither file is in the repo.** They were previously read from a Mount Sinai HPC path
(`/sc/arion/projects/DiseaseGeneCell/.../CSF_proteomics_AD_onset/`), so a fresh clone has
never had them. Ask the lab for the two TSVs and drop them here.

Sumstats that already carry a `gene` column of Ensembl IDs -- the format
`sample_data/sample_sumstats/AD_meta_sumstats_all.csv` is in -- do not need these tables
at all, and are read without consulting them.

Until the files are present, running any of the four methods on a UK Biobank-format file
(including the tutorial's `sample_disease/`) stops with a `MissingGeneIdLookup` naming both
paths. That is deliberate: it used to continue silently with an unmapped frame and fail
several hundred lines later on an unrelated `KeyError`.
