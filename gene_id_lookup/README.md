# Gene ID lookup tables

These tables support the UK Biobank Proteome-Phenome summary-statistics loader in `python_main_cell_type_spec_method/utils.py`.

- `gene_id_symbol_df.tsv`: columns `gene_ids` and `gene_symbols`
- `gene_id_symbol_hgnc.tsv`: HGNC, approved-symbol, Ensembl, NCBI, and UCSC identifier columns

They are required only for summary-statistics files in the UK Biobank format. Pre-mapped files with a `gene` column do not use these tables.
