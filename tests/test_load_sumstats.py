"""Which reader `load_sumstats` picks, and what it does when that reader cannot run.

The bug these cover: the four method scripts used to wrap the UK Biobank reader in
`try: ... except: pd.read_csv(...)` with a bare `except`. Any failure inside the
reader -- including the gene-ID lookup tables simply not being on disk -- was
therefore read as "this file must be in the other format", and the run continued
with a frame that had no `gene` index and no `logHR`/`logOR`. It died a few hundred
lines later on `KeyError: "['OR', 'logOR'] not in index"`, naming columns that were
never the problem.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "python_main_cell_type_spec_method"),
)

import utils  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_SUMSTATS = os.path.join(REPO_ROOT, "sample_data", "sample_sumstats")

# One row of each real format, taken from the shapes in sample_data/sample_sumstats.
PRE_MAPPED = "P_value,logOR,OR,gene\n1.0,0.0204,1.0206,ENSG00000244752\n"
UK_BIOBANK = (
    "Disease_category,Disease,Protein,Protein_definition,NB_individual,NB_case,HR[95%CI],P_value\n"
    "Chapter XI,Alcoholic liver disease,A1BG,Alpha-1B-glycoprotein,43311,131,3.82 [1.41-10.37],0.0085\n"
)


def _write(tmp_path, name, text):
    (tmp_path / f"{name}.csv").write_text(text)
    return str(tmp_path)


def test_is_pre_mapped_discriminates_the_two_real_sample_files():
    """The repo ships one file of each format; the test is only meaningful if
    they really do differ in the column the dispatch reads."""
    pre_mapped = pd.read_csv(os.path.join(SAMPLE_SUMSTATS, "AD_meta_sumstats_all.csv"), nrows=0)
    uk_biobank = pd.read_csv(os.path.join(SAMPLE_SUMSTATS, "Alcoholic_liver_disease.csv"), nrows=0)

    assert utils.is_pre_mapped_sumstats(pre_mapped)
    assert not utils.is_pre_mapped_sumstats(uk_biobank)


def test_pre_mapped_file_is_read_without_touching_the_lookup_tables(tmp_path, monkeypatch):
    """A file that already carries Ensembl IDs must not need the lookup tables --
    it is the case that legitimately bypasses load_prot_data."""
    base = _write(tmp_path, "PreMapped", PRE_MAPPED)
    monkeypatch.setattr(utils, "GENE_ID_SYMBOLS", str(tmp_path / "absent.tsv"))
    monkeypatch.setattr(utils, "GENE_ID_HGNC", str(tmp_path / "absent_too.tsv"))

    df = utils.load_sumstats(base, "PreMapped", pd.DataFrame())

    assert list(df["gene"]) == ["ENSG00000244752"]


def test_uk_biobank_file_raises_when_lookup_tables_are_missing(tmp_path, monkeypatch):
    """The regression. Before the fix this returned the raw CSV instead of raising,
    so the caller got a frame with `Protein`/`HR[95%CI]` and no `gene`."""
    base = _write(tmp_path, "Alcoholic_liver_disease", UK_BIOBANK)
    monkeypatch.setattr(utils, "GENE_ID_SYMBOLS", str(tmp_path / "absent.tsv"))
    monkeypatch.setattr(utils, "GENE_ID_HGNC", str(tmp_path / "absent_too.tsv"))

    with pytest.raises(utils.MissingGeneIdLookup) as excinfo:
        utils.load_sumstats(base, "Alcoholic_liver_disease", pd.DataFrame())

    # The message has to name the files, or it is no better than the KeyError it replaced.
    assert "absent.tsv" in str(excinfo.value)
    assert "absent_too.tsv" in str(excinfo.value)


def test_uk_biobank_reader_errors_are_not_swallowed(tmp_path, monkeypatch):
    """Lookup tables present but the reader fails for some other reason (a renamed
    column, an unmapped protein). That is a bug to surface, not a signal to switch
    formats -- the bare `except` could not tell the two apart."""
    base = _write(tmp_path, "Alcoholic_liver_disease", UK_BIOBANK)
    for name in ("GENE_ID_SYMBOLS", "GENE_ID_HGNC"):
        present = tmp_path / f"{name}.tsv"
        present.write_text("gene_ids\tgene_symbols\nENSG1\tA1BG\n")
        monkeypatch.setattr(utils, name, str(present))

    def boom(*_args, **_kwargs):
        raise ValueError("unmappable protein")

    monkeypatch.setattr(utils, "load_prot_data", boom)

    with pytest.raises(ValueError, match="unmappable protein"):
        utils.load_sumstats(base, "Alcoholic_liver_disease", pd.DataFrame())
