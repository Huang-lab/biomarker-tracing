"""load_prot_data on diseases where some proteins have no estimate, or where BNP proteins are missing.

The bugs these cover: rare diseases have proteins whose model failed (blank effect and P_value, or an
effect with a "nan-nan" interval). float(x.split(...)) raised on the blank, so the whole disease was lost.
And the NPPB/NTproBNP de-duplication raised when neither was present, and silently deleted the only one
when just one was present.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python_main_cell_type_spec_method"))

import utils  # noqa: E402

HEADER = "Disease_category,Disease,Protein,Protein_definition,NB_individual,NB_case,OR[95%CI],P_value\n"


def row(protein, effect, p):
    return f"Chapter I,Disease,{protein},{protein} protein,1000,100,{effect},{p}\n"


@pytest.fixture(scope="module")
def atlas():
    return pd.read_csv(os.path.join(REPO, "tutorial", "human_protein_atlas_seismic_cell_spec_matrix.tsv"),
                       sep="\t").set_index("gene")


def load(tmp_path, atlas, rows):
    (tmp_path / "Disease.csv").write_text(HEADER + "".join(rows))
    return utils.load_prot_data(str(tmp_path), "Disease", atlas).set_index("gene_name")


def test_proteins_without_an_estimate_are_dropped_not_fatal(tmp_path, atlas):
    out = load(tmp_path, atlas, [row("GDF15", "2.00 [1.50-2.60]", 1e-8),
                                 row("WFDC2", "", ""),
                                 row("TNFRSF1B", "1.30 [1.10-1.50]", 0.02)])
    assert set(out.index) == {"GDF15", "TNFRSF1B"}
    assert out.loc["GDF15", "OR"] == pytest.approx(2.0)


def test_effect_with_a_missing_interval_is_kept(tmp_path, atlas):
    out = load(tmp_path, atlas, [row("GDF15", "0.89 [nan-nan]", 0.04), row("TNFRSF1B", "1.30 [1.10-1.50]", 0.02)])
    assert "GDF15" in out.index
    assert np.isnan(out.loc["GDF15", "logOR_se"])


def test_non_positive_or_infinite_effects_are_dropped(tmp_path, atlas):
    out = load(tmp_path, atlas, [row("GDF15", "0.00 [0.00-0.10]", 0.5), row("WFDC2", "inf [1.00-inf]", 0.5),
                                 row("TNFRSF1B", "1.30 [1.10-1.50]", 0.02)])
    assert list(out.index) == ["TNFRSF1B"]


def test_both_bnp_proteins_keep_the_smaller_p_value(tmp_path, atlas):
    out = load(tmp_path, atlas, [row("NPPB", "1.50 [1.20-1.90]", 0.01), row("NTproBNP", "1.40 [1.10-1.80]", 0.20),
                                 row("TNFRSF1B", "1.30 [1.10-1.50]", 0.02)])
    assert "NPPB" in out.index and "NTproBNP" not in out.index


def test_a_lone_bnp_protein_is_not_deleted(tmp_path, atlas):
    only_nppb = load(tmp_path, atlas, [row("NPPB", "1.50 [1.20-1.90]", 0.01), row("TNFRSF1B", "1.30 [1.10-1.50]", 0.02)])
    assert "NPPB" in only_nppb.index
    only_nt = load(tmp_path, atlas, [row("NTproBNP", "1.40 [1.10-1.80]", 0.20), row("TNFRSF1B", "1.30 [1.10-1.50]", 0.02)])
    assert "NTproBNP" in only_nt.index


def test_neither_bnp_protein_present_does_not_raise(tmp_path, atlas):
    out = load(tmp_path, atlas, [row("GDF15", "2.00 [1.50-2.60]", 1e-8), row("TNFRSF1B", "1.30 [1.10-1.50]", 0.02)])
    assert len(out) == 2


def test_equal_p_values_for_both_bnp_proteins_do_not_raise(tmp_path, atlas):
    out = load(tmp_path, atlas, [row("NPPB", "1.50 [1.20-1.90]", 0.5), row("NTproBNP", "1.40 [1.10-1.80]", 0.5),
                                 row("TNFRSF1B", "1.30 [1.10-1.50]", 0.02)])
    assert len({"NPPB", "NTproBNP"} & set(out.index)) == 1
