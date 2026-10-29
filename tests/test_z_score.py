"""The signed z-score used as the outcome when P_value has underflowed to 0.

The bug these cover: `stats.norm.isf(0 / 2)` is inf, and prep_data then replaced every
inf in the frame with the largest finite -log10(pval). That put a -log10(p) value into
the z_score column (NTproBNP in heart failure got z = 226.6 against a next-largest 32.2;
146 proteins in chronic kidney disease all got 310.3) and made it positive even for
proteins with HR < 1.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "python_main_cell_type_spec_method"),
)

import utils  # noqa: E402

ARGS = SimpleNamespace(output_label="z_score", abs_hr=0, gene_weight_minmax=0)
FLOOR = stats.norm.isf(np.finfo(float).tiny / 2)   # |z| of the smallest positive double


def _frame(hr, p, se=None):
    df = pd.DataFrame({"HR": hr, "logHR": np.log(hr), "P_value": p})
    if se is not None:
        df["logHR_se"] = se
    return df


def test_finite_p_values_are_unchanged():
    df = _frame([1.5, 0.8, 1.2], [1e-10, 0.03, 0.5])
    out, _, _ = utils.prep_data(ARGS, df, "HR")
    expected = [stats.norm.isf(1e-10 / 2), -stats.norm.isf(0.03 / 2), stats.norm.isf(0.5 / 2)]
    np.testing.assert_allclose(out["z_score"], expected)


def test_underflowed_p_uses_the_confidence_interval():
    # NTproBNP in heart failure: HR 1.74 [1.70-1.79], reported P_value = 0
    se = (np.log(1.79) - np.log(1.70)) / (2 * stats.norm.isf(0.025))
    df = _frame([1.74, 2.30], [0.0, 2.6e-227], se=[se, 0.02])
    out, _, _ = utils.prep_data(ARGS, df, "HR")

    assert out["z_score"].iloc[0] == pytest.approx(np.log(1.74) / se)
    assert out["z_score"].iloc[0] < 60          # not the 226.6 of -log10(p) units
    assert out["z_score"].iloc[0] > out["z_score"].iloc[1]   # still the strongest protein


def test_underflowed_p_keeps_a_protective_sign():
    df = _frame([0.5, 1.1], [0.0, 0.2], se=[0.01, 0.05])
    out, _, _ = utils.prep_data(ARGS, df, "HR")
    assert out["z_score"].iloc[0] < -FLOOR


def test_underflowed_p_is_never_weaker_than_the_float_floor():
    # A rounded CI can give |log HR| / SE below what P_value = 0 implies
    df = _frame([1.2], [0.0], se=[0.1])
    out, _, _ = utils.prep_data(ARGS, df, "HR")
    assert out["z_score"].iloc[0] == pytest.approx(FLOOR)


def test_underflowed_p_ranks_above_every_reported_p():
    # Chronic kidney disease reports P_value = 4.96e-311 (a subnormal, |z| = 37.7, above
    # the normal-double floor) next to proteins with P_value = 0
    reported = stats.norm.isf(4.96e-311 / 2)
    df = _frame([1.2, 1.5], [0.0, 4.96e-311], se=[0.1, 0.01])
    out, _, _ = utils.prep_data(ARGS, df, "HR")
    assert reported > FLOOR
    assert out["z_score"].iloc[0] == pytest.approx(reported)


def test_underflowed_p_without_a_confidence_interval_uses_the_float_floor():
    df = _frame([1.3, 0.7], [0.0, 0.0])
    out, _, _ = utils.prep_data(ARGS, df, "HR")
    np.testing.assert_allclose(out["z_score"], [FLOOR, -FLOOR])


def test_minus_log10_p_weight_keeps_its_old_replacement():
    df = _frame([1.5, 1.3], [0.0, 1e-50], se=[0.01, 0.01])
    out, _, _ = utils.prep_data(ARGS, df, "HR")
    np.testing.assert_allclose(out["-log10(pval)"], [50.0, 50.0])


def test_se_column_is_not_returned_so_it_cannot_become_a_model_feature():
    df = _frame([1.5], [0.0], se=[0.01])
    out, _, _ = utils.prep_data(ARGS, df, "HR")
    assert "logHR_se" not in out.columns


def test_association_columns_only_adds_se_when_present():
    with_se = _frame([1.5], [0.01], se=[0.1])
    assert utils.association_columns(with_se, "HR", "gene") == ["HR", "logHR", "gene", "P_value", "logHR_se"]
    assert utils.association_columns(_frame([1.5], [0.01]), "HR") == ["HR", "logHR", "P_value"]


def test_uk_biobank_reader_derives_se_from_the_ci(tmp_path):
    (tmp_path / "Disease.csv").write_text(
        "Disease_category,Disease,Protein,Protein_definition,NB_individual,NB_case,HR[95%CI],P_value\n"
        "Chapter IX,Heart failure,GDF15,Growth differentiation factor 15,1000,100,1.74 [1.70-1.79],0\n"
        # load_prot_data currently expects both of these to be present
        "Chapter IX,Heart failure,NTproBNP,N-terminal prohormone BNP,1000,100,2.00 [1.90-2.10],1e-50\n"
        "Chapter IX,Heart failure,NPPB,Natriuretic peptide B,1000,100,1.50 [1.40-1.60],1e-20\n"
    )
    atlas = pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tutorial", "human_protein_atlas_seismic_cell_spec_matrix.tsv"),
        sep="\t",
    ).set_index("gene")
    prot = utils.load_prot_data(str(tmp_path), "Disease", atlas).set_index("gene_name")
    expected = (np.log(1.79) - np.log(1.70)) / (2 * stats.norm.isf(0.025))
    assert prot.loc["GDF15", "logHR_se"] == pytest.approx(expected)
