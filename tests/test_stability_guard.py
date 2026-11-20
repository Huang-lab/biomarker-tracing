"""Stability selection must not return an empty result when the penalty grid cannot select anything.

The failure these cover: stability_analyses.py scans a fixed penalty grid (0.1 to 1).  On unscaled specificity
(--ztransform_type -1, what the tutorial config used to ship for this step) and a z-score outcome of standard
deviation 4, the smallest penalty that keeps any coefficient is far below 0.1, so every stability score is 0.0 and
the script wrote an empty `selected_features_thres_*.tsv` and exited 0.

The data are synthetic and small; the real code (`train`, `prep_data`, `StabilitySelection`) runs on them.
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python_main_cell_type_spec_method"))
if importlib.util.find_spec("stability_selection") is None:      # not installed: skip.  Do not import it here, because the
    pytest.skip("stability-selection is not installed", allow_module_level=True)    # package only imports after the shim in stability_analyses
import stability_analyses  # noqa: E402     (a broken shim fails here, loudly)


def make_inputs(spread, signal, n=400, p=6, seed=0):
    """An atlas of `p` cell types with specificity of scale `spread`, and a z-score outcome (sd 4) with `signal` on cell type 0."""
    rng = np.random.default_rng(seed)
    genes = [f"ENSG{i:05d}" for i in range(n)]
    atlas = pd.DataFrame(np.abs(rng.normal(scale=spread, size=(n, p))), index=pd.Index(genes, name="gene"),
                         columns=[f"tissue_cell{j}" for j in range(p)])
    z = rng.normal(scale=4.0, size=n) + signal * (atlas.iloc[:, 0] - atlas.iloc[:, 0].mean()).to_numpy() / atlas.iloc[:, 0].std()
    prot = pd.DataFrame({"gene": genes, "HR": np.where(z > 0, 1.1, 0.9), "logHR": np.where(z > 0, np.log(1.1), np.log(0.9)),
                         "P_value": 2 * stats.norm.sf(np.abs(z))})
    return atlas, prot


def run(tmp_path, atlas, prot, ztransform_type):
    args = SimpleNamespace(save_path=str(tmp_path), ztransform_type=ztransform_type, abs_hr=False, output_label="z_score",
                           gene_weight_minmax=0, thres=0.6)
    stability_analyses.train(args, atlas, prot)
    return pd.read_csv(tmp_path / "selected_features_thres_0.6.tsv", sep="\t").selected_features.tolist()


def test_grid_above_alpha_max_raises_instead_of_writing_an_empty_result(tmp_path):
    atlas, prot = make_inputs(spread=0.05, signal=0.0)
    with pytest.raises(ValueError, match="no cell type at any of the 20 penalties"):
        run(tmp_path, atlas, prot, ztransform_type=-1)
    assert not (tmp_path / "selected_features_thres_0.6.tsv").exists()     # nothing that looks like a result is left behind


def test_a_real_signal_is_selected_when_the_inputs_are_z_scored(tmp_path):
    # Only membership is asserted: with outcome sd 5 the smallest penalties also let noise cell types through (the grid is not
    # tied to the data scale), and this test is about the guard not firing and the signal being reported, not about specificity
    atlas, prot = make_inputs(spread=0.05, signal=3.0)
    assert "tissue_cell0" in run(tmp_path, atlas, prot, ztransform_type=1)


def test_the_tutorial_config_does_not_use_the_setting_that_selects_nothing():
    import yaml
    with open(os.path.join(REPO, "tutorial", "univar_multivar_sample.yml")) as f:
        assert yaml.safe_load(f)["stability_selection"]["ztransform_type"] == 1
