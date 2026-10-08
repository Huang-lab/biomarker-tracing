import unittest
from pathlib import Path

import pandas as pd

from python_main_cell_type_spec_method import utils


REPO_ROOT = Path(__file__).parents[1]


class ProteinDataLoaderTests(unittest.TestCase):
    def setUp(self):
        atlas = pd.read_csv(
            REPO_ROOT / "tutorial/human_protein_atlas_seismic_cell_spec_matrix.tsv",
            sep="\t",
        )
        self.atlas = atlas.set_index("gene")

    def test_dispatches_custom_and_uk_biobank_formats(self):
        custom = utils.load_sumstats(
            str(REPO_ROOT / "sample_data/sample_sumstats"),
            "AD_meta_sumstats_all",
            self.atlas,
        )
        uk_biobank = utils.load_sumstats(
            str(REPO_ROOT / "tutorial/sample_disease"),
            "Alcoholic_liver_disease",
            self.atlas,
        )

        self.assertIn("gene", custom.columns)
        self.assertIn("OR", custom.columns)
        self.assertIn("gene", uk_biobank.columns)
        self.assertIn("HR", uk_biobank.columns)


if __name__ == "__main__":
    unittest.main()
