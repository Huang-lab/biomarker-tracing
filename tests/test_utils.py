import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        custom = utils.load_prot_data_for_disease(
            str(REPO_ROOT / "sample_data/sample_sumstats"),
            "AD_meta_sumstats_all",
            self.atlas,
        )
        uk_biobank = utils.load_prot_data_for_disease(
            str(REPO_ROOT / "tutorial/sample_disease"),
            "Alcoholic_liver_disease",
            self.atlas,
        )

        self.assertIn("gene", custom.columns)
        self.assertIn("OR", custom.columns)
        self.assertIn("gene", uk_biobank.columns)
        self.assertIn("HR", uk_biobank.columns)

    def test_missing_lookup_files_are_not_treated_as_custom_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "disease.csv"
            pd.DataFrame(
                {
                    "Protein": ["A1BG"],
                    "HR[95%CI]": ["1.2 [1.0-1.4]"],
                    "P_value": [0.05],
                }
            ).to_csv(data_path, index=False)

            with patch.object(utils, "GENE_ID_SYMBOLS", str(Path(temp_dir) / "symbols.tsv")), patch.object(
                utils, "GENE_ID_HGNC", str(Path(temp_dir) / "hgnc.tsv")
            ):
                with self.assertRaisesRegex(FileNotFoundError, "lookup files"):
                    utils.load_prot_data_for_disease(temp_dir, "disease", self.atlas)


if __name__ == "__main__":
    unittest.main()
