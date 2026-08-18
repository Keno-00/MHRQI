import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "analysis" / "recompute_paper_statistics.py"
SPEC = importlib.util.spec_from_file_location("revision_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RevisionAnalysisTests(unittest.TestCase):
    def test_sample_standard_deviation_is_used(self):
        result = MODULE.summarize([1.0, 2.0, 3.0], seed=1, resamples=99)
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["sample_sd"], 1.0)

    def test_rank_biserial_keeps_direction(self):
        result = MODULE.signed_rank_details(
            np.array([4.0, 5.0, 6.0]), np.array([1.0, 2.0, 3.0])
        )
        self.assertEqual(result["w_minus"], 0.0)
        self.assertEqual(result["rank_biserial_proposed_minus_baseline"], 1.0)


if __name__ == "__main__":
    unittest.main()

