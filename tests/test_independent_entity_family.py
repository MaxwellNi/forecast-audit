"""Independent formula, tail-rank and family checks for the synthetic study."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/analysis"))
import independent_entity_family as study


class FamilyChecks(unittest.TestCase):
    def test_bilinear_scores_match_rowwise_fit(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "scripts/analysis/equal_entity_directional.py"
        spec = importlib.util.spec_from_file_location("rowwise", source)
        rowwise = importlib.util.module_from_spec(spec); spec.loader.exec_module(rowwise)
        rng = np.random.default_rng(786345)
        x, y = rng.normal(size=(2, 8, 20))
        matrices, used = study.score_matrices()
        values, reference_used = rowwise.entity_scores(x, y, rowwise.make_folds(20), gap=5)
        np.testing.assert_array_equal(reference_used, np.broadcast_to(used, x.shape))
        for m, key in enumerate(("standard", "gapped_complementary", "directional")):
            actual = np.einsum("gt,tu,gu->g", x, matrices[m], y)
            np.testing.assert_allclose(actual, values[key], atol=2e-15, rtol=1e-13)
            np.testing.assert_allclose(matrices[m] @ np.ones(20), 0., atol=1e-15)
            np.testing.assert_allclose(np.ones(20) @ matrices[m], 0., atol=1e-15)

    def test_by_matches_independent_implementation(self):
        rng = np.random.default_rng(315)
        for k in (10, 11):
            for _ in range(20):
                p = rng.uniform(size=k)**5
                np.testing.assert_allclose(study.by_adjust(p), multipletests(p, method="fdr_by")[1], atol=2e-15)

    def test_calibration_ties_and_extreme_tail(self):
        reference = np.array([-1., 0., 0., 2.])
        np.testing.assert_array_equal(study.calibration_p(np.array([-2., 0., 3.]), reference), [1., .8, .2])

    def test_studentization_and_numerical_abstention(self):
        x = np.array([1., 2., 3., 4.])
        mean, se, t, guard = study.statistic(x)
        self.assertFalse(guard)
        self.assertAlmostEqual(se, np.sqrt(5/3)/2)
        self.assertAlmostEqual(t, mean/se)
        self.assertTrue(study.statistic(np.ones(10))[3])

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError): study.run(directory, 2, 2)


if __name__ == "__main__": unittest.main()
