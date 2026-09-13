import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from conditional_gaussian_reference import GaussianReference, leave_cluster_operator


class GaussianReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(917)
        self.z = self.rng.normal(size=24)
        self.phi = np.column_stack([np.ones(24), self.z])

    def test_matches_partial_regression_t(self):
        x, y = self.rng.normal(size=(2, 24))
        design = np.column_stack([self.phi, x])
        coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ coefficient
        se = np.sqrt((residual @ residual / 21) * np.linalg.inv(design.T @ design)[-1, -1])
        got = GaussianReference(self.phi, np.eye(24)).evaluate(y[:, None], x[:, None])
        self.assertAlmostEqual(got["statistic"][0], coefficient[-1] / se, places=11)
        self.assertAlmostEqual(got["pvalue"][0], t.sf(coefficient[-1] / se, 21), places=12)

    def test_known_covariance_and_mean_shift_invariance(self):
        covariance = .6 ** np.abs(np.subtract.outer(np.arange(24), np.arange(24)))
        reference = GaussianReference(self.phi, covariance)
        y, direction = self.rng.normal(size=(2, 24))
        a = reference.evaluate(y[:, None], direction[:, None])
        b = reference.evaluate((y + self.phi @ [30., -17.])[:, None], direction[:, None])
        np.testing.assert_allclose(a["pvalue"], b["pvalue"], atol=2e-13)
        white = GaussianReference(reference.whiten(self.phi), np.eye(24))
        c = white.evaluate(reference.whiten(y[:, None]), reference.cholesky.T @ direction[:, None])
        np.testing.assert_allclose(a["pvalue"], c["pvalue"], atol=2e-13)

    def test_leave_cluster_target_and_saturation_failure(self):
        groups = np.repeat(np.arange(6), 4)
        a = leave_cluster_operator(self.phi, groups)
        np.testing.assert_allclose(a @ self.phi, 0., atol=1e-14)
        for g in np.unique(groups):
            ix = np.flatnonzero(groups == g)
            np.testing.assert_allclose(a[np.ix_(ix, ix)], np.eye(4) / 24, atol=1e-14)
        covariance = np.zeros((24, 24))
        for g in np.unique(groups):
            ix = np.flatnonzero(groups == g)
            covariance[np.ix_(ix, ix)] = self.rng.normal(size=(4, 4))
        self.assertAlmostEqual(np.trace(a @ covariance), np.trace(covariance) / 24, places=13)
        with self.assertRaisesRegex(ValueError, "deleting"):
            leave_cluster_operator(np.eye(6)[groups], groups)

    def test_degenerate_and_invalid_inputs(self):
        reference = GaussianReference(self.phi, np.eye(24))
        result = reference.evaluate(self.z[:, None], np.ones((24, 1)))
        self.assertEqual(result["pvalue"][0], 1.)
        self.assertFalse(result["defined"][0])
        with self.assertRaises(ValueError):
            GaussianReference(np.eye(24), np.eye(24))
        with self.assertRaises(ValueError):
            reference.evaluate(np.zeros((23, 1)), np.zeros((23, 1)))


if __name__ == "__main__":
    unittest.main()
