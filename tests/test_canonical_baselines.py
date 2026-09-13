"""Independent formula and data-isolation checks for the new benchmark."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from canonical_baselines import (WEIGHTS, Q_LADDER, binned_predictions,
                                covariance_test, partial_distance_summary,
                                spline_predictions, u_centered_distances, u_inner)


class CanonicalBaselineTests(unittest.TestCase):
    def test_u_centering_explicit_loops(self):
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        d = abs(x[:, None] - x[None, :])
        expected = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                if i != j:
                    expected[i, j] = d[i, j] - sum(d[i]) / 3 - sum(d[:, j]) / 3 + d.sum() / 12
        actual = u_centered_distances(x)
        assert_allclose(actual, expected, atol=1e-13)
        assert_allclose(actual.sum(axis=0), 0, atol=1e-13)
        assert_allclose(actual.diagonal(), 0)

    def test_distance_known_published_examples(self):
        a = np.array([1, 1, 2, 2, 3])
        b = np.array([1, 2, 1, 2, 1])
        c = np.array([1, 2, 2, 1, 2])
        self.assertAlmostEqual(partial_distance_summary(a, a, c)[1], 1)
        self.assertAlmostEqual(partial_distance_summary(a, b, c)[1], -0.5)
        self.assertAlmostEqual(partial_distance_summary(a, c, c)[1], 0)

    def test_distance_projection_orthogonality_and_constant_control(self):
        rng = np.random.default_rng(901)
        x, y, z = rng.normal(size=(3, 17))
        a, c = u_centered_distances(x), u_centered_distances(z)
        pa = a - u_inner(a, c) / u_inner(c, c) * c
        self.assertAlmostEqual(u_inner(pa, c), 0, places=13)
        cov, _ = partial_distance_summary(x, y, np.ones(len(x)))
        self.assertAlmostEqual(cov, u_inner(a, u_centered_distances(y)), places=13)

    @unittest.skipUnless(importlib.util.find_spec("dcor"), "Optional dcor reference is not installed")
    def test_independent_dcor_package_matches(self):
        import dcor
        rng = np.random.default_rng(871)
        for n in (7, 19, 41):
            x, y, z = [rng.normal(size=(n, d)) for d in (1, 2, 3)]
            cov, corr = partial_distance_summary(x, y, z)
            self.assertAlmostEqual(cov, float(dcor.partial_distance_covariance(x, y, z)), places=12)
            self.assertAlmostEqual(corr, float(dcor.partial_distance_correlation(x, y, z)), places=12)

    def test_extrapolation_constraints(self):
        assert_allclose(WEIGHTS.sum(), 1, atol=1e-13)
        assert_allclose(WEIGHTS @ (1 / Q_LADDER), 0, atol=1e-13)
        assert_allclose(WEIGHTS, [-.49609375, .01953125, .27734375, .53515625, .6640625], atol=1e-13)

    def test_bin_nuisance_heldout_response_isolation(self):
        rng = np.random.default_rng(551)
        z = rng.normal(size=80)
        values = rng.normal(size=(80, 2))
        folds = np.arange(80) % 2
        expected = binned_predictions(values, z, folds, 8)
        changed = values.copy()
        changed[folds == 0] += 100
        actual = binned_predictions(changed, z, folds, 8)
        assert_allclose(actual[folds == 0], expected[folds == 0], atol=0)

    def test_spline_nuisance_heldout_response_isolation(self):
        rng = np.random.default_rng(781)
        z = rng.normal(size=80)
        values = rng.normal(size=(80, 2))
        folds = np.arange(80) % 2
        expected = spline_predictions(values, z, folds)
        changed = values.copy()
        changed[folds == 0] += 100
        actual = spline_predictions(changed, z, folds)
        assert_allclose(actual[folds == 0], expected[folds == 0], atol=0)

    def test_singleton_cluster_studentization(self):
        s = np.array([1, 2, 4, 8, 16, 32], dtype=float)
        statistic, pvalue = covariance_test(s)
        self.assertAlmostEqual(statistic, np.sqrt(len(s)) * s.mean() / s.std(ddof=0))
        self.assertTrue(0 <= pvalue <= 1)

    @unittest.skipUnless(importlib.util.find_spec("causallearn"), "Optional causal-learn reference is not installed")
    def test_kci_library_kernel_score_and_gamma_formula(self):
        from causallearn.utils.KCI.KCI import KCI_CInd
        from scipy.stats import gamma
        rng = np.random.default_rng(5107)
        z = rng.normal(size=(31, 1))
        x, y = z + rng.normal(size=(31, 1)), z + rng.normal(size=(31, 1))
        model = KCI_CInd()
        pvalue, statistic = model.compute_pvalue(x, y, z)
        kx, ky, kz, _ = model.kernel_matrix(x, y, z)
        residual = .001 * np.linalg.pinv(kz + .001 * np.eye(len(z)))
        ax = residual @ kx @ residual
        ay = residual @ ky @ residual
        expected_statistic = np.trace(ax @ ay)
        self.assertAlmostEqual(statistic, expected_statistic, places=8)
        wx, vx = np.linalg.eigh((ax + ax.T)/2)
        wy, vy = np.linalg.eigh((ay + ay.T)/2)
        ix, iy = wx > wx.max()*1e-5, wy > wy.max()*1e-5
        fx = vx[:, ix] * np.sqrt(wx[ix])
        fy = vy[:, iy] * np.sqrt(wy[iy])
        # Row-wise tensor features; Gram=Hadamard product of their two Grams.
        gram = (fx @ fx.T) * (fy @ fy.T)
        mean, variance = np.trace(gram), 2*np.sum(gram*gram)
        expected_p = gamma.sf(statistic, mean*mean/variance, scale=variance/mean)
        self.assertAlmostEqual(pvalue, expected_p, places=8)


if __name__ == "__main__":
    unittest.main()
