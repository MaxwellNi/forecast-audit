"""Independent rowwise, analytic and calibration tests for the FOD comparison."""
import unittest
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/analysis"))
import fod_comparison as study


def setUpModule():
    study.configure_legacy()


def rowwise_fod(x, y, used):
    values, weights = [], []
    for r in np.flatnonzero(used):
        c = np.sqrt((len(y[0])-r-1)/(len(y[0])-r))
        values.append((x[:, r]-x[:, :r].mean(axis=1)) * c * (y[:, r]-y[:, r+1:].mean(axis=1)))
        weights.append(c)
    return np.sum(values, axis=0)/sum(weights)


class FODChecks(unittest.TestCase):
    def test_matrix_equals_independent_rowwise_formula(self):
        matrices, used, _ = study.all_matrices()
        rng = np.random.default_rng(9132)
        x, y = rng.normal(size=(2, 23, 20))
        actual = np.einsum("gt,tu,gu->g", x, matrices[-1], y)
        np.testing.assert_allclose(actual, rowwise_fod(x, y, used), atol=1e-14)
        np.testing.assert_array_equal(np.flatnonzero(used), np.arange(4, 16))

    def test_classical_FOD_rows_are_orthonormal(self):
        length = 20
        transform = np.zeros((length-1, length))
        for r in range(length-1):
            c = np.sqrt((length-r-1)/(length-r))
            transform[r, r] = c
            transform[r, r+1:] = -c/(length-r-1)
        np.testing.assert_allclose(transform @ transform.T, np.eye(length-1), atol=1e-14)
        np.testing.assert_allclose(transform @ np.ones(length), 0., atol=1e-14)

    def test_null_and_alternative_expectations(self):
        matrices, _, _ = study.all_matrices()
        A = matrices[-1]
        np.testing.assert_allclose(A @ np.ones(20), 0., atol=1e-14)
        np.testing.assert_allclose(np.ones(20) @ A, 0., atol=1e-14)
        self.assertAlmostEqual(np.trace(A), 1.)
        # X-row indices are never later than Y-column indices, so all causal
        # lag-response matrices have zero Frobenius inner product with A.
        np.testing.assert_array_equal(np.tril(A, -1), 0.)
        for lag in range(1, 20):
            causal_covariance = np.eye(20, k=-lag)
            self.assertAlmostEqual(float(np.sum(A*causal_covariance)), 0.)
        for row in study.analytic_moments(matrices):
            if row["method"] == study.METHODS[-1]:
                self.assertAlmostEqual(row["exact_mean_affected_model"], row["signal"])
                self.assertAlmostEqual(row["exact_mean_unaffected_model"], 0.)

    def test_generator_matches_independent_primitives_and_injection(self):
        matrices, used, _ = study.all_matrices()
        b, k, G, T, p = 2, 4, 400, 20, 5
        rng = np.random.default_rng(6491)
        eps = rng.normal(size=(b, G, T+p))
        ay = rng.normal(size=(b, G, 1))
        ax = rng.normal(size=(b, k, G, 1))
        common = rng.normal(size=(b, 1, G, T))
        own = rng.normal(size=(b, k, G, T))
        X = np.empty((b, k, G, T))
        for r in range(T):
            X[..., r] = ax[..., 0] + .2*(p*ay[:, None, :, 0]+eps[:, None, :, r:r+p].sum(axis=-1)) + (common[..., r]+own[..., r])/np.sqrt(2.)
        Y = ay+eps[..., p:]
        for signal in (0., .06):
            actual = study.legacy.draw_scores(np.random.default_rng(6491), b, k, (signal,), matrices)[:, 0, -1]
            injected = X.copy()
            injected[:, :3] += signal*eps[:, None, :, p:]
            expected = np.stack([np.stack([rowwise_fod(injected[i, j], Y[i], used) for j in range(k)]) for i in range(b)])
            np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=1e-12)

    def test_target_normalization_is_scalar_and_preserves_statistic(self):
        A, c = study.fod_matrix()
        rng = np.random.default_rng(21)
        x, y = rng.normal(size=(2, 400, 20))
        normalized = np.einsum("gt,tu,gu->g", x, A, y)
        unnormalized = normalized*c.sum()/12
        self.assertAlmostEqual(float(study.legacy.statistic(normalized)[2]),
                               float(study.legacy.statistic(unnormalized)[2]))

    def test_rank_pvalues_and_BY(self):
        ref = np.array([-1., 0., 0., 2.])
        np.testing.assert_array_equal(study.legacy.calibration_p(np.array([-2., 0., 3.]), ref), [1., .8, .2])
        rng = np.random.default_rng(243)
        for _ in range(10):
            p = norm.sf(rng.normal(size=11)*3)
            np.testing.assert_allclose(study.legacy.by_adjust(p), multipletests(p, method="fdr_by")[1], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
