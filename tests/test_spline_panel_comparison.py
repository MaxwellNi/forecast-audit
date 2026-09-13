"""Independent linear-algebra and target-isolation checks for spline extension."""
import sys
from pathlib import Path
import unittest

import numpy as np
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
import spline_panel_comparison as study


class SplinePanelTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(920260907)
        entity = np.tile(np.arange(7), 15)
        clusters = np.repeat(np.arange(15), 7)
        folds = clusters // 5
        z = rng.uniform(size=len(entity))
        y = np.column_stack([np.sin(5 * z) + entity / 9, z ** 2 - entity / 7]) + rng.normal(0, .1, (len(z), 2))
        return y, z, entity, folds, clusters

    def test_against_dense_penalized_dummy_design(self):
        y, z, entity, folds, clusters = self.fixture()
        got, _ = study.fit_spline(y, z, entity, folds, clusters)
        for fold in np.unique(folds):
            tr, te = folds != fold, folds == fold
            spline = SplineTransformer(n_knots=8, degree=3, knots="quantile", extrapolation="linear", include_bias=False)
            basis = spline.fit_transform(z[tr, None])
            design = np.column_stack([np.eye(7)[entity[tr]], basis])
            penalty = np.diag(np.r_[np.zeros(7), np.full(basis.shape[1], np.sqrt(.001))])
            coefficient = np.linalg.lstsq(np.vstack([design, penalty]), np.vstack([y[tr], np.zeros((design.shape[1], 2))]), rcond=None)[0]
            reference = np.column_stack([np.eye(7)[entity[te]], spline.transform(z[te, None])]) @ coefficient
            np.testing.assert_allclose(got[te], reference, atol=2e-12, rtol=2e-12)

    def test_evaluation_target_isolation_and_unseen_entity(self):
        y, z, entity, folds, clusters = self.fixture()
        entity[(folds == 0) & (entity == 0)] = 99
        first, diagnostics = study.fit_spline(y, z, entity, folds, clusters)
        changed = y.copy()
        changed[folds == 0] += 100
        second, _ = study.fit_spline(changed, z, entity, folds, clusters)
        np.testing.assert_array_equal(first[folds == 0], second[folds == 0])
        self.assertEqual(diagnostics[0]["unseen_evaluation_rows"], 5)

    def test_constant_control_fallback_and_split_rejection(self):
        y, z, entity, folds, clusters = self.fixture()
        prediction, diagnostics = study.fit_spline(y, np.full(len(z), .5), entity, folds, clusters)
        for fold in np.unique(folds):
            for level in np.unique(entity):
                np.testing.assert_allclose(prediction[(folds == fold) & (entity == level)], np.broadcast_to(y[(folds != fold) & (entity == level)].mean(axis=0), (5, 2)), atol=1e-14)
        self.assertTrue(all(item["distinct_knots"] == 1 for item in diagnostics))
        bad = folds.copy()
        bad[0] = 1
        with self.assertRaises(ValueError):
            study.fit_spline(y, z, entity, bad, clusters)

    def test_independent_irregular_calendar_covariance(self):
        products = np.array([.2, -.1, .4, .8, -.3])
        labels = np.array(["2020-01-01", "2020-01-01", "2020-03-01", "2020-05-01", "2020-05-01"])
        mean, se = study.verify_scores(products, labels, 2, "M")
        explicit = np.array([.1 - 2 * mean, 0, .4 - mean, 0, .5 - 2 * mean])
        covariance = np.eye(5)
        for i in range(5):
            for j in range(5):
                covariance[i, j] = max(0, 1 - abs(i - j) / 3)
        self.assertAlmostEqual(se, np.sqrt(explicit @ covariance @ explicit) / 5, places=15)


if __name__ == "__main__":
    unittest.main()
