"""Independent behavioral checks for the new reference operations."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "cluster_covariance_reference.py"
)
SPEC = importlib.util.spec_from_file_location("cluster_covariance_reference", MODULE_PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class AdditiveNuisanceTests(unittest.TestCase):
    @staticmethod
    def panel():
        # Each of four independent clusters contains every entity/bin pair.
        entity = np.tile(np.repeat(np.arange(3), 2), 4)
        bins = np.tile(np.arange(2), 12)
        cluster = np.repeat(np.arange(4), 6)
        folds = cluster % 2
        y = 7.0 + np.array([-2., 0., 2.])[entity] + np.array([-1., 1.])[bins]
        return y, entity, bins, cluster, folds

    def test_known_additive_signal_is_recovered(self):
        y, entity, bins, cluster, folds = self.panel()
        predicted = reference.crossfit_additive_nuisance(
            y, [entity, bins], folds, cluster_ids=cluster
        )
        np.testing.assert_allclose(predicted, y, atol=1e-12)

    def test_evaluation_targets_cannot_change_their_nuisance_predictions(self):
        y, entity, bins, cluster, folds = self.panel()
        baseline = reference.crossfit_additive_nuisance(
            y, [entity, bins], folds, cluster_ids=cluster
        )
        changed = y.copy()
        held_out = folds == 0
        changed[held_out] += np.linspace(-100., 300., held_out.sum())
        predicted = reference.crossfit_additive_nuisance(
            changed, [entity, bins], folds, cluster_ids=cluster
        )
        np.testing.assert_array_equal(predicted[held_out], baseline[held_out])

    def test_cluster_cannot_cross_folds(self):
        y, entity, bins, cluster, folds = self.panel()
        folds[0] = 1
        with self.assertRaisesRegex(ValueError, "wholly"):
            reference.crossfit_additive_nuisance(
                y, [entity, bins], folds, cluster_ids=cluster
            )

    def test_unseen_group_uses_training_intercept(self):
        # The evaluated group has no training rows in either fold.
        predicted = reference.crossfit_additive_nuisance(
            [1., 3., 10., 14.], [["a", "a", "b", "b"]], [0, 0, 1, 1]
        )
        np.testing.assert_allclose(predicted, [12., 12., 2., 2.])

    def test_unbalanced_additive_fit_matches_independent_least_squares(self):
        entity = np.array([0, 0, 0, 1, 1, 2, 2, 2, 0, 1, 2, 2])
        bins = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1])
        folds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        y = np.array([1., 2., 5., 4., 6., 8., 10., 12., 7., 3., 9., 11.])
        predicted = reference.crossfit_additive_nuisance(y, [entity, bins], folds)
        # Independent dummy-matrix fit, with one reference category per block.
        design = np.column_stack([np.ones(len(y)), entity == 1, entity == 2, bins == 1])
        for fold in (0, 1):
            training = folds != fold
            beta, *_ = np.linalg.lstsq(design[training], y[training], rcond=None)
            expected = design[~training] @ beta
            np.testing.assert_allclose(predicted[~training], expected, atol=1e-8)


class ClusterScaleTests(unittest.TestCase):
    def test_equal_cluster_sizes_match_hand_calculation(self):
        # Cluster sums 4 and 10; n_g * mean = 7 for each cluster.
        actual = reference.cluster_standard_error([1., 3., 4., 6.], [0, 0, 1, 1])
        self.assertAlmostEqual(actual, np.sqrt(18.0) / 4.0)

    def test_unequal_cluster_sizes_use_observation_weighted_centring(self):
        # Cluster sums 2 and 10; mean=4, sizes 1 and 2, centred totals -2 and 2.
        actual = reference.cluster_standard_error([2., 4., 6.], [0, 1, 1])
        self.assertAlmostEqual(actual, np.sqrt(8.0) / 3.0)
        self.assertNotAlmostEqual(actual, np.sqrt(32.0) / 3.0)
        self.assertAlmostEqual(
            reference.cluster_t_statistic([2., 4., 6.], [0, 1, 1]),
            12.0 / np.sqrt(8.0),
        )

    def test_zero_cluster_variance_refuses_inference(self):
        # Unequal sizes, with exactly the same mean in both clusters.
        products = [2., 1., 3.]
        self.assertEqual(reference.cluster_standard_error(products, [0, 1, 1]), 0.)
        with self.assertRaisesRegex(ValueError, "zero"):
            reference.cluster_t_statistic(products, [0, 1, 1])

    def test_single_cluster_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "two nonempty clusters"):
            reference.cluster_standard_error([1., 2.], [0, 0])

    def test_roundoff_does_not_create_a_constant_signal_statistic(self):
        self.assertEqual(reference.cluster_standard_error([.1, .1, .1], [0, 1, 1]), 0.)
        with self.assertRaisesRegex(ValueError, "numerical precision"):
            reference.cluster_t_statistic([.2, .1, .3], [0, 1, 1])


class FutureReturnTests(unittest.TestCase):
    def test_three_month_window_uses_only_subsequent_months(self):
        actual = reference.future_compound_return([.9, .1, .2, .3, .4], 3)
        np.testing.assert_allclose(actual[:2], [.716, 1.184], atol=1e-12)
        self.assertTrue(np.isnan(actual[2:]).all())

    def test_missing_future_month_invalidates_the_whole_window(self):
        actual = reference.future_compound_return([.9, .1, np.nan, .3, .4, .5], 2)
        expected = [np.nan, np.nan, .82, 1.10, np.nan, np.nan]
        np.testing.assert_allclose(actual, expected, atol=1e-12, equal_nan=True)

    def test_one_month_horizon_and_short_series_boundary(self):
        np.testing.assert_allclose(
            reference.future_compound_return([.4, -.2, .1], 1),
            [-.2, .1, np.nan], equal_nan=True,
        )
        self.assertTrue(np.isnan(reference.future_compound_return([.1, .2], 3)).all())

    def test_invalid_horizon_is_rejected(self):
        for horizon in (0, -1, 1.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                reference.future_compound_return([.1, .2], horizon)


if __name__ == "__main__":
    unittest.main()
