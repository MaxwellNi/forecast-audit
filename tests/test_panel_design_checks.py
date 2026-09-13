"""Regression checks of temporal and peer-sampling score expectations."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from panel_design_checks import (
    dense_score_expectation, feedback_expectation, fixed_bernoulli_check,
    resampled_bernoulli_check, time_directed_residuals,
)


class PanelDesignChecks(unittest.TestCase):
    def test_formula_matches_independent_covariance_contraction(self):
        rng = np.random.default_rng(819)
        for sizes in ([2, 5, 1, 3], [1, 1, 1], [4, 2, 4]):
            folds = np.repeat(np.arange(len(sizes)), sizes)
            n = len(folds)
            weights = np.tril(rng.normal(size=(n, n)), -1)
            formula = feedback_expectation(weights, folds, variance=2.5)
            dense = dense_score_expectation(weights, folds, 2.5*np.eye(n))
            self.assertAlmostEqual(formula["expected_numerator"], dense, places=12)
            self.assertEqual(dense_score_expectation(weights, folds, np.eye(n), True), 0)

    def test_serial_dependence_changes_temporal_expectation(self):
        folds = np.repeat(np.arange(3), 2)
        covariance = 0.8**abs(np.subtract.outer(np.arange(6), np.arange(6)))
        actual = dense_score_expectation(np.eye(6, k=-1), folds, covariance, True)
        self.assertAlmostEqual(actual, 0.35584, places=12)

    def test_actual_residuals_equal_dense_linear_score(self):
        rng = np.random.default_rng(32)
        folds = np.repeat(np.arange(3), 2)
        weights = np.eye(6, k=-1)
        shocks = rng.normal(size=6)
        outcome = 2+shocks
        forecast = weights @ outcome
        rows, left, right = time_directed_residuals(forecast, outcome, folds, 1)
        np.testing.assert_array_equal(rows, [2, 3])
        np.testing.assert_allclose(left, forecast[2:4]-forecast[:2].mean())
        np.testing.assert_allclose(right, shocks[2:4]-shocks[4:].mean())

    def test_fixed_and_resampled_designs_have_different_centering(self):
        same, opposite = fixed_bernoulli_check(), fixed_bernoulli_check(True)
        for result in (same, opposite):
            self.assertAlmostEqual(result["enumerated_probability"], 1)
            self.assertLess(result["maximum_identity_error"], 1e-14)
            self.assertAlmostEqual(result["shared_expected_score"], 0)
        self.assertAlmostEqual(same["distinct_expected_score"], -1/90)
        self.assertAlmostEqual(opposite["distinct_expected_score"], 1/90)
        resampled = resampled_bernoulli_check()
        self.assertAlmostEqual(resampled["enumerated_probability"], 1)
        self.assertAlmostEqual(resampled["distinct_expected_score"], 0)
        self.assertGreater(resampled["shared_expected_score"], 0)
        self.assertLess(resampled["maximum_identity_error"], 1e-14)

    def test_invalid_temporal_designs_are_rejected(self):
        weights, folds = np.eye(6, k=-1), np.repeat(np.arange(3), 2)
        for variance in (0, -1, float("nan")):
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                feedback_expectation(weights, folds, variance)
        with self.assertRaisesRegex(ValueError, "earlier rows only"):
            feedback_expectation(weights+np.eye(6), folds)
        with self.assertRaisesRegex(ValueError, "chronological"):
            feedback_expectation(weights, folds[::-1])
        with self.assertRaisesRegex(ValueError, "chronological"):
            feedback_expectation(weights, folds[::-1].astype(np.uint64))
        for label in (0, 2):
            with self.assertRaisesRegex(ValueError, "earlier and later"):
                time_directed_residuals(np.arange(6), np.arange(6), folds, label)
        with self.assertRaisesRegex(ValueError, "evaluation fold"):
            time_directed_residuals(np.arange(6), np.arange(6), folds, 4)
        with self.assertRaisesRegex(ValueError, "nonempty earlier and later"):
            dense_score_expectation(weights, [0, 0, 0, 1, 1, 1], np.eye(6), True)
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            dense_score_expectation(weights, folds, -np.eye(6))


if __name__ == "__main__":
    unittest.main()
