"""Integrity checks for the paired extension, without inspecting study outcomes."""
import sys
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from canonical_baselines import METHODS, Q_LADDER, WEIGHTS
from canonical_beta2 import (assert_replay_pair, exponent_weights,
                                      validate_draw_rows)


class PairedExtensionTests(unittest.TestCase):
    def test_weights_equal_closed_form_intercept_and_cancel_term(self):
        for beta in (1, 2):
            x = Q_LADDER.astype(float) ** -beta
            expected = (np.sum(x * x) - np.sum(x) * x) / (len(x) * np.sum(x * x) - np.sum(x) ** 2)
            actual = exponent_weights(beta)
            assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
            assert_allclose(actual.sum(), 1, atol=1e-13)
            assert_allclose(actual @ x, 0, atol=1e-13)
        assert_allclose(exponent_weights(1), WEIGHTS, atol=1e-13)

    def test_replay_guard_rejects_changed_statistic_and_decision(self):
        stored = {"statistic": "0", "pvalue": "0.5", "reject": "0"}
        self.assertEqual(assert_replay_pair((0., .5), stored), 0.)
        with self.assertRaises(AssertionError):
            assert_replay_pair((.1, .5), stored)
        with self.assertRaises(AssertionError):
            assert_replay_pair((0., .5), dict(stored, reject="1"))

    def test_pairing_guard_rejects_missing_duplicate_or_different_draw(self):
        rows = [{"method": method, "sample_sha256": "sample", "fold_sha256": "fold",
                 "partial_distance_covariance": "0", "partial_distance_correlation": "0"}
                for method in METHODS]
        validate_draw_rows(rows, "sample", "fold")
        with self.assertRaises(AssertionError):
            validate_draw_rows(rows[:-1], "sample", "fold")
        with self.assertRaises(AssertionError):
            validate_draw_rows(rows[:-1] + [rows[0]], "sample", "fold")
        with self.assertRaises(AssertionError):
            validate_draw_rows(rows, "other-sample", "fold")
        with self.assertRaises(AssertionError):
            validate_draw_rows(rows, "sample", "other-fold")


if __name__ == "__main__":
    unittest.main()
