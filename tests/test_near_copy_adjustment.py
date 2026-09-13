"""Check the optimized synthetic fits against the released implementations."""
from pathlib import Path
import sys
import unittest

import numpy as np

import near_copy_adjustment as study

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/"scripts/analysis"
sys.path.insert(0, str(SCRIPTS))
import audit_panel_predictions as core
import spline_panel_comparison as spline


class ControlChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(92386)
        self.order = np.stack([rng.permutation(80) for _ in range(25)])
        self.values = rng.normal(size=(25, 80, 2))
        self.entities = np.tile(np.arange(80), 25)
        self.periods = np.repeat(np.arange(25), 80)
        self.folds = np.repeat(np.arange(5), 400)

    def test_bins_match_released_sparse_solver(self):
        for q in (8, 32):
            basis = study.baseline_bases()[f"bins{q}"]
            actual = study.fit_additive(self.values, basis[self.order], np.zeros(q)).reshape(-1, 2)
            bins = np.minimum((q*(self.order.ravel()+.5)/80).astype(int), q-1)
            for target in range(2):
                reference = core.crossfit_additive_least_squares(self.values[:, :, target].ravel(),
                    [self.entities, bins], self.folds, self.periods)
                np.testing.assert_allclose(actual[:, target], reference, atol=2e-10, rtol=1e-9)

    def test_spline_matches_released_solver(self):
        basis = study.baseline_bases()["spline"]
        actual = study.fit_additive(self.values, basis[self.order], np.full(basis.shape[1], .001)).reshape(-1, 2)
        reference, _ = spline.fit_spline(self.values.reshape(-1, 2), (self.order.ravel()+.5)/80,
                                        self.entities, self.folds, self.periods)
        np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=1e-9)

    def test_unpenalized_linear_term_reproduces_copy(self):
        basis = study.baseline_bases()["linear_plus_spline"]
        pct = (self.order+.5)/80
        values = np.stack([3*pct-2, self.values[:, :, 0]], axis=-1)
        penalty = np.r_[0., np.full(basis.shape[1]-1, .001)]
        actual = study.fit_additive(values, basis[self.order], penalty)
        np.testing.assert_allclose(actual[:, :, 0], values[:, :, 0], atol=2e-11)

    def test_evaluation_target_does_not_enter_its_fit(self):
        basis = study.baseline_bases()["linear_plus_spline"]
        penalty = np.r_[0., np.full(basis.shape[1]-1, .001)]
        original = study.fit_additive(self.values, basis[self.order], penalty)
        changed = self.values.copy(); changed[:5] += 1000
        actual = study.fit_additive(changed, basis[self.order], penalty)
        np.testing.assert_array_equal(original[:5], actual[:5])


if __name__ == "__main__": unittest.main()
