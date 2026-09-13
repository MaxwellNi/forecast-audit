"""Meaningful cache-equivalence and paired-design checks on unused fixtures."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/analysis"))
import current_core_sensitivity as sensitivity


class CurrentCoreSensitivityTests(unittest.TestCase):
    def test_cached_score_is_exact_full_current_core_for_multiple_ladders(self):
        primitives = sensitivity.primitive_draw(4879101, max_clusters=10, entities=12)
        for frame in sensitivity.frames_from_primitives(primitives, 10):
            cache = sensitivity.cached_products(frame, [2, 3, 4, 6])
            for qs in [(2, 3, 4), (3, 4, 6)]:
                for beta in [.5, 1., 2.]:
                    cached = sensitivity.from_cached_products(frame, cache, qs, beta)
                    full = sensitivity.core.audit_extrapolated_panel(frame, 0, None, qs=qs, betas=(beta,))[0]
                    for key in ["mean_product", "standard_error", "statistic", "p_one_sided"]:
                        np.testing.assert_equal(cached[key], full[key])
                    self.assertEqual(cached["inference_status"], full["inference_status"])

    def test_cluster_prefix_pairing_and_fixed_grid_control(self):
        primitives = sensitivity.primitive_draw(4879102, max_clusters=10, entities=12)
        short = sensitivity.frames_from_primitives(primitives, 5)
        long = sensitivity.frames_from_primitives(primitives, 10)
        for left, right in zip(short, long):
            np.testing.assert_array_equal(left.to_numpy(), right.iloc[:len(left)].to_numpy())
        grid = np.linspace(-1, 1, 12)
        for values in primitives[-1]:
            np.testing.assert_allclose(np.sort(values)-values.mean(), grid, atol=1e-15)
        self.assertEqual(sum(len(sensitivity.configurations(m))*3 for m in sensitivity.CLUSTERS), 33)

    def test_covariance_formula_uses_independent_clusters_not_rows(self):
        frame = sensitivity.frames_from_primitives(sensitivity.primitive_draw(4879103, 10, 12), 10)[0]
        product = np.repeat(np.arange(10, dtype=float), 12)
        cache = {q: {"product": product, "numerical_zero": False, "nuisance_r2": 0.} for q in [2, 3, 4]}
        value = sensitivity.from_cached_products(frame, cache, (2, 3, 4), 1.)
        np.testing.assert_allclose(value["standard_error"], np.sqrt(np.var(np.arange(10.))/10), atol=1e-14)
        self.assertGreater(value["standard_error"], np.std(product)/np.sqrt(len(product)))


if __name__ == "__main__":
    unittest.main()
