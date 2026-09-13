"""Decision-layer contracts; synthetic inputs only."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from audit_decisions import inspect_control_redundancy, apply_decision_policy


class ExactRedundancyTests(unittest.TestCase):
    def test_exact_copy_signed_zeros_and_one_ulp_difference(self):
        baseline = np.array([0., 1., 1., 3.])
        same = inspect_control_redundancy([-0., 1., 1., 3.], baseline, [0] * 4)
        self.assertTrue(same.exact_baseline_copy)
        perturbed = baseline.copy(); perturbed[-1] = np.nextafter(3., 4.)
        changed = inspect_control_redundancy(perturbed, baseline, [0] * 4)
        self.assertFalse(changed.exact_baseline_copy)
        self.assertTrue(changed.same_weak_order)

    def test_nonlinear_monotone_and_decreasing_transforms_preserve_ties(self):
        baseline = np.array([-2., 0., 0., 3., -1., 1., 2.])
        clusters = [0] * 4 + [1] * 3
        increasing = inspect_control_redundancy(baseline ** 3, baseline, clusters)
        decreasing = inspect_control_redundancy(-baseline ** 3, baseline, clusters)
        self.assertTrue(increasing.same_weak_order)
        self.assertTrue(decreasing.reversed_weak_order)
        self.assertTrue(increasing.abstain and decreasing.abstain)

    def test_changed_tie_partition_is_not_same_or_reversed_order(self):
        evidence = inspect_control_redundancy([1., 1., 2.], [1., 2., 3.], [0] * 3)
        self.assertFalse(evidence.abstain)

    def test_nonlinear_nonmonotone_relation_is_not_inferred_from_unique_controls(self):
        evidence = inspect_control_redundancy([4., 1., 0., 1., 4.], [-2., -1., 0., 1., 2.], [0] * 5)
        self.assertFalse(evidence.abstain)

    def test_cluster_constant_user_bias_and_singletons(self):
        evidence = inspect_control_redundancy([3., 3., 3., 8.], [1., 2., 3., 1.], [0, 0, 0, 1])
        self.assertTrue(evidence.zero_forecast_rank_variation)
        self.assertEqual(evidence.singleton_clusters, 1)
        singletons = inspect_control_redundancy([1., 2.], [5., 8.], [0, 1])
        self.assertTrue(singletons.zero_forecast_rank_variation)

    def test_one_singleton_does_not_override_other_clusters(self):
        evidence = inspect_control_redundancy([1., 3., 2., 7.], [1., 2., 3., 9.], [0, 0, 0, 1])
        self.assertFalse(evidence.abstain)

    def test_one_common_orientation_required(self):
        evidence = inspect_control_redundancy([1., 2., 3., 3., 2., 1.], [1., 2., 3.] * 2, [0] * 3 + [1] * 3)
        self.assertFalse(evidence.abstain)

    def test_row_permutation_and_cluster_renaming_invariance(self):
        forecast = np.array([4., 1., 1., 3., 9., 4.])
        baseline = np.array([2., 0., 0., 1., 3., 2.])
        labels = np.array([7, 7, 7, 2, 2, 2])
        order = np.array([5, 2, 4, 0, 3, 1])
        a = inspect_control_redundancy(forecast, baseline, labels)
        b = inspect_control_redundancy(forecast[order], baseline[order], labels[order] + 100)
        self.assertEqual(a, b)

    def test_invalid_vectors_are_rejected(self):
        for forecast, baseline, labels in [([], [], []), ([1, np.nan], [1, 2], [0, 0]),
                ([1, 2], [1, np.inf], [0, 0]), ([1, 2], [1], [0, 0]),
                ([1, 2], [1, 2], [0, None]), ([True, False], [1, 2], [0, 0]),
                ([1j, 2j], [1, 2], [0, 0])]:
            with self.assertRaises(ValueError):
                inspect_control_redundancy(forecast, baseline, labels)


class FamilyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.family = ["copy", "signal", "undefined"]
        self.evidence = {
            "copy": inspect_control_redundancy([1, 2, 3], [1, 2, 3], [0] * 3),
            "signal": inspect_control_redundancy([1, 3, 2], [1, 2, 3], [0] * 3),
            "undefined": inspect_control_redundancy([1, 3, 2], [1, 2, 3], [0] * 3),
        }
        self.profile = pd.DataFrame({"model": self.family, "beta": [1] * 3,
            "p_one_sided": [.001, .005, 1.], "statistic": [3.1, 2.57, np.nan],
            "inference_status": ["computed", "computed", "undefined_rank_or_scale"]})

    def apply(self, profile=None):
        return apply_decision_policy(self.profile if profile is None else profile,
            self.evidence, family_models=self.family, specification_columns=["beta"])

    def test_full_family_and_raw_values_preserved_with_independent_by(self):
        result = self.apply()
        pd.testing.assert_frame_equal(result[self.profile.columns], self.profile)
        np.testing.assert_array_equal(result.p_policy, [1., .005, 1.])
        oracle = multipletests(result.p_policy, method="fdr_by", alpha=.05)
        np.testing.assert_array_equal(result.policy_by_reject, oracle[0])
        np.testing.assert_allclose(result.policy_by_adjusted_p, oracle[1])
        self.assertTrue((result.policy_family_size == 3).all())
        self.assertEqual(result.policy_decision.tolist(), ["ABSTAIN_OBSERVED_CONTROL_REDUNDANCY",
            "NOMINAL_POSITIVE", "ABSTAIN_UNDEFINED_AUDIT"])

    def test_outcome_induced_raw_p_change_cannot_make_copy_positive(self):
        for pvalue in [0., 1e-300, .049, .9, 1.]:
            frame = self.profile.copy(); frame.loc[0, "p_one_sided"] = pvalue
            result = self.apply(frame)
            self.assertEqual(result.loc[0, "p_policy"], 1.)
            self.assertFalse(result.loc[0, "policy_nominal_positive"])
            self.assertFalse(result.loc[0, "policy_by_reject"])

    def test_pvalue_increase_and_rejection_containment_many_families(self):
        rng = np.random.default_rng(20260905)
        for _ in range(100):
            frame = self.profile.copy()
            frame.loc[:1, "p_one_sided"] = rng.uniform(0, .1, 2)
            result = self.apply(frame)
            self.assertTrue((result.p_policy >= result.p_one_sided).all())
            self.assertFalse((result.policy_by_reject & ~result.raw_by_reject_recomputed).any())

    def test_family_cannot_be_filtered_or_duplicated(self):
        for frame in [self.profile.iloc[:2], pd.concat([self.profile, self.profile.iloc[:1]])]:
            with self.assertRaises(ValueError): self.apply(frame)

    def test_each_specification_retains_full_family(self):
        frame = pd.concat([self.profile, self.profile.assign(beta=2)], ignore_index=True)
        result = self.apply(frame)
        self.assertEqual(len(result), 6)
        self.assertTrue((result.groupby("beta").policy_family_size.first() == 3).all())
        with self.assertRaises(ValueError): self.apply(frame.iloc[:-1])

    def test_undefined_score_abstains_even_if_given_small_raw_p(self):
        frame = self.profile.copy(); frame.loc[2, "p_one_sided"] = .001
        result = self.apply(frame)
        self.assertEqual(result.loc[2, "p_policy"], 1.)
        self.assertTrue(result.loc[2, "undefined_audit_abstain"])

    def test_boundary_and_invalid_pvalues(self):
        frame = self.profile.copy(); frame.loc[1, "p_one_sided"] = .05
        self.assertFalse(self.apply(frame).loc[1, "policy_nominal_positive"])
        for value in [np.nan, -.1, 1.1]:
            frame.loc[1, "p_one_sided"] = value
            with self.assertRaises(ValueError): self.apply(frame)


if __name__ == "__main__":
    unittest.main()
