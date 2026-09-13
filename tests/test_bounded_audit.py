"""Independent formula, exact probability, and counterexample checks."""
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from bounded_audit import (bounded_mean_certificate, project_cluster_rms,
                           rank_cluster_products)


class BoundedAuditTests(unittest.TestCase):
    def test_unequal_weight_formula_and_alpha_spending(self):
        result = bounded_mean_certificate([0.2, 0.8, -0.1], [-1, 0, -2], [1, 1, 2],
                                          cluster_weights=[2, 3, 5], bias_upper=.04,
                                          alpha=.1, envelope_failure_probability=.02)
        weight = [0.2, .3, .5]
        mean = sum(a*b for a, b in zip(weight, [.2, .8, -.1]))
        v = sum((a*b)**2 for a, b in zip(weight, [2, 1, 4]))
        radius = math.sqrt(v * math.log(1/.08) / 2)
        self.assertAlmostEqual(result.weighted_mean, mean)
        self.assertAlmostEqual(result.lower_bound, mean-.04-radius)
        self.assertAlmostEqual(result.p_one_sided, min(1., .02+math.exp(-2*(mean-.04)**2/v)))
        self.assertAlmostEqual(result.effective_clusters, 1/sum(a*a for a in weight))

    def test_exact_binomial_null_superuniformity_and_coverage(self):
        # Enumerate exact finite distributions, with no normal approximation.
        for n in [1, 3, 12, 40]:
            for probability in [.05, .3, .5, .8]:
                theta = 0.0
                bias = probability
                for alpha in [.01, .05, .2]:
                    reject_probability = 0.0
                    coverage_failure = 0.0
                    for count in range(n+1):
                        scores = [1.] * count + [0.] * (n-count)
                        result = bounded_mean_certificate(scores, 0, 1, bias_upper=bias, alpha=alpha)
                        mass = math.comb(n, count) * probability**count * (1-probability)**(n-count)
                        reject_probability += mass * (result.p_one_sided <= alpha)
                        coverage_failure += mass * (result.lower_bound > theta)
                    self.assertLessEqual(reject_probability, alpha + 1e-13)
                    self.assertLessEqual(coverage_failure, alpha + 1e-13)

    def test_exact_nonidentical_unequal_cluster_distribution(self):
        probabilities, weights = [.1, .4, .8, .2], [1, 2, 4, 7]
        mu = sum(p*w for p, w in zip(probabilities, weights))/sum(weights)
        for alpha in [.01, .05, .3]:
            failure = 0.0
            for outcomes in itertools.product([0, 1], repeat=4):
                probability = math.prod(p if x else 1-p for p, x in zip(probabilities, outcomes))
                result = bounded_mean_certificate(outcomes, 0, 1, cluster_weights=weights,
                                                  bias_upper=mu, alpha=alpha)
                failure += probability * result.reject
            self.assertLessEqual(failure, alpha + 1e-13)

    def test_probabilistic_envelope_failure_budget(self):
        # An external coin gives a false bias envelope with probability delta.
        # On that branch arbitrarily certain false rejection is allowed; the
        # p-value still cannot have probability > alpha below alpha.
        delta, n, mu = .02, 30, .5
        for alpha in [.03, .05, .2]:
            rejected = 0.0
            for bad, external_mass in [(False, 1-delta), (True, delta)]:
                for count in range(n+1):
                    probability = math.comb(n, count) / 2**n
                    result = bounded_mean_certificate([1]*count+[0]*(n-count), 0, 1,
                              bias_upper=-10 if bad else mu, alpha=alpha,
                              envelope_failure_probability=delta)
                    rejected += external_mass * probability * (result.p_one_sided <= alpha)
            self.assertLessEqual(rejected, alpha + 1e-13)
            self.assertGreaterEqual(rejected, delta - 1e-13)

    def test_sharp_rank_bound_with_ties(self):
        n = 80
        ranks = rankdata([0.] * (n-1) + [1.])
        ranks = (ranks-ranks.mean())/ranks.std(ddof=1)
        self.assertAlmostEqual(max(ranks), (n-1)/math.sqrt(n))
        self.assertGreater(max(ranks), math.sqrt(3))
        self.assertAlmostEqual(np.mean(ranks**2), (n-1)/n)

    def test_projection_and_independent_product_reconstruction(self):
        groups = np.array([0]*3+[1]*8+[2])
        x, y = np.zeros(12), np.zeros(12)
        for g in [0, 1]:
            mask = groups == g
            rank = rankdata(np.arange(mask.sum()))
            x[mask] = (rank-rank.mean())/rank.std(ddof=1)
            y[mask] = -x[mask]
        fx = np.column_stack([np.linspace(-20, 30, 12), np.full(12, 1e300)])
        fy = np.column_stack([np.linspace(4, -7, 12), np.zeros(12)])
        weights = [-1/3, 4/3]
        result = rank_cluster_products(x, y, fx, fy, groups, weights)
        expected = []
        for g in [0, 1, 2]:
            mask = groups == g
            value = 0.
            for j, w in enumerate(weights):
                a, b = fx[mask, j].copy(), fy[mask, j].copy()
                # Independent normalisation uses Python's scaled hypot.
                na = math.hypot(*a)/math.sqrt(len(a))
                nb = math.hypot(*b)/math.sqrt(len(b))
                a /= max(1., na)
                b /= max(1., nb)
                value += w * sum((x[mask]-a)*(y[mask]-b))/len(a)
            expected.append(value)
        np.testing.assert_allclose(result["cluster_scores"], expected, atol=1e-13)
        for key in ["projected_forecast_fits", "projected_outcome_fits"]:
            for g in [0, 1, 2]:
                self.assertTrue(np.all(np.mean(result[key][groups == g]**2, axis=0) <= 1+1e-14))
        self.assertLessEqual(max(abs(result["cluster_scores"])), result["score_upper"])

    def test_outcome_changes_cannot_change_fit_projection(self):
        fits = np.array([[7., -2.], [1., 5.], [3., 4.]])
        before = project_cluster_rms(fits, [0, 0, 1])
        # The projector has no target argument and does not mutate its input.
        np.testing.assert_array_equal(fits, [[7., -2.], [1., 5.], [3., 4.]])
        np.testing.assert_array_equal(before, project_cluster_rms(fits, [0, 0, 1]))

    def test_degenerate_bounds_and_invalid_contracts(self):
        positive = bounded_mean_certificate([1, 2], [1, 2], [1, 2], bias_upper=0)
        self.assertTrue(positive.reject)
        self.assertEqual(positive.p_one_sided, 0)
        zero = bounded_mean_certificate([1, 2], [1, 2], [1, 2], bias_upper=1.5)
        self.assertFalse(zero.reject)
        self.assertEqual(zero.p_one_sided, 1)
        for kwargs in [{"bias_upper": 0, "alpha": .02, "envelope_failure_probability": .02},
                       {"bias_upper": float("nan")},
                       {"bias_upper": 0, "cluster_weights": [1, -1]},
                       {"bias_upper": 0, "cluster_weights": [0, 0]}]:
            with self.assertRaises(ValueError):
                bounded_mean_certificate([0, 1], 0, 1, **kwargs)
        with self.assertRaises(ValueError):
            bounded_mean_certificate([2], 0, 1, bias_upper=0)
        with self.assertRaises(ValueError):
            rank_cluster_products([2], [0], [0], [0], [0], [1])

    def test_extreme_finite_ranges_and_tiny_alpha(self):
        for scale in [1e-200, 1e200]:
            result = bounded_mean_certificate([0, scale], 0, scale, bias_upper=0)
            self.assertTrue(np.isfinite(result.concentration_radius))
            self.assertGreater(result.concentration_radius, 0)
            self.assertAlmostEqual(result.concentration_radius/scale, math.sqrt(math.log(20)/4))
            self.assertAlmostEqual(result.p_one_sided, math.exp(-1))
            self.assertFalse(result.reject)
        result = bounded_mean_certificate([0, 1], 0, 1, bias_upper=0, alpha=1e-320)
        self.assertTrue(np.isfinite(result.concentration_radius))
        self.assertAlmostEqual(result.concentration_radius, math.sqrt(-math.log(1e-320)/4))

    def test_constant_null_with_zero_or_one_ulp_ranges_never_certifies(self):
        for constant in [.1, -.1, 1., -1., 1e-200, 1e200]:
            for count in [5, 6, 9, 31]:
                for bounds in [(constant, constant),
                               (constant, np.nextafter(constant, np.inf)),
                               (np.nextafter(constant, -np.inf), np.nextafter(constant, np.inf))]:
                    result = bounded_mean_certificate([constant]*count, *bounds, bias_upper=constant)
                    self.assertFalse(result.reject, (constant, count, bounds))
                    self.assertLessEqual(result.lower_bound, 0)
                    self.assertEqual(result.p_one_sided, 1)
                weights = np.arange(1, count+1, dtype=float)
                result = bounded_mean_certificate([constant]*count, constant, constant,
                                                  bias_upper=constant, cluster_weights=weights)
                self.assertEqual(result.lower_bound, 0)
                self.assertEqual(result.p_one_sided, 1)

    def test_outward_rounded_boundary_and_unrepresentable_lower_bound(self):
        from decimal import Decimal, localcontext
        from fractions import Fraction
        scores = [.2, .3, .7, .9, .4]
        result = bounded_mean_certificate(scores, 0., 1., bias_upper=-.1)
        exact_mean = sum((Fraction.from_float(x) for x in scores), Fraction())/len(scores)
        exact_gap = exact_mean-Fraction.from_float(-.1)
        with localcontext() as context:
            context.prec = 100
            a = Decimal.from_float(.05)
            radius = (-a.ln()/(2*len(scores))).sqrt()
            gap = Decimal(exact_gap.numerator)/Decimal(exact_gap.denominator)
            self.assertLessEqual(Decimal.from_float(result.lower_bound), gap-radius)
            self.assertGreaterEqual(Decimal.from_float(result.p_one_sided), (-2*len(scores)*gap*gap).exp())
        with self.assertRaises(ValueError):
            bounded_mean_certificate([-1e308], -1e308, 0, bias_upper=0)

    def test_independent_split_does_not_remove_approximation_bias(self):
        # X=Y=Z, Z uniform [-1,1], is conditionally independent given Z.
        # A fixed zero nuisance has residual-product mean 1/3 despite honesty.
        from numpy.polynomial import Polynomial
        product = Polynomial([0, 0, 1]).integ()
        self.assertAlmostEqual((product(1)-product(-1))/2, 1/3)

    def test_projection_can_destroy_extrapolation_bias_cancellation(self):
        # X=Y=0, f_q=2/q at q=1,2. The original bias 4/q^2 cancels.
        # Projection changes both fits to one and leaves positive bias one.
        original = (-1/3)*2**2+(4/3)*1**2
        fit = project_cluster_rms([[2, 1]], [0])
        projected = (-1/3)*fit[0, 0]**2+(4/3)*fit[0, 1]**2
        self.assertEqual(original, 0)
        self.assertEqual(projected, 1)

    def test_rank_helper_exact_norm_and_resolution_weight_contracts(self):
        from fractions import Fraction
        subtly_invalid = [np.nextafter(1., np.inf), np.nextafter(1., 0.)]
        self.assertEqual(np.mean(np.square(subtly_invalid)), 1.)
        with self.assertRaises(ValueError):
            rank_cluster_products(subtly_invalid, [0, 0], [0, 0], [0, 0], [0, 0], [1])
        with self.assertRaises(ValueError):
            # A floating sum of one hides an exact sum of zero.
            rank_cluster_products([0], [0], [[0]*4], [[0]*4], [0], [1e16, -1, -1e16, 1])
        for radius in [1., .1, np.nextafter(0., 1.)]:
            projected = project_cluster_rms([1., 2., -3.], [0, 0, 0], radius=radius)
            exact_norm = sum((Fraction.from_float(float(value))**2 for value in projected), Fraction())
            self.assertLessEqual(exact_norm, 3*Fraction.from_float(float(radius))**2)
        x, y = [1., -1., .1], [-1., .2, .3]
        a, b = [[.2, -.1], [.4, .3], [.1, .2]], [[.1, .5], [-.2, .3], [.2, .1]]
        weights = [-1/3, 4/3]
        result = rank_cluster_products(x, y, a, b, [0, 0, 0], weights)
        total = sum((Fraction.from_float(w) for w in weights), Fraction())
        exact = Fraction()
        for j, w in enumerate(weights):
            score = sum(((Fraction.from_float(xi)-Fraction.from_float(ai[j]))*
                         (Fraction.from_float(yi)-Fraction.from_float(bi[j]))
                         for xi, yi, ai, bi in zip(x, y, a, b)), Fraction())/3
            exact += Fraction.from_float(w)/total*score
        self.assertLessEqual(Fraction.from_float(float(result["cluster_scores"][0])), exact)
        l1 = sum((abs(Fraction.from_float(w)/total) for w in weights), Fraction())
        self.assertGreaterEqual(Fraction.from_float(result["score_upper"]), 4*l1)


if __name__ == "__main__":
    unittest.main()
