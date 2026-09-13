"""Independent exact finite-support checks of distinct-reference identities.

Does not import the proposed implementation or simulation generator. Enumeration
uses Fraction arithmetic; Gaussian and fixed-grid references use deterministic
quadrature/formulas. These checks supplement, rather than replace, the proof.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def comparison(x, reference):
    return F(int(reference < x)) + F(int(reference == x), 2)


def distribution(alternative):
    # Z~Bernoulli(1/2), X=Z xor ex, Y=Z xor ey. Null noises are
    # conditionally independent with flip probabilities 1/4 and 1/3.
    p11 = F(1, 12) + (F(1, 24) if alternative else 0)
    flips = {(0, 0): F(1) - F(1, 4) - F(1, 3) + p11,
             (1, 0): F(1, 4) - p11, (0, 1): F(1, 3) - p11,
             (1, 1): p11}
    return [((z ^ ex, z ^ ey, z), p / 2)
            for z in (0, 1) for (ex, ey), p in flips.items()]


def population(atoms, misspecified):
    fx = {x: sum(p * comparison(x, a[0]) for a, p in atoms) for x in (0, 1)}
    fy = {y: sum(p * comparison(y, a[1]) for a, p in atoms) for y in (0, 1)}
    mx = {z: sum(p * fx[a[0]] for a, p in atoms if a[2] == z) * 2 for z in (0, 1)}
    my = {z: sum(p * fy[a[1]] for a, p in atoms if a[2] == z) * 2 for z in (0, 1)}
    hx = {z: mx[z] + (F(1, 16) if misspecified else 0) for z in mx}
    hy = {z: my[z] - (F(1, 24) if misspecified else 0) for z in my}
    theta = sum(p * (fx[x] - mx[z]) * (fy[y] - my[z]) for (x, y, z), p in atoms)
    bias = sum(p * (mx[z] - hx[z]) * (my[z] - hy[z]) for (_, _, z), p in atoms)
    gamma = sum(p * (sum(q * comparison(x, a[0]) * comparison(y, a[1])
                              for a, q in atoms) - fx[x] * fy[y])
                for (x, y, _), p in atoms)
    return hx, hy, theta, bias, gamma


def cluster_scores(rows, hx, hy):
    n = len(rows)
    naive = F(0)
    triple = F(0)
    compact = F(0)
    for i, (x, y, z) in enumerate(rows):
        refs = [j for j in range(n) if i != j]
        a = {j: comparison(x, rows[j][0]) for j in refs}
        b = {j: comparison(y, rows[j][1]) for j in refs}
        ux, uy = sum(a.values()) / (n - 1), sum(b.values()) / (n - 1)
        q = (sum(a.values()) * sum(b.values()) - sum(a[j] * b[j] for j in refs)) / ((n - 1) * (n - 2))
        naive += (ux - hx[z]) * (uy - hy[z]) / n
        compact += (q - hx[z] * uy - hy[z] * ux + hx[z] * hy[z]) / n
        triple += sum((a[j] - hx[z]) * (b[k] - hy[z])
                      for j in refs for k in refs if j != k) / (n * (n - 1) * (n - 2))
    assert compact == triple
    assert -1 <= triple <= 1
    return naive, triple


def exact_enumeration():
    results = []
    for alternative, misspecified, n in itertools.product((False, True), (False, True), (3, 4)):
        atoms = distribution(alternative)
        hx, hy, theta, bias, gamma = population(atoms, misspecified)
        naive_mean = F(0)
        corrected_mean = F(0)
        count = 0
        for sample in itertools.product(atoms, repeat=n):
            rows, masses = zip(*sample)
            p = math.prod(masses)
            naive, corrected = cluster_scores(rows, hx, hy)
            naive_mean += p * naive
            corrected_mean += p * corrected
            count += 1
        assert corrected_mean == theta + bias
        assert naive_mean == theta + bias + gamma / (n - 1)
        results.append({'alternative': alternative, 'misspecified': misspecified, 'N': n,
                        'enumerated_samples': count, 'theta': str(theta),
                        'nuisance_product_bias': str(bias), 'gamma': str(gamma),
                        'naive_mean': str(naive_mean), 'corrected_mean': str(corrected_mean)})
    atoms = distribution(False)
    hx, hy, theta, _, _ = population(atoms, False)
    duplicate_mean = sum(p * cluster_scores([a] * 4, hx, hy)[1] for a, p in atoms)
    assert theta == 0 and duplicate_mean == F(1, 96)
    return results, {'raw_conditional_independence': True, 'population_target': str(theta),
                     'duplicated_row_corrected_mean': str(duplicate_mean),
                     'iid_rows': False}


def deterministic_references():
    gamma_null = (math.asin(.5) - math.asin(.25)) / (2 * math.pi)
    theta_alt = (math.asin(.4) - math.asin(.25)) / (2 * math.pi)
    gamma_alt = (math.asin(.8) - math.asin(.4)) / (2 * math.pi)
    phi = lambda b: math.exp(-b*b/2) / math.sqrt(2*math.pi)
    q, err = quad(lambda b: phi(b) * (ndtr(b/math.sqrt(3)) - .5)**2,
                  -np.inf, np.inf, epsabs=1e-13, epsrel=1e-13)
    expected_duplicate = math.asin(.25) / (2 * math.pi)
    assert abs(q - expected_duplicate) < 1e-12
    n = 8
    grid = np.linspace(-1, 1, n)
    a = ndtr((grid[:, None] - grid[None, :]) / (.1 * math.sqrt(2)))
    biases = []
    errors = []
    for i in range(n):
        ai = np.delete(a[i], i)
        bi = 1 - ai
        q_i = sum(ai[j] * bi[k] for j in range(n-1) for k in range(n-1) if j != k) / ((n-1)*(n-2))
        observed = q_i - float(ai.mean() * bi.mean())
        expected = float(ai.var() / (n-2))
        errors.append(abs(observed - expected))
        biases.append(expected)
    assert max(errors) < 1e-15 and np.mean(biases) > 0
    effective = 400 * (128 // 3)
    radius = math.sqrt(2 * math.log(20) / effective)
    # A distribution-free lower bound on power follows by applying the lower
    # tail inequality when the known mean exceeds the rejection threshold.
    power_lower = 1 - math.exp(-effective * max(0, theta_alt - radius)**2 / 2)
    return {'gaussian_null_gamma': gamma_null, 'gaussian_alternative_theta': theta_alt,
            'gaussian_alternative_gamma': gamma_alt,
            'gaussian_duplicated_row_bias': q,
            'gaussian_duplicated_row_quadrature_error_estimate': err,
            'gaussian_duplicated_row_formula_difference': abs(q - expected_duplicate),
            'fixed_grid_opposite_bias_by_position': biases,
            'fixed_grid_opposite_average_bias': float(np.mean(biases)),
            'fixed_grid_formula_max_difference': max(errors),
            'finite_radius_N128_M400_alpha05': radius,
            'finite_power_lower_bound_N128_M400': power_lower,
            'N128_M400_power_lower_bound_is_weak': True}


def main():
    exact, boundary = exact_enumeration()
    result = {'status': 'PASS', 'review_scope': 'shared-context independent derivation and verifier; not blind',
              'implementation_or_study_imported': False,
              'exact_fraction_checks': exact, 'exact_dependent_row_counterexample': boundary,
              'deterministic_references': deterministic_references(),
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = Path(__file__).with_name('NOVELTY_THEORY_EXACT_CHECKS_20260905.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
