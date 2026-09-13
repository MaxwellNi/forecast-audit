"""Post-run deterministic population reference for the fixed-grid null.

This analysis explains, and does not select or change, the frozen design.
Its formula is exact in the stated probability model; decimal outputs use
ordinary double-precision normal CDF and quadrature evaluations.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def population_reference():
    n, sigma = 80, .1
    grid = np.linspace(-1, 1, n)
    scale = np.sqrt(n*(n+1)/12)
    conditional_mean = (norm.cdf((grid[:, None]-grid[None, :])/(sigma*np.sqrt(2))).sum(axis=1)-n/2)/scale
    quadrature_checks = []
    for focal in [0, 39, 79]:
        difference = (grid[focal]-np.delete(grid, focal))/sigma
        integral, integration_error = quad(
            lambda noise: float(norm.pdf(noise)*(1+norm.cdf(difference+noise).sum())),
            -12., 12., epsabs=1e-10, epsrel=1e-11)
        reference = (integral-(n+1)/2)/scale
        error = abs(reference-conditional_mean[focal])
        assert error < 1e-12
        quadrature_checks.append({"focal_grid_index": focal, "direct_noise_integral_mean": reference,
                                  "closed_formula_mean": float(conditional_mean[focal]),
                                  "absolute_difference": error,
                                  "quadrature_reported_error_before_standardization": integration_error})
    ladders = {"coarse": [4, 6, 8, 12, 16], "primary": [8, 12, 16, 24, 32],
               "fine": [16, 24, 32, 48, 64]}
    all_qs = sorted({q for ladder in ladders.values() for q in ladder}|{80, 160})
    biases, counts = {}, {}
    for q in all_qs:
        bins = np.minimum((q*(np.arange(n)+.5)/n).astype(int), q-1)
        fitted = np.array([conditional_mean[bins == label].mean() for label in bins])
        biases[q] = float(np.mean((conditional_mean-fitted)**2))
        counts[q] = np.unique(np.bincount(bins), return_counts=True)
    assert biases[80] == biases[160] == 0.
    rows = []
    for name, qs in ladders.items():
        for beta in ([.5, 1., 2.] if name == "primary" else [1.]):
            design = np.column_stack([np.ones(len(qs)), np.asarray(qs, float)**(-beta)])
            weights = np.linalg.pinv(design)[0]
            np.testing.assert_allclose(weights@design, [1., 0.], atol=1e-12)
            rows.append({"ladder": name, "beta": beta, "weights": weights.tolist(),
                         "population_intercept_bias": float(weights@np.array([biases[q] for q in qs]))})
    return {"status": "PASS", "scope": "Post-run deterministic reference; no simulation parameters, seeds, estimates or decisions changed.",
            "rank_mean_formula": "m_k=(sum_j Phi((c_k-c_j)/(sigma*sqrt(2)))-n/2)/sqrt(n*(n+1)/12)",
            "rank_mean_derivation": "Rank_i=1+sum_{j!=i}1{X_i>X_j}; each difference has Gaussian noise variance2sigma^2. The self-term Phi(0)=1/2 yields the displayed formula. Untied rank sample SD is sqrt(n(n+1)/12).",
            "bin_bias_formula": "mean_k (m_k - mean_{l:bin(l)=bin(k)}m_l)^2",
            "bin_bias_derivation": "The two ranks are conditionally independent given focal baseline rank, and have the same conditional mean m_k. Population entity effects vanish by random permutation. The within-bin conditional product mean therefore equals variance(m_k | bin); each focal grid index has probability1/n.",
            "effect_of_extrapolation": "Signed OLS intercept weights can make these positive per-q biases sum to a negative value, even with population bin nuisances. More independent evaluation clusters do not remove a fixed nonzero population approximation bias.",
            "probability_model": {"entities": n, "sigma": sigma, "grid": "linspace(-1,1,80)"},
            "per_q_population_bin_bias": biases, "weighted_population_references": rows,
            "independent_quadrature_checks": quadrature_checks,
            "isolating_each_grid_rank_gives_zero_bin_bias": True,
            "numerical_scope": "Analytic model identities evaluated in double precision. Quadrature uses [-12,12]; the omitted Gaussian rank-mean tail is bounded by 2*n*norm.sf(12), far below1e-28.",
            "not_identical_to_finite_sample_fit": "The production run estimates nuisances in finite training folds. These values describe population bin fits and must not be substituted for a certified finite-sample nuisance-bias bound."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = population_reference()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "population_intercepts": result["weighted_population_references"]}))
