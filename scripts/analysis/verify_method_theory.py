"""Independent arithmetic checks for the audit method and its bias examples.

Reads frozen scientific outcomes; produces no new empirical hypothesis search.
Gaussian checks concern population quantile bins, not fitted panel rank bins.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = ROOT / "scripts/analysis"
sys.path.insert(0, str(SCRIPTS))
import audit_panel_predictions as core


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hac_checks():
    # Unequal observed clusters and missing calendar dates. Calendar labels
    # represent complete daily bins. All products here are synthetic fixtures.
    labels = np.array(["2026-01-01"] * 3 + ["2026-01-03"] * 2 +
                      ["2026-01-04"] * 5 + ["2026-01-07"])
    values = np.array([1., 3., -2., 5., -1., 4., 2., 9., -7., 6., -3.])
    n = len(values)
    mean = float(values.mean())
    counts = np.array([3, 0, 2, 5, 0, 0, 1])
    totals = np.array([2., 0., 4., 14., 0., 0., -3.])
    manual = totals - counts * mean
    implementation, gaps = core.calendar_scores(values - mean, labels, "D")
    np.testing.assert_allclose(implementation[:, 0], manual, atol=2e-14)
    assert gaps == 3
    rows = []
    for lag in [0, 1, 2, 6, 10]:
        # Separate all-pair expression for the Bartlett kernel.
        kernel = np.maximum(0., 1. - np.abs(np.subtract.outer(np.arange(7), np.arange(7))) / (lag + 1))
        direct = float(manual @ kernel @ manual) / n**2
        implemented = float(core.bartlett_score_covariance(implementation, lag)[0, 0]) / n**2
        # Independent moving-window identity, zero-padded beyond the calendar.
        windows = np.convolve(manual, np.ones(lag + 1), mode="full")
        moving = float(windows @ windows) / ((lag + 1) * n**2)
        np.testing.assert_allclose([implemented, moving], [direct, direct], rtol=1e-13, atol=1e-14)
        rows.append({"lag": lag, "variance": direct, "implementation_difference": implemented - direct,
                     "moving_window_difference": moving - direct})
    zero = np.sum(manual**2) / n**2
    assert abs(zero - rows[0]["variance"]) < 1e-14
    return {"n": n, "observed_clusters": 4, "calendar_bins": 7,
            "calendar_gaps": gaps, "counts": counts.tolist(),
            "centered_cluster_scores": manual.tolist(), "checks": rows,
            "normalization": "n^-2, no finite-cluster correction or calendar-length divisor"}


def grid_and_sine():
    grid_path = ROOT / "results/current_core_sensitivity/draws.csv"
    smooth_path = ROOT / "results/smooth_bin_validation/draws.csv"
    df = pd.read_csv(grid_path)
    keys = ["study", "condition", "clusters", "ladder", "beta"]
    groups = df.groupby(keys).size().rename("rows").reset_index()
    assert len(df) == 9900 and len(groups) == 33 and (groups.rows == 300).all()
    main = groups[groups.study == "cluster_count_exponent"]
    extra = groups[groups.study == "ladder"]
    assert len(main) == 27 and set(main.ladder) == {"primary"}
    assert set(main.clusters) == {25, 50, 100} and set(main.beta) == {.5, 1., 2.}
    assert len(extra) == 6 and set(extra.clusters) == {100}
    assert set(extra.ladder) == {"coarse", "fine"} and set(extra.beta) == {1.}
    assert ((norm.sf(df.statistic) <= .05) == df.reject).all()
    smooth = pd.read_csv(smooth_path)
    assert len(smooth) == 21600
    assert ((norm.sf(smooth.statistic) <= .05) == smooth.reject).all()
    sine = smooth[(smooth.design == "smooth_nonlinear") & (smooth.noise_correlation == 0)
                  & (smooth.method == "extrapolation_beta2")]
    rows = []
    for (n, q), part in sine.groupby(["n", "q_min"]):
        count = int(part.reject.sum())
        lo, hi = proportion_confint(count, len(part), alpha=.05, method="wilson")
        ladder = q * np.array([1., 2., 4.])
        w = np.linalg.pinv(np.column_stack([np.ones(3), ladder**-2]))[0]
        exact_bias = float(w @ (18 * (1 - np.sinc(1 / ladder)**2)))
        rows.append({"n": int(n), "q_min": int(q), "replications": len(part), "rejects": count,
                     "rate": count / len(part), "wilson95": [lo, hi],
                     "mean_T": float(part.statistic.mean()), "oracle_bias": exact_bias,
                     "sqrt_n_times_oracle_bias": float(np.sqrt(n) * exact_bias)})
    assert [row["rejects"] for row in rows] == [47, 26, 36]
    return {"source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in [grid_path, smooth_path]},
            "grid_rows": len(df), "primary_cells": len(main), "additional_cells": len(extra),
            "all_cells": groups.to_dict("records"), "smooth_rows": len(smooth), "sine": rows}


def gaussian_quantile_checks():
    mp.mp.dps = 80
    sqrt2 = mp.sqrt(2)
    def phi(x):
        return mp.exp(-x*x/2) / mp.sqrt(2*mp.pi) if mp.isfinite(x) else mp.mpf(0)
    def quantile(p):
        return sqrt2 * mp.erfinv(2*p-1)
    rows = []
    quadrature_checks = []
    # Every finite value is a population integral with unit slope a=1.
    for q in [4, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024]:
        edges = [quantile(mp.mpf(j)/q) for j in range(q + 1)]
        means = [q * (phi(a)-phi(b)) for a, b in zip(edges[:-1], edges[1:])]
        exact = 1 - mp.fsum(mu*mu for mu in means)/q
        t = edges[-2]
        lam = q*phi(t)
        tail_variance = 1+t*lam-lam*lam
        bound = 2*tail_variance/q
        interior_upper = mp.quad(lambda z: 1/phi(z), [-t, 0, t]) / (4*q*q)
        upper = bound + interior_upper
        assert upper >= exact >= bound > 0
        rows.append({"q": q, "exact_unit_slope_bias_65_digits": mp.nstr(exact, 65),
                     "tail_lower_bound_65_digits": mp.nstr(bound, 65),
                     "total_upper_bound_65_digits": mp.nstr(upper, 65),
                     "q_squared_bias": float(q*q*exact),
                     "q_logq_bias": float(q*mp.log(q)*exact),
                     "q_logq_tail_lower_bound": float(q*mp.log(q)*bound),
                     "tail_variance_t_squared": float(t*t*tail_variance)})
        if q in [4, 8, 16, 32]:
            # Quadrature of (z-bin mean)^2 phi(z) uses a different formula.
            integral = sum(quad(lambda z, mean=float(mu): (z-mean)**2 * norm.pdf(z),
                                float(a), float(b), epsabs=1e-12, epsrel=1e-12)[0]
                           for a, b, mu in zip(edges[:-1], edges[1:], means))
            difference = abs(integral-float(exact))
            assert difference < 2e-13
            quadrature_checks.append({"q": q, "quadrature": integral, "difference": difference})
    # Tail asymptotics can be checked at huge q without summing q bins.
    tails = []
    for power in [2, 4, 8, 16, 32]:
        q = mp.mpf(10)**power
        t = quantile(1-1/q)
        lam = q*phi(t)
        variance = 1+t*lam-lam*lam
        tails.append({"q_power_of_10": power, "t_squared_variance": float(t*t*variance),
                      "q_logq_times_tail_bound": float(2*mp.log(q)*variance)})
    return {"precision_decimal_digits": 80, "scope": "Oracle equal-probability Gaussian bins and f(z)=g(z)=z; not fitted bins or deployed ranks",
            "exact_finite_formula": "D_q=1-q*sum_j(phi(z[j-1])-phi(z[j]))^2",
            "proved_asymptotic": "D_q=Theta(1/(q log q)); 1<=liminf q log(q) D_q<=limsup q log(q) D_q<=5/4",
            "implication": "Neither D_q=c/q^2+O(q^-4) nor D_q=c/q+O(q^-2) is possible for any fixed finite c",
            "not_claimed": "No complete asymptotic equality or exact leading constant is asserted",
            "finite_bins": rows, "independent_quadrature": quadrature_checks, "tail_asymptotic_checks": tails}


def population_rank_checks():
    # Phi(X), Phi(Y) for jointly generated standard Gaussian variables.
    # These are population CDF transforms, without peer dependence or ties.
    rows, checks, derivatives = [], [], []
    for rho in [.5, .8, .9]:
        a = rho / np.sqrt(2-rho*rho)
        population_second_moment = .25 + np.arcsin(rho*rho/2)/(2*np.pi)
        for q in [8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
            edges = norm.ppf(np.arange(q+1, dtype=float)/q)
            integrals = np.array([quad(lambda z: norm.cdf(a*z)*norm.pdf(z), left, right,
                                       epsabs=3e-13, epsrel=3e-13)[0]
                                  for left, right in zip(edges[:-1], edges[1:])])
            assert abs(integrals.sum()-.5) < 3e-13
            bias = float(population_second_moment-q*(integrals @ integrals))
            assert bias > 0
            rows.append({"rho_X": rho, "rho_Y": rho, "a_squared": a*a, "q": q,
                         "population_rank_bin_bias": bias, "q_squared_bias": q*q*bias})
            if q in [8, 32]:
                # Direct centered integrals avoid subtracting two second moments.
                direct = sum(quad(lambda z, mu=q*integral: (norm.cdf(a*z)-mu)**2 * norm.pdf(z),
                                  left, right, epsabs=1e-13, epsrel=1e-12)[0]
                             for left, right, integral in zip(edges[:-1], edges[1:], integrals))
                assert abs(direct-bias) < 4e-13
                checks.append({"rho": rho, "q": q, "direct_bias": direct,
                               "second_moment_difference": abs(direct-bias)})
        for zmax in [1., 2., 4., 8.]:
            integral = quad(lambda z: a*a / np.sqrt(2*np.pi) * np.exp(-(2*a*a-1)*z*z/2),
                            -zmax, zmax, epsabs=1e-12, epsrel=1e-12)[0]
            derivatives.append({"rho": rho, "z_truncation": zmax,
                                "truncated_derivative_integral": integral,
                                "lower_limit_q_squared_bias": integral/12})
        if 2*a*a > 1:
            numerical = quad(lambda z: a*a / np.sqrt(2*np.pi) * np.exp(-(2*a*a-1)*z*z/2),
                             -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
            analytic = a*a/np.sqrt(2*a*a-1)
            assert abs(numerical-analytic) < 1e-12
            checks.append({"rho": rho, "full_derivative_integral": analytic,
                           "numeric_difference": abs(numerical-analytic)})
    return {"scope": "Independent Gaussian scalar observations; known population CDF ranks and known equal-probability bins; neither finite panel ranks nor fitted nuisance means",
            "means": "f(u)=Phi(a Phi^-1(u)), g(u)=Phi(b Phi^-1(u)), a=rho_X/sqrt(2-rho_X^2), b similarly",
            "assumptions": "0<rho_X,rho_Y<1; independent standard Gaussian Z,epsilon_X,epsilon_Y; raw X,Y standard normal and conditionally independent given Z",
            "derivative_integral": "integral_0^1 f'g'=ab/sqrt(a^2+b^2-1) if a^2+b^2>1; infinity otherwise",
            "proved_consequence": "When a^2+b^2<=1, q^2 B_q -> infinity by nonnegative bin covariances and the smooth interior expansion; beta=2 leading expansion with finite coefficient is impossible",
            "not_claimed": "Integrability alone does not prove the O(q^-4) remainder; no exact leading exponent is claimed in the divergent regime",
            "numerical_bins": rows, "independent_checks": checks, "truncated_derivative_integrals": derivatives}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; use a new receipt path")
    report = {"status": "PASS_WITH_SCOPE_LIMITS", "date": "2026-09-07",
              "scope": "Independent formula checks and oracle counterexample; these checks do not establish deployment validity",
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), Path(core.__file__)]},
              "hac": hac_checks(), "original_row_recount": grid_and_sine(),
              "gaussian_counterexample": gaussian_quantile_checks(),
              "population_rank_counterexample": population_rank_checks()}
    path = args.output
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(path), "hac_checks": 5,
                      "grid_cells": 33, "smooth_rows": 21600, "gaussian_bins": 11, "population_rank_bins": 27}))
