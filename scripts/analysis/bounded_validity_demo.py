#!/usr/bin/env python3
"""Freeze, run, and independently verify the bounded-certificate example.

The example uses known population histogram means to isolate finite-resolution
bias and checks its analytic allowance. It does not empirically validate a
learned nuisance fit or the sampling assumptions of the rank-panel audit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial import Polynomial

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts/analysis/bounded_audit.py"
TESTS = ROOT / "tests/test_bounded_audit.py"
sys.path.insert(0, str(MODULE.parent))
from bounded_audit import bounded_mean_certificate

DEFAULT_OUTPUT = ROOT / "results/bounded_validity"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(), MODULE, TESTS]}


def analytic_checks():
    """Integrate polynomials and optimize all support corners independently."""
    w = [-1/3, 4/3]
    exact_biases = []
    for q in [1, 2]:
        bias = 0.0
        for k in range(q):
            left, right = k/q, (k+1)/q
            mean = (right**3-left**3)/(3*(right-left))
            residual = Polynomial([-mean, 0, 1])
            antiderivative = (residual*residual).integ()
            bias += antiderivative(right)-antiderivative(left)
        exact_biases.append(bias)
        np.testing.assert_allclose(bias, 1/(9*q*q)-1/(45*q**4), atol=1e-15)
    combined = sum(a*b for a, b in zip(w, exact_biases))
    np.testing.assert_allclose(combined, 1/180, atol=1e-15)
    extrema = {}
    for name, corners in [("noise_free", [(0, 0)]),
                           ("bounded_noise", [(-1, -1), (-1, 1), (1, -1), (1, 1)])]:
        values = []
        for left, right, bin_mean in [(0, .5, 1/12), (.5, 1., 7/12)]:
            for ex, ey in corners:
                poly = (w[0]*Polynomial([ex-1/3, 0, 1])*Polynomial([ey-1/3, 0, 1])
                        +w[1]*Polynomial([ex-bin_mean, 0, 1])*Polynomial([ey-bin_mean, 0, 1]))
                points = [left, right]
                for point in poly.deriv().roots():
                    if abs(point.imag) < 1e-12 and left <= point.real <= right:
                        points.append(point.real)
                values.extend(float(poly(point)) for point in points)
        extrema[name] = [min(values), max(values)]
    np.testing.assert_allclose(extrema["noise_free"], [-1/36, 7/48], atol=1e-14)
    np.testing.assert_allclose(extrema["bounded_noise"], [-37/36, 95/48], atol=1e-14)
    # A tied rank of this size refutes the tempting uniform sqrt(3) bound.
    rank_max = 79/math.sqrt(80)
    # Entity 1 has raw X=Y=B=0. Entity 2 has X=Y=B=H with H=+-1.
    # Raw X and Y are deterministic given their own B, yet entity 1's two
    # standardized ranks share H, which its own raw B does not reveal.
    peer_rank_values = []
    for h in [-1, 1]:
        raw = np.array([0., h])
        ranks = np.argsort(np.argsort(raw)).astype(float)+1
        ranks = (ranks-ranks.mean())/ranks.std(ddof=1)
        peer_rank_values.append(ranks[0])
    rank_covariance = float(np.var(peer_rank_values))
    np.testing.assert_allclose(rank_covariance, .5, atol=1e-15)
    return {"population_bin_biases": exact_biases,
            "extrapolated_bias": combined, "support_extrema": extrema,
            "tied_rank_max_n80": rank_max,
            "raw_CI_rank_counterexample_covariance": rank_covariance,
            "pass": True}


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "finite-sample bounded certificate with an external analytic bias envelope",
        "not_a_general_rank_panel_calibration": True,
        "source_sha256": source_hashes(),
        "numpy_version": np.__version__,
        "n_values": [128, 512, 2048], "replications": 300,
        "seed_base": 202609056711000, "seed_stride_by_n": 100000,
        "draw_order": ["uniform_U", "rademacher_ex1", "rademacher_ex2",
                       "rademacher_ey1", "rademacher_ey2", "rademacher_V"],
        "conditions": ["noise_free_null", "independent_noise_null", "shared_signal_alternative"],
        "condition_dependence": "three conditions share the primitive draws within each n/replication",
        "independent_primitive_draws": 900, "condition_datasets": 2700,
        "g": "U**2", "q": [1, 2], "beta": 2,
        "weights": [-1/3, 4/3],
        "nuisance": "known population bin means (l*l+l*r+r*r)/3; no fitted noise",
        "oracle_thetas": [0, 0, .25], "external_bias_upper": 1/180+1e-12,
        "analytic_bias": "1/180 exactly; plus 1e-12 outward arithmetic allowance for the generated floating score",
        "numerical_bias_padding": 1e-12,
        "envelope_failure_probability": 0., "alpha": .05,
        "bounds_noise_free": [-1/36-1e-12, 7/48+1e-12],
        "bounds_bounded_noise": [-37/36-1e-12, 95/48+1e-12],
        "numerical_support_padding": 1e-12,
        "comparison": "same Hoeffding score interval with bias_upper=0, deliberately unjustified for oracle theta",
        "primary_decision": "strictly positive lower confidence bound",
        "no_pilot_no_optional_stopping": True,
        "supersedes_failed_protocol_sha256": "ab49471bd8e1dc4f0d9701ca915ef1b240dc15c97f866bf9bb26c7deb3ed1d4a",
        "superseded_run": "strict exact support check stopped at n=2048, replication=118, noise-free null; numerical subtraction put one score 3.47e-18 below the exact endpoint; no completed draw file or rate summary; only numerical support padding and independent replacement seeds changed in the experiment",
        "supersedes_numerically_unsafe_protocol_sha256": "c81217ec02fe268cd0c768f77b054841536540756d76d8a3cd0966fe6ec570dd",
        "numerical_certificate_amendment": "Exact dyadic weighted sums and ranges, outward Decimal confidence calculations and exact rank-helper guards prevent floating-point drift from producing a positive lower bound for constant null scores. A 1e-12 score-arithmetic allowance is added to the analytic bias bound. The numerical validation uses disjoint seeds from the earlier bounded-score study.",
    }
    write_json(output/"protocol.json", protocol)
    print(json.dumps({"protocol": str(output/"protocol.json"), "sha256": sha(output/"protocol.json")}))


def load_protocol(output):
    protocol = json.loads((output/"protocol.json").read_text())
    if source_hashes() != protocol["source_sha256"]:
        raise RuntimeError("source hashes changed since protocol freeze")
    if np.__version__ != protocol["numpy_version"]:
        raise RuntimeError("NumPy version differs from frozen protocol")
    return protocol


def generated_conditions(n, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    ex1, ex2, ey1, ey2, v = [2*rng.integers(0, 2, n)-1 for _ in range(5)]
    return u, [(np.zeros(n), np.zeros(n)),
               (.5*(ex1+ex2), .5*(ey1+ey2)),
               (.5*(ex1+v), .5*(ey1+v))]


def production_scores(u, ex, ey):
    x, y = u*u+ex, u*u+ey
    scores = np.zeros(len(u))
    for q, weight in [(1, -1/3), (2, 4/3)]:
        bin_number = np.minimum((q*u).astype(int), q-1)
        left, right = bin_number/q, (bin_number+1)/q
        mean = (left*left+left*right+right*right)/3
        scores += weight*(x-mean)*(y-mean)
    return scores


def run(output):
    protocol = load_protocol(output)
    if (output/"draws.csv").exists() or (output/"receipt.json").exists():
        raise FileExistsError("refusing to overwrite an existing simulation")
    start = time.perf_counter()
    checks = analytic_checks()
    rows = []
    for cell, n in enumerate(protocol["n_values"]):
        for replication in range(protocol["replications"]):
            seed = protocol["seed_base"]+cell*protocol["seed_stride_by_n"]+replication
            u, conditions = generated_conditions(n, seed)
            for condition, (ex, ey) in enumerate(conditions):
                scores = production_scores(u, ex, ey)
                bounds = protocol["bounds_noise_free"] if condition == 0 else protocol["bounds_bounded_noise"]
                result = bounded_mean_certificate(scores, *bounds, bias_upper=protocol["external_bias_upper"],
                                                   alpha=protocol["alpha"])
                unadjusted = bounded_mean_certificate(scores, *bounds, bias_upper=0, alpha=protocol["alpha"])
                rows.append({"n": n, "replication": replication, "seed": seed,
                             "condition": protocol["conditions"][condition],
                             "theta": protocol["oracle_thetas"][condition],
                             **result.to_dict(),
                             "unadjusted_lower_bound": unadjusted.lower_bound,
                             "unadjusted_p_one_sided": unadjusted.p_one_sided,
                             "unadjusted_reject": unadjusted.reject})
    with (output/"draws.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    load_protocol(output)
    write_json(output/"receipt.json", {
        "protocol_sha256": sha(output/"protocol.json"), "draws_sha256": sha(output/"draws.csv"),
        "datasets": len(rows), "independent_primitive_draws": 900,
        "elapsed_seconds": time.perf_counter()-start,
        "analytic_checks": checks, "source_sha256": source_hashes(),
    })
    print(json.dumps({"datasets": len(rows), "elapsed_seconds": time.perf_counter()-start}))


def wilson(k, n):
    z = 1.959963984540054
    denominator = 1+z*z/n
    center = (k/n+z*z/(2*n))/denominator
    half = z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/denominator
    return [center-half, center+half]


def verify(output):
    # Reconstruct recorded rows independently; original source identities remain in the protocol.
    protocol = json.loads((output/"protocol.json").read_text())
    receipt = json.loads((output/"receipt.json").read_text())
    assert receipt["draws_sha256"] == sha(output/"draws.csv")
    assert receipt["protocol_sha256"] == sha(output/"protocol.json")
    rows = list(csv.DictReader((output/"draws.csv").open()))
    assert len(rows) == 2700
    indexed = {(int(row["n"]), int(row["replication"]), row["condition"]): row for row in rows}
    assert len(indexed) == len(rows)
    summary = []
    largest_reconstruction_error = 0.
    for cell, n in enumerate(protocol["n_values"]):
        for condition, name in enumerate(protocol["conditions"]):
            selected = [indexed[n, rep, name] for rep in range(protocol["replications"])]
            adjusted_count = unadjusted_count = coverage_failures = 0
            for replication, row in enumerate(selected):
                seed = protocol["seed_base"]+cell*protocol["seed_stride_by_n"]+replication
                assert int(row["seed"]) == seed
                u, triples = generated_conditions(n, seed)
                ex, ey = triples[condition]
                # Independent closed-form score, without histogram estimation
                # or the bounded_audit module's concentration calculations.
                a = u*u-np.where(u < .5, 0., 2/3)
                independent_score = (ex+a)*(ey+a)-1/36
                mean = float(sum(independent_score)/n)
                bounds = protocol["bounds_noise_free"] if condition == 0 else protocol["bounds_bounded_noise"]
                radius = (bounds[1]-bounds[0])*math.sqrt(math.log(20)/(2*n))
                lower = mean-protocol["external_bias_upper"]-radius
                gap = max(mean-protocol["external_bias_upper"], 0)
                pvalue = math.exp(-2*n*gap*gap/(bounds[1]-bounds[0])**2)
                np.testing.assert_allclose([float(row["weighted_mean"]), float(row["concentration_radius"]),
                                            float(row["lower_bound"]), float(row["p_one_sided"])],
                                           [mean, radius, lower, pvalue], atol=3e-13, rtol=1e-11)
                largest_reconstruction_error = max(largest_reconstruction_error,
                                                   abs(mean-float(row["weighted_mean"])))
                assert (row["reject"] == "True") == (lower > 0)
                assert (row["unadjusted_reject"] == "True") == (mean-radius > 0)
                adjusted_count += lower > 0
                unadjusted_count += mean-radius > 0
                coverage_failures += lower > protocol["oracle_thetas"][condition]
            count = len(selected)
            summary.append({"n": n, "condition": name, "replications": count,
                            "bounded_rejections": adjusted_count,
                            "bounded_rate": adjusted_count/count,
                            "bounded_wilson95": wilson(adjusted_count, count),
                            "unadjusted_rejections": unadjusted_count,
                            "unadjusted_rate": unadjusted_count/count,
                            "lower_bound_coverage_failures": coverage_failures})
    result = {"pass": True, "all_draws_independently_reconstructed": len(rows),
              "max_mean_absolute_reconstruction_error": largest_reconstruction_error,
              "analytic_checks": analytic_checks(), "summary": summary,
              "protocol_sha256": sha(output/"protocol.json"),
              "draws_sha256": sha(output/"draws.csv")}
    write_json(output/"verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "run", "verify"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[arguments.phase](arguments.output_dir)
