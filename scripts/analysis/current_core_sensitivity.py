"""Prospectively frozen sensitivity of the current cluster-held-out panel audit.

Run freeze, benchmark, run, verify in that order in a new output directory.
The simulation describes the specified rank target, not universal CI validity.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

import audit_panel_predictions as core


ENTITIES = 80
CLUSTERS = (25, 50, 100)
REPLICATIONS = 300
SEED_START = 202609058430000
BETAS = (0.5, 1.0, 2.0)
LADDERS = {"coarse": (4, 6, 8, 12, 16), "primary": (8, 12, 16, 24, 32),
           "fine": (16, 24, 32, 48, 64)}
CONDITIONS = ("independent_forecast_null", "high_overlap_conditional_null",
              "high_overlap_shared_noise_alternative")
WORKERS = 8
SCRIPT_PATH = Path(__file__).resolve()
CORE_PATH = Path(core.__file__).resolve()
TEST_PATH = SCRIPT_PATH.parents[2]/"tests/test_current_core_sensitivity.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def specification():
    return {
        "entities": ENTITIES, "cluster_counts": list(CLUSTERS),
        "replications_per_cell": REPLICATIONS, "independent_primitive_draws": REPLICATIONS,
        "condition_datasets": 2700, "audit_rows": 9900, "summary_cells": 33,
        "conditions": list(CONDITIONS), "seed_start": SEED_START,
        "pairing": "For replication r use default_rng(seed_start+r). Generate a 100 by 80 primitive panel once; M=25 and M=50 use its first M clusters. All conditions and specifications share primitives. The 2700 condition datasets are dependent within each of 300 independent replication draws.",
        "primitive_draw_order": "100x80 independent normal baseline, 100x80 normal epsilon_X, 100x80 normal epsilon_Y; 100 independent rng.permutation(80) calls; 100 uniform(-.25,.25) location shifts.",
        "independent_forecast_null": "B iid N(0,1); X=epsilon_X; Y=a_i+.6*B+.3*sin(B)+epsilon_Y; a_i=linspace(-.6,.6,80). The whole forecast-rank vector is independent of baseline/outcome arrays; its conditional mean given entity and baseline rank is zero.",
        "high_overlap_baseline": "B_ti=c_pi_t(i)+T_t, c=linspace(-1,1,80), independent uniform permutations pi_t and continuous uniform(-.25,.25) shifts T_t. No entity effects or heterogeneous noise scales in these two regimes.",
        "high_overlap_conditional_null": "X=B+.1*epsilon_X; Y=B+.1*epsilon_Y, with independent homogeneous iid standard-normal errors.",
        "high_overlap_rank_null_proof": "Conditional on a focal baseline rank k and any competitor-grid permutation, forecast and outcome ranks depend on disjoint noise arrays and are independent. Their respective conditional rank distributions depend only on the fixed competitor-grid multiset, not its assignment. Mixing over permutations therefore preserves the product law. The common shift cancels from both ranks. Untied standardized ranks have deterministic centering/scale. Thus focal transformed conditional covariance given entity and exact baseline percentile is zero. Coarse binning may leave bias; fitted nuisance error is not assumed zero.",
        "high_overlap_shared_noise_alternative": "X=B+.1*epsilon_X; Y=B+.1*(.6*epsilon_X+.8*epsilon_Y). Conditional raw residual covariance is .006; this is a prespecified positive control, not a claimed analytic effect size on the rank scale.",
        "cluster_count_exponent_study": {"M": list(CLUSTERS), "ladder": "primary", "betas": list(BETAS)},
        "ladder_study": {"M": 100, "additional_ladders": ["coarse", "fine"], "beta": 1.0,
                         "primary_comparator": "M=100,beta=1 row of cluster_count_exponent study"},
        "ladders": {key: list(value) for key, value in LADDERS.items()},
        "nuisance_fitting": "Call unchanged current core.panel_residuals once per distinct q, with all five complete contiguous cluster folds; entity plus rank-bin additive nuisance, trained only on the other folds. Reuse those exact residual products for prespecified weights.",
        "inference": "Unchanged current audit_extrapolated_panel formula: OLS-intercept weights on q^-beta; mean residual product; observation-centered cluster sums; lag zero; normal.sf(mean/SE), including fixed 1e-10 numerical score floor. No finite-M correction or p-value recalibration.",
        "decision": "One-sided p<=.05; one fixed model in each condition. No overlap filter, multiplicity selection, optional stopping, or parameter/seed changes after benchmark.",
        "diagnostics": "Report all cells with pointwise 95% Wilson intervals, empirical SD of mean product, average reported SE and their ratio, standardized statistic mean/SD and .025/.05/.5/.95/.975 quantiles, Monte Carlo Student-t interval for the mean product across independent replications, undefined counts, and per-q forecast nuisance R2. Low positive rejection alone is not interpreted as calibration if null statistics are shifted or dispersed. No cell or replication is excluded for overlap or solver results.",
        "verification": "Before production, generated unit fixtures use seeds outside production range. During production replication zero, compare all 33 specification results to full unchanged core.audit_extrapolated_panel calls. Verify output completeness, every normal tail, decisions, and independent statsmodels Wilson intervals after run.",
        "benchmark": "After freeze, time one complete paired replication; report only runtime and row/check counts. Production reruns it and counts it once. No statistical output is used to change the design.",
        "execution": {"processes": WORKERS, "threads_per_process": 1, "start_method": "spawn"},
        "stopping": "Exactly 300 paired replications. A failure is reported; any correction requires a separately documented freeze instead of overwriting this protocol.",
        "limitations": "Fixed independent clusters, balanced size80, continuous Gaussian-noise ranks and a deliberately homogeneous shifted-grid high-overlap construction. This cannot establish calibration for arbitrary baselines, hidden heterogeneous cluster states, serial real panels, or all conditional-independence nulls. Wilson intervals quantify Monte Carlo uncertainty only; sensitivities share draws and are not independent experiments.",
    }


def check_protocol(output):
    protocol = json.loads((output/"protocol.json").read_text())
    if protocol["specification"] != specification():
        raise RuntimeError("Frozen design differs from current specification")
    for key, path in [("script_sha256", SCRIPT_PATH), ("core_sha256", CORE_PATH), ("test_sha256", TEST_PATH)]:
        if protocol[key] != sha(path):
            raise RuntimeError(f"Frozen source changed: {path.name}")
    return protocol


def primitive_draw(seed, max_clusters=100, entities=ENTITIES):
    rng = np.random.default_rng(seed)
    baseline, noise_x, noise_y = (rng.standard_normal((max_clusters, entities)) for _ in range(3))
    grid = np.linspace(-1, 1, entities)
    ordered = np.stack([grid[rng.permutation(entities)] for _ in range(max_clusters)])
    shift = rng.uniform(-.25, .25, max_clusters)
    return baseline, noise_x, noise_y, ordered+shift[:, None]


def frames_from_primitives(primitives, clusters):
    baseline, noise_x, noise_y, high_baseline = [x[:clusters] for x in primitives]
    entities = baseline.shape[1]
    common = {"entity": np.tile(np.arange(entities), clusters),
              "period": np.repeat(np.arange(clusters), entities)}
    effects = np.linspace(-.6, .6, entities)[None, :]
    definitions = [(baseline, noise_x, effects+.6*baseline+.3*np.sin(baseline)+noise_y),
                   (high_baseline, high_baseline+.1*noise_x, high_baseline+.1*noise_y),
                   (high_baseline, high_baseline+.1*noise_x,
                    high_baseline+.1*(.6*noise_x+.8*noise_y))]
    return [pd.DataFrame({**common, "baseline": b.ravel(), "prediction": x.ravel(), "y": y.ravel()})
            for b, x, y in definitions]


def cached_products(frame, qs):
    cache = {}
    for q in qs:
        fit = core.panel_residuals(frame, "contiguous", q=q)
        residual_x, residual_y = fit["forecast_residual"], fit["outcome_residual"]
        numerical_zero = any(np.max(np.abs(fit[f"{side}_residual"])) <=
                             1e-10*(1+np.max(np.abs(fit[f"{side}_rank"])))
                             for side in ["forecast", "outcome"])
        scale = float(fit["forecast_rank"].var())
        cache[q] = {"product": residual_x*residual_y, "numerical_zero": numerical_zero,
                    "nuisance_r2": 1-float(np.mean(residual_x**2))/scale if scale > 0 else None}
    return cache


def from_cached_products(frame, cache, qs, beta):
    products = np.column_stack([cache[q]["product"] for q in qs])
    design = np.column_stack([np.ones(len(qs)), np.asarray(qs, float)**(-beta)])
    weights = np.linalg.pinv(design)[0]
    np.testing.assert_allclose(weights@design, [1., 0.], atol=1e-12)
    combined = products@weights
    mean = float(combined.mean())
    scores = pd.DataFrame({"score": combined-mean, "period": frame.period}).groupby("period").score.sum().to_numpy()[:, None]
    se = float(np.sqrt(float((scores.T@scores)[0, 0])/len(frame)**2))
    if not np.isfinite(se):
        raise ValueError("Nonfinite score standard error")
    defined = not all(cache[q]["numerical_zero"] for q in qs) and se > max(
        core.NUMERICAL_SCORE_FLOOR, 8*np.finfo(float).eps*np.max(np.abs(combined)))
    statistic = mean/se if defined else np.nan
    return {"mean_product": mean, "standard_error": se, "statistic": statistic,
            "p_one_sided": float(norm.sf(statistic)) if defined else 1.,
            "inference_status": "computed" if defined else "undefined_rank_or_scale",
            "weight_l1": float(np.abs(weights).sum()),
            **{f"nuisance_r2_q{q}": cache[q]["nuisance_r2"] for q in qs},
            **{f"product_mean_q{q}": float(cache[q]["product"].mean()) for q in qs}}


def configurations(clusters):
    values = [("cluster_count_exponent", "primary", beta) for beta in BETAS]
    if clusters == 100:
        values += [("ladder", name, 1.) for name in ["coarse", "fine"]]
    return values


def run_replication(replication):
    primitives = primitive_draw(SEED_START+replication)
    rows, checks = [], []
    for clusters in CLUSTERS:
        settings = configurations(clusters)
        qs = sorted({q for _, name, _ in settings for q in LADDERS[name]})
        for condition, frame in zip(CONDITIONS, frames_from_primitives(primitives, clusters)):
            cache = cached_products(frame, qs)
            for study, ladder, beta in settings:
                actual = from_cached_products(frame, cache, LADDERS[ladder], beta)
                rows.append({"replication": replication, "seed": SEED_START+replication,
                             "study": study, "condition": condition, "clusters": clusters,
                             "entities": ENTITIES, "n": len(frame), "ladder": ladder,
                             "beta": beta, "reject": actual["p_one_sided"] <= .05, **actual})
                if replication == 0:
                    full = core.audit_extrapolated_panel(frame, lag=0, frequency=None,
                                                        qs=LADDERS[ladder], betas=(beta,))[0]
                    differences = {}
                    for key in ["mean_product", "standard_error", "statistic", "p_one_sided"]:
                        np.testing.assert_allclose(actual[key], full[key], rtol=0, atol=0, equal_nan=True)
                        differences[key] = abs(actual[key]-full[key]) if np.isfinite(actual[key]) else None
                    assert actual["inference_status"] == full["inference_status"]
                    checks.append({"clusters": clusters, "condition": condition, "ladder": ladder,
                                   "beta": beta, "status": "PASS", "absolute_differences": differences})
    return rows, checks


def wilson(count, total):
    z = float(norm.ppf(.975))
    p = count/total
    d = 1+z*z/total
    centre = (p+z*z/(2*total))/d
    radius = z*np.sqrt(p*(1-p)/total+z*z/(4*total**2))/d
    return max(0., centre-radius), min(1., centre+radius)


def summarize(frame):
    rows = []
    keys = ["study", "clusters", "condition", "ladder", "beta"]
    for key, group in frame.groupby(keys, sort=True):
        rejected = int(group.reject.sum())
        lo, hi = wilson(rejected, len(group))
        row = {**dict(zip(keys, key)), "replications": len(group), "rejections": rejected,
               "rejection_rate": rejected/len(group), "wilson_95_lower": lo, "wilson_95_upper": hi,
               "mean_product_mean": float(group.mean_product.mean()),
               "mean_product_empirical_sd": float(group.mean_product.std(ddof=1)),
               "standard_error_mean": float(group.standard_error.mean()),
               "statistic_mean": float(group.statistic.mean()), "statistic_sd": float(group.statistic.std(ddof=1)),
               "undefined_results": int((group.inference_status != "computed").sum())}
        empirical_sd = row["mean_product_empirical_sd"]
        mc_radius = float(student_t.ppf(.975, len(group)-1))*empirical_sd/np.sqrt(len(group))
        row["mean_product_mc95_lower"] = row["mean_product_mean"]-mc_radius
        row["mean_product_mc95_upper"] = row["mean_product_mean"]+mc_radius
        row["reported_se_over_empirical_sd"] = row["standard_error_mean"]/empirical_sd if empirical_sd > 0 else np.nan
        for quantile in [.025, .05, .5, .95, .975]:
            row[f"statistic_quantile_{quantile:g}"] = float(group.statistic.quantile(quantile))
        for q in LADDERS[key[3]]:
            row[f"nuisance_r2_q{q}_mean"] = float(group[f"nuisance_r2_q{q}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def verify(output):
    from statsmodels.stats.proportion import proportion_confint
    check_protocol(output)
    receipt = json.loads((output/"receipt.json").read_text())
    for name, expected in receipt["output_sha256"].items():
        assert sha(output/name) == expected, name
    frame = pd.read_csv(output/"draws.csv")
    assert len(frame) == 9900
    assert not frame.duplicated(["replication", "study", "clusters", "condition", "ladder", "beta"]).any()
    defined = frame.inference_status == "computed"
    np.testing.assert_allclose(frame.loc[defined, "p_one_sided"], norm.sf(frame.loc[defined, "statistic"]), atol=1e-14, rtol=1e-10)
    np.testing.assert_array_equal(frame.loc[~defined, "p_one_sided"], 1.)
    np.testing.assert_array_equal(frame.reject, frame.p_one_sided <= .05)
    expected = summarize(frame)
    recorded = pd.read_csv(output/"summary.csv")
    pd.testing.assert_frame_equal(expected, recorded, rtol=1e-10, atol=1e-13, check_dtype=False)
    for key, group in frame.groupby(["study", "clusters", "condition", "ladder", "beta"]):
        assert set(group.replication) == set(range(REPLICATIONS))
        np.testing.assert_allclose(wilson(int(group.reject.sum()), len(group)),
                                   proportion_confint(group.reject.sum(), len(group), method="wilson"), atol=1e-14)
    checks = json.loads((output/"full_core_checks.json").read_text())
    assert len(checks) == 33 and all(check["status"] == "PASS" for check in checks)
    return {"status": "PASS", "draw_rows": len(frame), "summary_cells": len(expected),
            "independent_primitive_draws": REPLICATIONS, "full_core_exact_checks": len(checks),
            "protocol_sha256": sha(output/"protocol.json"), "draws_sha256": sha(output/"draws.csv")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "benchmark", "run", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if args.phase == "freeze":
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"protocol.json", {"recorded_utc": now(),
            "status": "Frozen locally before benchmark or production; not an external preregistration",
            "specification": specification(), "script_sha256": sha(SCRIPT_PATH),
            "core_sha256": sha(CORE_PATH), "test_sha256": sha(TEST_PATH)})
        print(json.dumps({"status": "FROZEN", "protocol_sha256": sha(output/"protocol.json")}))
        return
    check_protocol(output)
    if args.phase == "verify":
        result = verify(output)
        write_json(output/"verification.json", result)
        print(json.dumps(result))
        return
    if args.phase == "benchmark":
        if (output/"benchmark.json").exists() or (output/"run_started.json").exists():
            raise FileExistsError("Benchmark/production already started")
        started = time.perf_counter()
        rows, checks = run_replication(0)
        result = {"recorded_utc": now(), "elapsed_seconds": time.perf_counter()-started,
                  "audit_rows": len(rows), "exact_full_core_checks": len(checks),
                  "note": "No statistical output used for design changes; replication zero reruns in production and counts once."}
        write_json(output/"benchmark.json", result)
        print(json.dumps(result))
        return
    if not (output/"benchmark.json").exists():
        raise RuntimeError("Benchmark must follow freeze and precede production")
    write_json(output/"run_started.json", {"recorded_utc": now(), "protocol_sha256": sha(output/"protocol.json")})
    started = time.perf_counter()
    rows, checks = [], []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_replication, replication) for replication in range(REPLICATIONS)]
        for count, future in enumerate(as_completed(futures), 1):
            new_rows, new_checks = future.result()
            rows.extend(new_rows)
            checks.extend(new_checks)
            if count % 25 == 0:
                print(f"Completed {count}/{REPLICATIONS} paired draws; {time.perf_counter()-started:.1f}s", flush=True)
    check_protocol(output)
    assert len(rows) == 9900 and len(checks) == 33
    frame = pd.DataFrame(rows).sort_values(["replication", "clusters", "condition", "ladder", "beta"])
    frame.to_csv(output/"draws.csv", index=False, mode="x")
    summarize(frame).to_csv(output/"summary.csv", index=False, mode="x")
    write_json(output/"full_core_checks.json", checks)
    receipt = {"status": "COMPLETE", "recorded_utc": now(), "elapsed_seconds": time.perf_counter()-started,
               "independent_primitive_draws": REPLICATIONS, "condition_datasets": 2700, "audit_rows": len(frame),
               "core_sha256": sha(CORE_PATH), "script_sha256": sha(SCRIPT_PATH),
               "output_sha256": {name: sha(output/name) for name in ["protocol.json", "benchmark.json", "run_started.json", "draws.csv", "summary.csv", "full_core_checks.json"]}}
    write_json(output/"receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
