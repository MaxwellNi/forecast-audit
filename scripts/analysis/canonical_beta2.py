"""Paired exponent-two extension of the frozen scalar comparison.

This adds one prespecified statistic to previously generated draws. Existing
comparator outputs are retained, with every data/fold identity re-established
and all binned baseline statistics replayed before an added row is accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm

import canonical_baselines as canonical

NEW_METHOD = "extrapolated_beta2"
DEPENDENCIES = ("numpy", "scipy", "scikit-learn")
ATOL = 2e-12
RTOL = 2e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, allow_nan=False)
        handle.write("\n")


def exponent_weights(beta):
    design = np.column_stack([np.ones(len(canonical.Q_LADDER)),
                              canonical.Q_LADDER.astype(float) ** (-beta)])
    return np.linalg.lstsq(design.T, [1.0, 0.0], rcond=None)[0]


def array_hashes(x, y, z, folds):
    sample = hashlib.sha256(np.column_stack([x, y, z]).astype("<f8").tobytes()).hexdigest()
    fold = hashlib.sha256(np.asarray(folds).astype("<i8").tobytes()).hexdigest()
    return sample, fold


def assert_replay_pair(actual, stored):
    expected = np.array([float(stored["statistic"]), float(stored["pvalue"])])
    actual = np.array(actual, dtype=float)
    if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
        raise AssertionError(f"Stored binned comparator does not replay: {actual} versus {expected}")
    if int(stored["reject"]) != int(actual[1] <= .05):
        raise AssertionError("Replay changed the rejection decision")
    return float(np.max(abs(actual - expected)))


def validate_draw_rows(rows, sample_hash, fold_hash):
    if len(rows) != len(canonical.METHODS) or {r["method"] for r in rows} != set(canonical.METHODS):
        raise AssertionError("Draw must contain exactly the five original methods")
    for field, expected in (("sample_sha256", sample_hash), ("fold_sha256", fold_hash)):
        if {row[field] for row in rows} != {expected}:
            raise AssertionError(f"Paired {field} mismatch")
    for field in ("partial_distance_covariance", "partial_distance_correlation"):
        if len({row[field] for row in rows}) != 1:
            raise AssertionError("Descriptive distance values differ within one draw")


def freeze(protocol_path, original):
    old = json.loads((original / "protocol.json").read_text())
    config = {
        "schema": 1,
        "experiment": "Paired exponent-two extension of the scalar comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_status": "Locally recorded before any new exponent-two outcome; not independently timestamped preregistration. The original exponent-one comparison and separate panel studies were already observed.",
        "script_sha256": sha256(__file__),
        "maintained_harness_sha256": sha256(canonical.__file__),
        "original_files": {name: sha256(original / name) for name in
                           ("protocol.json", "replications.csv", "summary.json")},
        "dependencies": {p: importlib.metadata.version(p) for p in DEPENDENCIES},
        "n": old["n"], "replications": old["replications"], "alpha": old["alpha"],
        "seed_start": old["seed_start"],
        "cells": [[r, a] for r in old["regimes"] for a in old["alternatives"]] + [old["extra_cell"]],
        "q_ladder": canonical.Q_LADDER.tolist(),
        "beta_added": 2,
        "weights_added": exponent_weights(2).tolist(),
        "beta_original": 1,
        "weights_original": canonical.WEIGHTS.tolist(),
        "added_method": NEW_METHOD,
        "existing_methods": list(canonical.METHODS),
        "pairing": "Reuse all 2700 original seeds and balanced two-fold assignments. Regenerate every raw x,y,z and fold array; require SHA256 equality with every original method row. Use the maintained training-fold-only binned predictions and the original fixed ladder. No ranks, subsampling, fixed effects, model selection or alternative ladder.",
        "statistic": "For each observation combine the five residual products using the minimum-Euclidean-norm weights satisfying sum(w)=1 and sum(w*q^-2)=0. Use the original one-sided normal covariance test, standard error sqrt(sum((s-mean(s))^2))/n.",
        "comparison_identity": "Before computing the new statistic for any draw, replay beta-one, coarse-bin and fine-bin statistic/p-value pairs and require original rejection identity, with absolute and relative tolerance 2e-12. All five original comparator rows, including official KCI and spline, are copied without changing any original field. KCI is not run again.",
        "analysis": "Publish all 2700 new rows, all 13500 original comparator rows, all 54 method cells and Wilson 95% intervals. Report the nine paired beta-one/beta-two rejection discordance tables. Report every null, positive-covariance and zero-covariance-dependence result. No choice of exponent, ladder, sample, threshold or model after these outcomes.",
        "limits": "This tests a raw scalar Gaussian-control benchmark with estimated quantile bins. The fixed-resolution beta-two smooth-uniform asymptotic calculation is not automatically its finite-sample justification. Normal and gamma reference tests have different targets and approximate calibration. Rejection on the zero-covariance dependent case is detection for KCI but not power for the residual-covariance test. This extension is not a new independent replication and cannot demonstrate panel calibration or a finite confidence certificate.",
        "failure_policy": "Exclusive output creation, immutable protocol, and persistent started/failed/completed records. On a software failure keep the partial attempt and document an amended protocol before any rerun. No rerun in response to unfavorable scientific outcomes.",
        "resources": {"processes": 1, "numerical_threads": 1, "estimated_minutes": 3,
                      "estimated_memory_gb": 1, "gpu": False},
    }
    write_json(protocol_path, config)
    print(f"Frozen protocol SHA256: {sha256(protocol_path)}", flush=True)


def run(protocol_path, original, output):
    config = json.loads(protocol_path.read_text())
    if sha256(__file__) != config["script_sha256"] or sha256(canonical.__file__) != config["maintained_harness_sha256"]:
        raise RuntimeError("Extension or maintained harness changed after freezing")
    for name, expected in config["original_files"].items():
        if sha256(original / name) != expected:
            raise RuntimeError(f"Original input changed: {name}")
    for name, expected in config["dependencies"].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Dependency mismatch: {name}")
    old_summary = json.loads((original / "summary.json").read_text())
    if old_summary["protocol_sha256"] != config["original_files"]["protocol.json"] or old_summary["raw_rows_sha256"] != config["original_files"]["replications.csv"]:
        raise AssertionError("Original receipt does not bind original inputs")
    output.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    write_json(output / "attempt_started.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path), "script_sha256": sha256(__file__),
        "python": platform.python_version(), "status": "STARTED",
    })
    try:
        _run(config, protocol_path, original, output, begin)
    except BaseException as error:
        write_json(output / "attempt_failed.json", {
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "exception": repr(error), "traceback": traceback.format_exc(),
            "seconds": time.perf_counter() - begin,
        })
        raise


def _run(config, protocol_path, original, output, begin):
    with (original / "replications.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = reader.fieldnames
        source_rows = list(reader)
    groups = {}
    for row in source_rows:
        key = (row["regime"], row["alternative"], int(row["replication"]))
        groups.setdefault(key, []).append(row)
    expected_draws = len(config["cells"]) * config["replications"]
    if len(source_rows) != expected_draws * len(canonical.METHODS) or len(groups) != expected_draws:
        raise AssertionError("Original file has missing or duplicate draws")
    new_weights = np.array(config["weights_added"])
    maximum_errors = {method: 0.0 for method in ("extrapolated", "gcm_coarse", "gcm_fine")}
    methods = list(canonical.METHODS) + [NEW_METHOD]
    results, pairing, comparisons = [], [], []
    rows_path = output / "replications.csv"
    with rows_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields + ["seed", "origin"])
        writer.writeheader()
        for ci, (regime, alternative) in enumerate(config["cells"]):
            method_rows = {m: [] for m in methods}
            for replication in range(config["replications"]):
                seed = config["seed_start"] + ci * 100000 + replication
                x, y, z = canonical.draw_sample(regime, alternative, seed, config["n"])
                folds = canonical.balanced_folds(config["n"], seed + 1000000000)
                sample_hash, fold_hash = array_hashes(x, y, z, folds)
                old_rows = groups.pop((regime, alternative, replication))
                validate_draw_rows(old_rows, sample_hash, fold_hash)
                by_method = {row["method"]: row for row in old_rows}
                start = time.perf_counter()
                values = np.column_stack([x, y])
                products = []
                for q in canonical.Q_LADDER:
                    residuals = values - canonical.binned_predictions(values, z, folds, int(q))
                    products.append(residuals[:, 0] * residuals[:, 1])
                product_matrix = np.column_stack(products)
                replay = {
                    "extrapolated": canonical.covariance_test(product_matrix @ canonical.WEIGHTS),
                    "gcm_coarse": canonical.covariance_test(product_matrix[:, 0]),
                    "gcm_fine": canonical.covariance_test(product_matrix[:, -1]),
                }
                for method, pair in replay.items():
                    maximum_errors[method] = max(maximum_errors[method], assert_replay_pair(pair, by_method[method]))
                statistic, pvalue = canonical.covariance_test(product_matrix @ new_weights)
                new_row = dict(by_method["extrapolated"], method=NEW_METHOD, statistic=statistic,
                               pvalue=pvalue, reject=int(pvalue <= config["alpha"]),
                               seconds=time.perf_counter() - start)
                for row in old_rows + [new_row]:
                    origin = "new_paired_exponent_two" if row["method"] == NEW_METHOD else "original_comparator_unchanged"
                    writer.writerow(dict(row, seed=seed, origin=origin))
                    method_rows[row["method"]].append(row)
                pairing.append({"regime": regime, "alternative": alternative,
                                "replication": replication, "seed": seed,
                                "sample_sha256": sample_hash, "fold_sha256": fold_hash})
            for method, rows in method_rows.items():
                count = sum(int(row["reject"]) for row in rows)
                results.append({"regime": regime, "alternative": alternative, "method": method,
                                "n": config["n"], "replications": len(rows), "rejections": count,
                                "rate": count / len(rows), "wilson95": canonical.wilson(count, len(rows)),
                                "mean_statistic": float(np.mean([float(row["statistic"]) for row in rows])),
                                "seconds": float(sum(float(row["seconds"]) for row in rows)),
                                "timing_origin": "new execution including bin fits and replay checks" if method == NEW_METHOD else "retained original execution"})
            b1 = np.array([int(row["reject"]) for row in method_rows["extrapolated"]])
            b2 = np.array([int(row["reject"]) for row in method_rows[NEW_METHOD]])
            comparisons.append({"regime": regime, "alternative": alternative, "replications": len(b1),
                                "neither_reject": int(np.sum((b1 == 0) & (b2 == 0))),
                                "beta1_only": int(np.sum((b1 == 1) & (b2 == 0))),
                                "beta2_only": int(np.sum((b1 == 0) & (b2 == 1))),
                                "both_reject": int(np.sum((b1 == 1) & (b2 == 1))),
                                "rate_difference_beta2_minus_beta1": float(np.mean(b2 - b1))})
            handle.flush()
            print(f"Completed {regime}/{alternative}: {config['replications']} paired draws", flush=True)
    if groups:
        raise AssertionError("Unused original draws")
    pairing_path = output / "paired_data_hashes.csv"
    with pairing_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairing[0]))
        writer.writeheader()
        writer.writerows(pairing)
    write_json(output / "summary.json", {
        "status": "COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - begin,
        "protocol_sha256": sha256(protocol_path), "script_sha256": sha256(__file__),
        "maintained_harness_sha256": sha256(canonical.__file__),
        "raw_rows_sha256": sha256(rows_path), "paired_data_hashes_sha256": sha256(pairing_path),
        "independent_datasets": expected_draws, "new_datasets": 0,
        "new_statistic_evaluations": expected_draws,
        "retained_comparator_rows": len(source_rows), "total_rows": expected_draws * len(methods),
        "method_cells": len(results), "full_beta1_coarse_fine_replay_max_absolute_errors": maximum_errors,
        "all_sample_and_fold_hashes_match": True,
        "all_original_comparator_fields_retained": True,
        "rows": results, "paired_beta1_beta2": comparisons,
        "limits": config["limits"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.protocol, args.original)
    elif args.output is None:
        parser.error("run requires --output")
    else:
        run(args.protocol, args.original, args.output)


if __name__ == "__main__":
    main()
