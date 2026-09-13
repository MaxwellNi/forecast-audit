"""Recount the complete extension and replay its weights and score formula.

Comparator identity is checked as original CSV strings. The added score uses
closed-form intercept weights and an explicit variance formula, rather than
the extension's least-squares and covariance-test functions. Bin predictions
are reused from the maintained harness. This is an implementation cross-check,
not a separate research replication or external review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from canonical_baselines import balanced_folds, binned_predictions, draw_sample


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["regime"], row["alternative"], int(row["replication"]), row["method"]


def verify(root, original):
    begin = time.perf_counter()
    protocol = json.loads((root / "protocol.json").read_text())
    receipt = json.loads((root / "summary.json").read_text())
    assert receipt["protocol_sha256"] == digest(root / "protocol.json")
    assert receipt["raw_rows_sha256"] == digest(root / "replications.csv")
    assert receipt["paired_data_hashes_sha256"] == digest(root / "paired_data_hashes.csv")
    assert not (root / "attempt_failed.json").exists()
    for name, expected in protocol["original_files"].items():
        assert digest(original / name) == expected
    raw = csv_rows(root / "replications.csv")
    old = csv_rows(original / "replications.csv")
    assert len(raw) == 16200 and len(old) == 13500
    index = {key(row): row for row in raw}
    assert len(index) == len(raw)
    for row in old:
        new = index[key(row)]
        assert {field: new[field] for field in row} == row
        assert new["origin"] == "original_comparator_unchanged"
    blocks = defaultdict(list)
    for row in raw:
        blocks[(row["regime"], row["alternative"], row["method"])].append(row)
        assert int(row["reject"]) == int(float(row["pvalue"]) <= protocol["alpha"])
        assert 0 <= float(row["pvalue"]) <= 1
        assert np.isfinite(float(row["statistic"]))
        if row["method"] != "kci_gamma":
            np.testing.assert_allclose(float(row["pvalue"]), norm.sf(float(row["statistic"])), rtol=1e-12, atol=1e-14)
    assert len(blocks) == 54 and len(receipt["rows"]) == 54
    table = []
    for summary in receipt["rows"]:
        rows = blocks[(summary["regime"], summary["alternative"], summary["method"])]
        assert len(rows) == 300 and {int(r["replication"]) for r in rows} == set(range(300))
        count = sum(int(r["reject"]) for r in rows)
        assert count == summary["rejections"] and count / 300 == summary["rate"]
        lo, hi = proportion_confint(count, 300, alpha=.05, method="wilson")
        np.testing.assert_allclose([lo, hi], summary["wilson95"], atol=1e-14, rtol=1e-13)
        table.append({"regime": summary["regime"], "alternative": summary["alternative"],
                      "method": summary["method"], "rejections": count, "replications": 300,
                      "rate": count / 300, "wilson95_lower": lo, "wilson95_upper": hi,
                      "mean_statistic": np.mean([float(row["statistic"]) for row in rows])})
    pair_rows = csv_rows(root / "paired_data_hashes.csv")
    pair_index = {(r["regime"], r["alternative"], int(r["replication"])): r for r in pair_rows}
    assert len(pair_rows) == len(pair_index) == 2700
    q = np.array(protocol["q_ladder"], dtype=float)
    inverse_square = 1 / (q * q)
    # Intercept from an unweighted two-column least-squares regression.
    weights = (sum(inverse_square ** 2) - sum(inverse_square) * inverse_square) / (
        len(q) * sum(inverse_square ** 2) - sum(inverse_square) ** 2)
    np.testing.assert_allclose(weights, protocol["weights_added"], rtol=1e-13, atol=1e-13)
    max_error, max_p_error = 0., 0.
    for ci, (regime, alternative) in enumerate(protocol["cells"]):
        for replication in range(300):
            seed = protocol["seed_start"] + ci * 100000 + replication
            x, y, z = draw_sample(regime, alternative, seed, protocol["n"])
            folds = balanced_folds(protocol["n"], seed + 1000000000)
            sample_hash = hashlib.sha256(np.column_stack([x, y, z]).astype("<f8").tobytes()).hexdigest()
            fold_hash = hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest()
            identity = pair_index[(regime, alternative, replication)]
            assert identity["sample_sha256"] == sample_hash and identity["fold_sha256"] == fold_hash
            assert int(identity["seed"]) == seed
            methods = protocol["existing_methods"] + [protocol["added_method"]]
            for method in methods:
                row = index[(regime, alternative, replication, method)]
                assert row["sample_sha256"] == sample_hash and row["fold_sha256"] == fold_hash
                assert int(row["seed"]) == seed
            values = np.column_stack([x, y])
            combined = np.zeros(len(x))
            for bins, weight in zip(protocol["q_ladder"], weights):
                residuals = values - binned_predictions(values, z, folds, bins)
                combined += weight * residuals[:, 0] * residuals[:, 1]
            mean = np.mean(combined)
            se = np.std(combined, ddof=0) / np.sqrt(len(combined))
            statistic = mean / se
            pvalue = norm.sf(statistic)
            stored = index[(regime, alternative, replication, protocol["added_method"])]
            err = abs(float(stored["statistic"]) - statistic)
            p_err = abs(float(stored["pvalue"]) - pvalue)
            max_error, max_p_error = max(max_error, err), max(max_p_error, p_err)
            np.testing.assert_allclose([statistic, pvalue], [float(stored["statistic"]), float(stored["pvalue"])], rtol=2e-12, atol=2e-12)
            assert int(stored["reject"]) == int(pvalue <= protocol["alpha"])
    for row in receipt["paired_beta1_beta2"]:
        b1 = np.array([int(index[(row["regime"], row["alternative"], r, "extrapolated")]["reject"]) for r in range(300)])
        b2 = np.array([int(index[(row["regime"], row["alternative"], r, protocol["added_method"])]["reject"]) for r in range(300)])
        assert row["neither_reject"] == np.sum((b1 == 0) & (b2 == 0))
        assert row["beta1_only"] == np.sum((b1 == 1) & (b2 == 0))
        assert row["beta2_only"] == np.sum((b1 == 0) & (b2 == 1))
        assert row["both_reject"] == np.sum((b1 == 1) & (b2 == 1))
        assert row["rate_difference_beta2_minus_beta1"] == np.mean(b2 - b1)
    with (root / "comparison_coordinates.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    result = {"status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "seconds": time.perf_counter() - begin, "raw_rows": len(raw),
              "unchanged_original_rows": len(old), "method_cells": len(blocks),
              "full_sample_and_fold_hashes_replayed": len(pair_rows),
              "new_statistics_replayed_with_closed_form_weights_and_variance": 2700,
              "maximum_statistic_absolute_error": max_error, "maximum_pvalue_absolute_error": max_p_error,
              "wilson_reference": "statsmodels.stats.proportion.proportion_confint",
              "scope": "Separate arithmetic implementation by the same study author; no external review or new statistical replication. Maintained nuisance prediction routine is shared.",
              "protocol_sha256": digest(root / "protocol.json"),
              "raw_rows_sha256": digest(root / "replications.csv"),
              "summary_sha256": digest(root / "summary.json"),
              "coordinates_sha256": digest(root / "comparison_coordinates.csv"),
              "verifier_sha256": digest(__file__)}
    with (root / "verification.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.input, args.original), indent=2))
