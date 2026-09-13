"""Independently recount and spot-replay the matched canonical benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from canonical_baselines import (METHODS, balanced_folds, draw_sample,
                                 evaluate_sample, sha256)


def verify(root):
    config = json.loads((root / "protocol.json").read_text())
    receipt = json.loads((root / "summary.json").read_text())
    raw = pd.read_csv(root / "replications.csv", float_precision="round_trip", keep_default_na=False)
    assert receipt["protocol_sha256"] == sha256(root / "protocol.json")
    assert receipt["raw_rows_sha256"] == sha256(root / "replications.csv")
    cells = [(r, a) for r in config["regimes"] for a in config["alternatives"]]
    cells.append(tuple(config["extra_cell"]))
    assert len(raw) == len(cells) * config["replications"] * len(METHODS)
    assert not raw.duplicated(["regime", "alternative", "replication", "method"]).any()
    assert not raw.isna().any().any()
    assert set(raw.method) == set(METHODS)
    for _, draw in raw.groupby(["regime", "alternative", "replication"]):
        assert len(draw) == len(METHODS)
        assert draw.sample_sha256.nunique() == 1
        assert draw.fold_sha256.nunique() == 1
        assert draw.partial_distance_covariance.nunique() == 1
        assert draw.partial_distance_correlation.nunique() == 1
    assert np.equal(raw.reject, (raw.pvalue <= config["alpha"]).astype(int)).all()
    gcm = raw[raw.method != "kci_gamma"]
    assert_allclose(gcm.pvalue, norm.sf(gcm.statistic), rtol=1e-12, atol=1e-14)
    for row in receipt["rows"]:
        block = raw[(raw.regime == row["regime"]) & (raw.alternative == row["alternative"]) & (raw.method == row["method"])]
        assert len(block) == config["replications"]
        assert set(block.replication) == set(range(config["replications"]))
        assert block.reject.sum() == row["rejections"]
        assert block.reject.mean() == row["rate"]
        independent_interval = proportion_confint(int(block.reject.sum()), len(block), alpha=.05, method="wilson")
        assert_allclose(row["wilson95"], independent_interval, rtol=1e-12, atol=1e-14)
    max_error = 0.0
    for ci, (regime, alternative) in enumerate(cells):
        seed = config["seed_start"] + ci*100000
        x, y, z = draw_sample(regime, alternative, seed, config["n"])
        folds = balanced_folds(config["n"], seed+1000000000)
        expected_sha = hashlib.sha256(np.column_stack([x, y, z]).astype("<f8").tobytes()).hexdigest()
        scores, _, distance = evaluate_sample(x, y, z, folds)
        block = raw[(raw.regime == regime) & (raw.alternative == alternative) & (raw.replication == 0)].set_index("method")
        assert set(block.sample_sha256) == {expected_sha}
        for method in METHODS:
            expected = block.loc[method, ["statistic", "pvalue"]].to_numpy(dtype=float)
            actual = np.array(scores[method])
            assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
            max_error = max(max_error, float(abs(actual-expected).max()))
        assert_allclose(distance, block.iloc[0][["partial_distance_covariance", "partial_distance_correlation"]].to_numpy(dtype=float), atol=1e-13)
    result = {"status": "PASS", "raw_rows": len(raw), "independent_samples": len(raw)//len(METHODS),
              "method_cells": len(receipt["rows"]), "sample_and_fold_identity": True,
              "independent_wilson_intervals": "statsmodels proportion_confint",
              "spot_replayed_samples": len(cells), "spot_replay_max_absolute_error": max_error,
              "protocol_sha256": sha256(root / "protocol.json"), "raw_rows_sha256": sha256(root / "replications.csv"),
              "summary_sha256": sha256(root / "summary.json"), "verifier_sha256": sha256(__file__)}
    try:
        import dcor
        result["distance_reference_version"] = dcor.__version__
        result["distance_reference_source_sha256"] = sha256(inspect.getfile(dcor.partial_distance_correlation))
    except ImportError:
        result["distance_reference_version"] = None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.input)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
