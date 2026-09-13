#!/usr/bin/env python3
"""Review frozen public-study outputs without changing or refitting the study.

Checks use only the frozen models/choices and already revealed confirmation
observations. Default execution verifies the package without changing files.
They are deterministic verification, not another independent confirmation.
"""
import argparse
import contextlib
from datetime import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.special import betainc
from threadpoolctl import threadpool_limits


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STUDY = HERE


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def student_critical(df):
    """Check recorded producer constants with an independent Student CDF inversion.

    The original SciPy 1.13.1 inverse-CDF approximations define the recorded
    interval arithmetic. SciPy versions differ slightly in those approximations.
    Bisection verifies the mathematical reference; the constants then permit
    checking the recorded endpoints at the original strict arithmetic tolerance.
    """
    recorded = {5: 2.570581835636314, 11: 2.200985160082949}[df]
    lower, upper = 0., 10.
    for _ in range(80):
        midpoint = (lower+upper)/2
        upper_tail = .5*betainc(df/2, .5, df/(df+midpoint*midpoint))
        if upper_tail > 1-.975:
            lower = midpoint
        else:
            upper = midpoint
    independently_inverted = (lower+upper)/2
    assert abs(recorded-independently_inverted) < 1e-10
    return recorded


def main(study_directory=HERE, write_result=False, check_manifest=True):
    global STUDY
    STUDY = Path(study_directory).resolve()
    original_hashes = {p.name: sha(p) for p in STUDY.iterdir() if p.is_file()}
    manifest_entries = 0
    if check_manifest:
        manifest = json.loads((STUDY/"MANIFEST.json").read_text())
        actual = {p.name for p in STUDY.iterdir() if p.is_file() and p.name != "MANIFEST.json"}
        assert actual == set(manifest["files"]), "Manifest file set differs from public package"
        for name, entry in manifest["files"].items():
            assert sha(STUDY/name) == entry["sha256"]
            assert (STUDY/name).stat().st_size == entry["bytes"]
        manifest_entries = len(manifest["files"])
    verifier = load_module("public_verifier", STUDY/"verify.py")
    with contextlib.redirect_stdout(io.StringIO()):
        verifier_output = verifier.main(STUDY)

    study = load_module("public_study", STUDY/"study.py")
    protocol = json.loads((STUDY/"protocol.json").read_text())
    selection = json.loads((STUDY/"selection.json").read_text())
    timing = json.loads((STUDY/"forecast_timing_check.json").read_text())
    started = json.loads((STUDY/"confirmation_started.json").read_text())
    assert selection["protocol_sha256"] == sha(STUDY/"protocol.json")
    assert selection["source_sha256"] == sha(STUDY/"source.zip")
    assert selection["selection_csv_sha256"] == sha(STUDY/"selection_all_candidates.csv")
    assert timing["protocol_sha256"] == sha(STUDY/"protocol.json")
    assert timing["script_sha256"] == sha(STUDY/"study.py")
    assert started["protocol_sha256"] == sha(STUDY/"protocol.json")
    assert started["selection_sha256"] == sha(STUDY/"selection.json")
    assert datetime.fromisoformat(selection["selection_frozen_utc"]) < datetime.fromisoformat(timing["verified_utc"]) < datetime.fromisoformat(started["started_utc"])

    with open(STUDY/"fitted_models.pkl", "rb") as source:
        models, fits = pickle.load(source)
    frame = study.table(STUDY)
    panel, _ = study.build(frame, models, fits)
    co = study.mask(frame, "confirm")
    confirm_positions = np.flatnonzero(co)
    y = frame.y.to_numpy()[co]
    temporal = []
    for cutoff in [confirm_positions[0], confirm_positions[len(confirm_positions)//2], confirm_positions[-1]]:
        changed = frame.copy()
        changed.loc[cutoff:, "y"] += 1_000_000
        other, _ = study.build(changed, models, fits)
        eligible = (np.arange(len(frame)) >= 168) & (np.arange(len(frame)) <= cutoff)
        max_gap = 0.
        for base in study.BASELINES:
            for key in ["b", "mY"]:
                max_gap = max(max_gap, np.max(abs(panel[base][key][eligible]-other[base][key][eligible])))
            for candidate in study.CANDIDATES:
                for key in ["x", "rX"]:
                    max_gap = max(max_gap, np.max(abs(panel[base]["candidates"][candidate][key][eligible]-other[base]["candidates"][candidate][key][eligible])))
        assert max_gap == 0.
        if cutoff+1 < len(frame):
            difference = other[study.BASELINES[0]]["candidates"]["persistence_1h"]["x"][cutoff+1]-panel[study.BASELINES[0]]["candidates"]["persistence_1h"]["x"][cutoff+1]
            assert difference == 1_000_000
        temporal.append(dict(cutoff_row=int(cutoff), cutoff_time=str(frame.date.iloc[cutoff]),
                             largest_at_or_before_prediction_change=float(max_gap)))

    recorded_metrics = pd.read_csv(STUDY/"confirmation_metrics.csv").set_index(["baseline", "method"])
    paired = pd.read_csv(STUDY/"paired_risk_improvements.csv")
    gain_table = pd.read_csv(STUDY/"gain_identity.csv").set_index(["baseline", "candidate"])
    forecast_lookup = {}
    max_metric_error, max_identity_error, max_interval_error = 0., 0., 0.
    selected_results = []
    rng = np.random.default_rng(study.SEED+1)
    bootstrap_indices = rng.integers(0, 12, size=(4000, 12))
    for base in study.BASELINES:
        my = panel[base]["mY"][co]
        predictions = {"baseline": my, "raw_baseline": panel[base]["b"][co]}
        for candidate in study.CANDIDATES:
            fixed = next(row for row in selection["rows"] if row["baseline"] == base and row["candidate"] == candidate)
            x = panel[base]["candidates"][candidate]["x"][co]
            rx = panel[base]["candidates"][candidate]["rX"][co]
            predictions["direct__"+candidate] = x
            predictions["convex__"+candidate] = (1-fixed["alpha"])*my+fixed["alpha"]*x
            predictions["augment__"+candidate] = my+fixed["t"]*rx
            predictions["gated__"+candidate] = predictions["augment__"+candidate] if fixed["gate"] else my
            observed_theta = np.mean(rx*(y-my))
            observed_variance = np.mean(rx*rx)
            empirical_gain = max(observed_theta, 0.)**2/observed_variance if observed_variance > 1e-12 else 0.
            empirical_coefficient = max(observed_theta, 0.)/observed_variance if observed_variance > 1e-12 else 0.
            recorded_identity = gain_table.loc[(base,candidate)]
            assert np.isclose(empirical_gain, recorded_identity["confirm_oracle_gain_diagnostic_only"], rtol=1e-10, atol=1e-7)
            assert np.isclose(empirical_coefficient, recorded_identity["oracle_coefficient_diagnostic_only"], rtol=1e-10, atol=1e-12)
            gain = np.mean((y-my)**2-(y-predictions["augment__"+candidate])**2)
            recomputed = 2*fixed["t"]*np.mean(rx*(y-my))-fixed["t"]**2*np.mean(rx*rx)
            max_identity_error = max(max_identity_error, abs(gain-recomputed), abs(gain-gain_table.loc[(base,candidate),"gain_confirm"]))
        for rule, candidate in selection["choices"][base].items():
            predictions["selected_"+rule] = my if candidate == "baseline" else predictions[rule+"__"+candidate]
        for name, predicted in predictions.items():
            metric = recorded_metrics.loc[(base, name)]
            expected = [np.mean((y-predicted)**2), np.sqrt(np.mean((y-predicted)**2)), np.mean(abs(y-predicted)), np.mean(predicted<0)]
            recorded = metric[["mse", "rmse", "mae", "negative_forecast_fraction"]].to_numpy(float)
            max_metric_error = max(max_metric_error, np.max(abs(np.asarray(expected)-recorded)))
            assert np.allclose(expected, recorded, rtol=1e-12, atol=1e-8)
        selected_results.append(dict(baseline=base,
            max_abs_gated_minus_ungated_prediction=float(np.max(abs(predictions["selected_gated"]-predictions["selected_augment"]))),
            gate_vs_ungated_mse_gain=float(np.mean((y-predictions["selected_augment"])**2-(y-predictions["selected_gated"])**2))))
        forecast_lookup[base] = predictions

    for row in paired.itertuples():
        prediction = forecast_lookup[row.baseline]
        hourly = (y-prediction[row.comparator])**2-(y-prediction[row.method])**2
        weekly = hourly.reshape(12, 168).mean(axis=1)
        for blocks, block_length, low, high in [(12,168,row.ci_low,row.ci_high),(6,336,row.fortnight_low,row.fortnight_high)]:
            means = hourly.reshape(blocks,block_length).mean(axis=1)
            centered = means-means.mean()
            # Explicit covariance quadratic form; separate implementation.
            covariance_sum = np.dot(centered, centered)+np.dot(centered[:-1], centered[1:])
            se = np.sqrt(max(0., covariance_sum/(blocks*(blocks-1))))
            radius = student_critical(blocks-1)*se
            max_interval_error = max(max_interval_error, abs(means.mean()-radius-low), abs(means.mean()+radius-high))
        bootstrap = weekly[bootstrap_indices].mean(axis=1)
        actual = np.quantile(bootstrap,[.025,.975])
        assert np.allclose(actual,[row.bootstrap_low,row.bootstrap_high],rtol=1e-10,atol=1e-7)
    assert max_interval_error < 1e-7
    assert original_hashes == {p.name: sha(p) for p in STUDY.iterdir() if p.is_file()}
    result = dict(passed=True, existing_verifier=verifier_output,
                  complete_manifest_entries_checked=manifest_entries,
                  additional_public_provenance_hash_links_checked=True,
                  producer_student_quantiles_independently_verified=True,
                  forecast_counterfactuals=temporal, hourly_reconstructed_forecasts=len(recorded_metrics),
                  largest_metric_absolute_error=float(max_metric_error),
                  largest_identity_absolute_error=float(max_identity_error),
                  largest_interval_absolute_error=float(max_interval_error),
                  paired_weekly_fortnight_bootstrap_comparisons=len(paired),
                  selected_gate_vs_ungated=selected_results,
                  public_package_files_unchanged=True,
                  scope="Frozen-prediction arithmetic and temporal feature-path checks only; no model refitting, new selection, independent confirmation, population-mean recovery or calibrated temporal inference.")
    if write_result:
        STUDY.joinpath("prediction_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=HERE)
    parser.add_argument("--write-result", action="store_true", help="Refresh prediction_verification.json; default is read-only")
    parser.add_argument("--skip-manifest", action="store_true", help="For package assembly before the manifest is written")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.study_dir, args.write_result, not args.skip_manifest)
