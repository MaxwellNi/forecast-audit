"""Independent calibration and full-family evaluation of raw panel scores.

This fully synthetic study uses independent entity trajectories, fixed temporal
folds, and equal entity weights. Supplied-null calibration uses the known data
generator. It is not a calibration procedure for an unknown real panel law.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm, t as student_t

METHODS = ("complementary", "gapped_complementary", "directional")
GROUPS = (25, 100, 400)
SIGNALS = (0., .03, .06, .10)
INFERENCE = ("normal", "student_t", "independent_supplied_null")
LENGTH = 20
LOOKBACK = 5
LOADING = .2


def score_matrices(length=LENGTH, lookback=LOOKBACK):
    """Exact bilinear form for the held-out residual product average."""
    if length < 5 or lookback < 1:
        raise ValueError("Require five nonempty folds and a positive lookback.")
    fold = np.empty(length, int)
    for f, indices in enumerate(np.array_split(np.arange(length), 5)):
        fold[indices] = f
    used = (fold > 0) & (fold < 4)
    matrices = np.zeros((len(METHODS), length, length))
    times = np.arange(length)
    for current in np.flatnonzero(used):
        f = fold[current]
        block = np.flatnonzero(fold == f)
        other = fold != f
        gap = (times < block[0] - lookback) | (times > block[-1] + lookback)
        for m, (earlier, later) in enumerate(((other, other), (gap, gap), (fold < f, fold > f))):
            if not earlier.any() or not later.any():
                raise ValueError("Every evaluated row needs both nuisance training sets.")
            wx = -earlier.astype(float) / earlier.sum()
            wy = -later.astype(float) / later.sum()
            wx[current] += 1
            wy[current] += 1
            matrices[m] += np.outer(wx, wy) / used.sum()
    return matrices, used


def statistic(entity_scores):
    """The existing raw branch already uses the G/(G-1) sample correction."""
    values = np.asarray(entity_scores, float)
    if values.ndim < 1 or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite entity scores are required.")
    mean = values.mean(axis=-1)
    se = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
    guarded = (~np.isfinite(se)) | (se <= 1e-10)
    t = np.divide(mean, se, out=np.zeros_like(mean), where=~guarded)
    return mean, se, t, guarded


def by_adjust(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim < 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite p-values in [0,1] are required.")
    k = p.shape[-1]
    order = np.argsort(p, axis=-1, kind="stable")
    ordered = np.take_along_axis(p, order, axis=-1)
    adjusted = ordered * k * np.sum(1 / np.arange(1, k + 1)) / np.arange(1, k + 1)
    adjusted = np.minimum.accumulate(adjusted[..., ::-1], axis=-1)[..., ::-1]
    out = np.empty_like(p)
    np.put_along_axis(out, order, np.minimum(adjusted, 1.), axis=-1)
    return out


def calibration_p(values, reference):
    """Conservative plus-one ranks, including ties, against independent draws."""
    ref = np.sort(np.asarray(reference, float))
    if ref.ndim != 1 or not len(ref) or not np.isfinite(ref).all():
        raise ValueError("Supply finite independent null statistics.")
    return (1 + len(ref) - np.searchsorted(ref, values, side="left")) / (len(ref) + 1)


def draw_scores(rng, batch, models, signal_values, matrices):
    """Return batch x signal x method x model x entity raw scores."""
    groups = max(GROUPS)
    shocks = rng.normal(size=(batch, groups, LENGTH + LOOKBACK))
    current = shocks[..., LOOKBACK:]
    past = sum(shocks[..., LOOKBACK-h:LOOKBACK+LENGTH-h] for h in range(1, LOOKBACK+1))
    ay = rng.normal(size=(batch, groups, 1))
    ax = rng.normal(size=(batch, models, groups, 1))
    common_noise = rng.normal(size=(batch, 1, groups, LENGTH))
    own_noise = rng.normal(size=(batch, models, groups, LENGTH))
    x = ax + LOADING * (LOOKBACK * ay[:, None] + past[:, None]) + (common_noise + own_noise) / np.sqrt(2.)
    y = current + ay
    projected_y = np.einsum("mtu,bgu->bmgt", matrices, y, optimize=True)
    base = np.einsum("bkgt,bmgt->bmkg", x, projected_y, optimize=True)
    increment = np.einsum("bgt,bmgt->bmg", current, projected_y, optimize=True)
    affected = np.arange(models) < 3
    return np.stack([base + signal * increment[:, :, None, :] * affected[None, None, :, None]
                     for signal in signal_values], axis=1)


def wilson(binary):
    n = len(binary); p = float(np.mean(binary)); z = norm.ppf(.975)
    denominator = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denominator
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0., centre-radius), min(1., centre+radius)


def write_csv(path, rows):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def run(output, calibration=5000, evaluation=1000, batch_size=20):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new directory; recorded runs are immutable.")
    if calibration < 2 or evaluation < 2 or batch_size < 1:
        raise ValueError("At least two calibration/evaluation draws are required.")
    output.mkdir(parents=True)
    start = time.perf_counter()
    matrices, used = score_matrices()
    protocol = {
        "calibration_seed": 491732061, "evaluation_seed": 670921843,
        "calibration_draws": calibration, "evaluation_families": evaluation,
        "entities": GROUPS, "times": LENGTH, "lookback": LOOKBACK,
        "past_loading": LOADING, "family_sizes": [10, 11],
        "signals_for_first_three_models": SIGNALS, "methods": METHODS,
        "inference": INFERENCE, "alpha": .05, "nuisance_folds": 5,
        "evaluation_rows_per_entity": int(used.sum()),
        "truth": "Null model has zero raw conditional innovation covariance. Directional fitted raw product expectation is exactly zero. Complementary and gapped fitted products may be biased for that target.",
        "data": "Independent entity trajectories; Gaussian innovations and intercepts. Forecast uses the preceding five outcome innovations plus unit-variance noise. Forecast noise has correlation 0.5 across models; outcomes are shared. No ranks, bins, estimated spline, or HAC is used.",
        "pairing": "Entity counts are nested, family sizes are nested, and all methods and injected alternatives share primitives within a family replication. Calibration and evaluation use independent RNG streams.",
        "guards": "Require complete fixed support with nonempty earlier/later training sides; finite raw inputs; at least two entities; assign p=1 whenever sample standard error is <=1e-10. All family members remain in BY. No p-value or outcome-dependent model filtering.",
        "studentization": "Sample SD uses ddof=1, so G/(G-1) is already present; t(G-1) changes only the reference law. This does not repair nuisance bias.",
        "calibration": "Each method and entity count has a separate bank of independent null statistics from the known generator. Plus-one upper-tail ranks include ties. The same independent bank is used for all exchangeable model marginals; no calibration/evaluation draws overlap.",
        "scope": "Complete raw equal-entity branch with simulated known null law, not the ranked observed-peer Algorithm 1 and not a real-panel or universal FDR guarantee. Alternatives inject current outcome information and are not prospective forecasts.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rng = np.random.default_rng(protocol["calibration_seed"])
    null_stats = np.empty((calibration, len(METHODS), len(GROUPS)))
    for begin in range(0, calibration, batch_size):
        n = min(batch_size, calibration-begin)
        scores = draw_scores(rng, n, 1, (0.,), matrices)[:, 0, :, 0]
        for g, count in enumerate(GROUPS):
            null_stats[begin:begin+n, :, g] = statistic(scores[..., :count])[2]
    np.savez_compressed(output / "independent_null_calibration.npz", statistics=null_stats)
    print(f"Calibration complete: {calibration} independent panels", flush=True)
    rng = np.random.default_rng(protocol["evaluation_seed"])
    shape = (evaluation, len(SIGNALS), len(METHODS), len(GROUPS), 11)
    means, ses, ts = (np.empty(shape) for _ in range(3)); guards = np.zeros(shape, bool)
    for begin in range(0, evaluation, batch_size):
        n = min(batch_size, evaluation-begin)
        scores = draw_scores(rng, n, 11, SIGNALS, matrices)
        for g, count in enumerate(GROUPS):
            mu, se, t, guard = statistic(scores[..., :count])
            means[begin:begin+n, :, :, g] = mu
            ses[begin:begin+n, :, :, g] = se
            ts[begin:begin+n, :, :, g] = t
            guards[begin:begin+n, :, :, g] = guard
    pvalues = np.empty(shape + (len(INFERENCE),))
    pvalues[..., 0] = norm.sf(ts)
    for g, count in enumerate(GROUPS):
        pvalues[:, :, :, g, :, 1] = student_t.sf(ts[:, :, :, g], count-1)
        for m in range(len(METHODS)):
            pvalues[:, :, m, g, :, 2] = calibration_p(ts[:, :, m, g], null_stats[:, m, g])
    pvalues[guards] = 1.
    np.savez_compressed(output / "evaluation_statistics.npz", mean=means, standard_error=ses,
                        statistic=ts, pvalue=pvalues, guarded=guards)
    print(f"Evaluation complete: {evaluation} independent families", flush=True)
    summary = []; families = []; singles = []; pairdiff = []
    for s, signal in enumerate(SIGNALS):
        for g, count in enumerate(GROUPS):
            for m, method in enumerate(METHODS):
                for inference, label in enumerate(INFERENCE):
                    single = pvalues[:, s, m, g, 0, inference] <= .05
                    lo, hi = wilson(single)
                    singles.append({"entities": count, "signal": signal, "method": method,
                        "inference": label, "replications": evaluation,
                        "rejections": int(single.sum()), "rejection_rate": float(single.mean()),
                        "wilson_low": lo, "wilson_high": hi,
                        "mean_score": float(means[:, s, m, g, 0].mean()),
                        "mean_se": float(ses[:, s, m, g, 0].mean()),
                        "empirical_score_sd": float(means[:, s, m, g, 0].std(ddof=1)),
                        "mean_statistic": float(ts[:, s, m, g, 0].mean())})
                    for k in (10, 11):
                        decisions = by_adjust(pvalues[:, s, m, g, :k, inference]) <= .05
                        null = np.ones(k, bool)
                        if signal > 0: null[:3] = False
                        rejected = decisions.sum(axis=1)
                        false = decisions[:, null].sum(axis=1)
                        fdp = false / np.maximum(rejected, 1)
                        power = decisions[:, ~null].mean(axis=1) if (~null).any() else np.zeros(evaluation)
                        common = {"entities": count, "signal": signal, "method": method,
                                  "inference": label, "family_size": k}
                        for rep in range(evaluation):
                            families.append({**common, "replication": rep, "rejected": int(rejected[rep]),
                                "false_rejected": int(false[rep]), "false_discovery_proportion": float(fdp[rep]),
                                "true_positive_fraction": float(power[rep]) if signal else ""})
                        mc = float(fdp.std(ddof=1)/np.sqrt(evaluation))
                        if signal == 0:
                            low, high = wilson(fdp)
                        else:
                            low, high = max(0., float(fdp.mean()-1.96*mc)), min(1., float(fdp.mean()+1.96*mc))
                        summary.append({**common, "evaluation_families": evaluation,
                            "mean_rejections": float(rejected.mean()), "FDR": float(fdp.mean()),
                            "FDR_mc_se": mc, "FDR_interval_low": low, "FDR_interval_high": high,
                            "FDR_interval_type": "pointwise Wilson" if signal == 0 else "pointwise normal Monte Carlo",
                            "FDR_hoeffding_low": max(0., float(fdp.mean()-np.sqrt(np.log(40.)/(2*evaluation)))),
                            "FDR_hoeffding_high": min(1., float(fdp.mean()+np.sqrt(np.log(40.)/(2*evaluation)))),
                            "FDR_hoeffding_scope": "pointwise finite-evaluation 95% bound, conditional on the independent calibration bank",
                            "power": float(power.mean()) if signal else "",
                            "power_mc_se": float(power.std(ddof=1)/np.sqrt(evaluation)) if signal else "",
                            "guarded_fraction": float(guards[:, s, m, g, :k].mean())})
            for k in (10, 11):
                if signal == 0: continue
                for inference, label in enumerate(INFERENCE):
                    for comparator in (0, 1):
                        decisions_a = by_adjust(pvalues[:, s, 2, g, :k, inference]) <= .05
                        decisions_b = by_adjust(pvalues[:, s, comparator, g, :k, inference]) <= .05
                        delta = decisions_a[:, :3].mean(axis=1)-decisions_b[:, :3].mean(axis=1)
                        se = float(delta.std(ddof=1)/np.sqrt(evaluation))
                        pairdiff.append({"entities": count, "signal": signal, "family_size": k,
                            "inference": label, "comparison": "directional_minus_"+METHODS[comparator],
                            "paired_power_difference": float(delta.mean()), "mc_se": se,
                            "interval_low": float(delta.mean()-1.96*se), "interval_high": float(delta.mean()+1.96*se),
                            "interpretation": "common nominal level with independent supplied-null calibration" if inference == 2 else "normal or t reference at the common nominal level"})
    write_csv(output / "family_summary.csv", summary)
    write_csv(output / "family_replications.csv.gz", families)
    write_csv(output / "single_model_summary.csv", singles)
    write_csv(output / "paired_power_differences.csv", pairdiff)
    expected = []
    lag = np.arange(LENGTH)[:, None]-np.arange(LENGTH)[None, :]
    cross = LOADING*((lag > 0) & (lag <= LOOKBACK))
    for signal in SIGNALS:
        for m, method in enumerate(METHODS):
            expected.append({"signal": signal, "method": method,
                             "exact_mean_raw_score": float(np.sum(matrices[m]*(cross+signal*np.eye(LENGTH))))})
    write_csv(output / "analytic_expectations.csv", expected)
    receipt = {"status": "completed", "source_sha256": protocol["source_sha256"],
               "elapsed_seconds": time.perf_counter()-start, "numpy_version": np.__version__,
               "family_summary_cells": len(summary), "family_records": len(families),
               "all_guarded_count": int(guards.sum()), "minimum_mc_p": 1/(calibration+1),
               "first_BY_threshold_10": .05/(10*np.sum(1/np.arange(1, 11))),
               "first_BY_threshold_11": .05/(11*np.sum(1/np.arange(1, 12)))}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("".join(hashlib.sha256(path.read_bytes()).hexdigest()+"  "+path.name+"\n"
        for path in sorted(output.iterdir()) if path.is_file()))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=5000)
    parser.add_argument("--evaluation", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    run(args.output, args.calibration, args.evaluation, args.batch_size)
