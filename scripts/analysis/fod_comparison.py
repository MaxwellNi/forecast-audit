"""Classical FOD/predictable-instrument moment on the frozen raw-family DGP.

This is a moment comparison, not an implementation of full panel GMM.
All existing inputs are read-only. The original random streams and batch size
are retained and the three original methods are verified against frozen arrays.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm, t as student_t

HERE = Path(__file__).resolve().parent
DEFAULT_BASE_STUDY = HERE / "independent_entity_family.py"
DEFAULT_BASE_RESULTS = HERE.parents[1] / "results/inference_validation/raw_family_study"
legacy = None
METHODS = ("complementary", "gapped_complementary", "directional", "classical_fod_recursive_instrument")
INFERENCE = ("normal", "student_t", "independent_supplied_null")


def configure_legacy(base_study=None):
    """Load the public adjacent simulator or an explicitly supplied source."""
    global legacy
    path = Path(base_study or os.environ.get("FORECAST_AUDIT_BASE_STUDY", DEFAULT_BASE_STUDY)).resolve()
    spec = importlib.util.spec_from_file_location("frozen_raw_family", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.METHODS) != METHODS[:3] or tuple(module.INFERENCE) != INFERENCE:
        raise ValueError("The supplied base study has different methods or inference rules.")
    legacy = module
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fod_matrix(length=20, used=None):
    """A with score X' A Y; retain classical FOD weights, normalize response.

    At zero-based r: c_r=sqrt((length-r-1)/(length-r)), instrument is
    X_r - mean(X[:r]), and transformed outcome is c_r*(Y_r-mean(Y[r+1:])).
    Dividing the sum by sum(c_r) makes tr(A)=1, so constant contemporaneous
    innovation covariance tau has mean tau. No selection depends on data.
    """
    if used is None:
        _, used = legacy.score_matrices(length)
    used = np.asarray(used, bool)
    if used.shape != (length,) or not used.any() or used[0] or used[-1]:
        raise ValueError("Use nonempty fixed rows with earlier and later observations.")
    rows = np.flatnonzero(used)
    c = np.sqrt((length-rows-1)/(length-rows))
    matrix = np.zeros((length, length))
    for r, factor in zip(rows, c):
        instrument = np.zeros(length)
        instrument[:r] = -1/r
        instrument[r] = 1.
        outcome = np.zeros(length)
        outcome[r] = factor
        outcome[r+1:] = -factor/(length-r-1)
        matrix += np.outer(instrument, outcome) / c.sum()
    return matrix, c


def all_matrices():
    original, used = legacy.score_matrices()
    additional, c = fod_matrix(legacy.LENGTH, used)
    return np.concatenate([original, additional[None]], axis=0), used, c


def analytic_moments(matrices):
    lag = np.arange(legacy.LENGTH)[:, None]-np.arange(legacy.LENGTH)[None, :]
    cross = legacy.LOADING*((lag > 0) & (lag <= legacy.LOOKBACK))
    rows = []
    for signal, (method, matrix) in itertools.product(legacy.SIGNALS, zip(METHODS, matrices)):
        bias = float(np.sum(matrix*cross))
        response = float(np.trace(matrix))
        rows.append({"method": method, "signal": signal,
                     "exact_null_mean": bias, "response_to_constant_covariance": response,
                     "exact_mean_affected_model": bias+signal*response,
                     "exact_mean_unaffected_model": bias,
                     "target_normalized_mean": bias/response+signal})
    return rows


def verify_original(null, means, ses, ts, pvalues, guards, base_results):
    old_null = np.load(base_results / "independent_null_calibration.npz")
    old_eval = np.load(base_results / "evaluation_statistics.npz")
    pairs = {"null_statistic": (null[:, :3], old_null["statistics"]),
             "mean": (means[:, :, :3], old_eval["mean"]),
             "standard_error": (ses[:, :, :3], old_eval["standard_error"]),
             "statistic": (ts[:, :, :3], old_eval["statistic"]),
             "pvalue": (pvalues[:, :, :3], old_eval["pvalue"])}
    result = {}
    for name, (new, old) in pairs.items():
        np.testing.assert_allclose(new, old, rtol=1e-11, atol=1e-12)
        result[name+"_max_absolute_difference"] = float(np.max(np.abs(new-old)))
    np.testing.assert_array_equal(guards[:, :, :3], old_eval["guarded"])
    result["guard_arrays_identical"] = True
    # Tail-rank ordering and final family decisions must be identical, not merely close.
    for k in (10, 11):
        for ref in range(len(INFERENCE)):
            np.testing.assert_array_equal(
                legacy.by_adjust(pvalues[:, :, :3, :, :k, ref]) <= .05,
                legacy.by_adjust(old_eval["pvalue"][..., :k, ref]) <= .05)
    result["all_original_BY_decisions_identical"] = True
    return result


def run(output, calibration=5000, evaluation=1000, batch_size=20,
        base_study=None, base_results=DEFAULT_BASE_RESULTS):
    base_study = configure_legacy(base_study)
    base_results = Path(base_results).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Recorded comparison directories are immutable.")
    if (calibration, evaluation, batch_size) != (5000, 1000, 20):
        raise ValueError("This fixed replay requires 5000/1000 draws and batch size 20.")
    prior = json.loads((base_results / "protocol.json").read_text())
    if digest(base_study) != prior["source_sha256"]:
        raise ValueError("The original simulator has changed; do not silently replay a new law.")
    output.mkdir(parents=True)
    matrices, used, c = all_matrices()
    protocol = {**prior, "methods": METHODS, "batch_size": batch_size,
                "comparison_source_sha256": digest(__file__),
                "legacy_source_sha256": digest(base_study),
                "base_study": "scripts/analysis/"+base_study.name,
                "base_results": "results/inference_validation/"+base_results.name,
                "legacy_record_hashes": {p.name: digest(p) for p in sorted(base_results.iterdir()) if p.is_file()},
                "primary_scope": "G=400, K=11; all-null and signals 0.03, 0.06, 0.10; all existing G,K preserved as sensitivities",
                "classical_source": "Arellano and Bover (1995), Journal of Econometrics 68:29-51, equations (24)-(25), pp. 41-42; https://www.cemfi.es/~arellano/arellano-bover-1995.pdf",
                "FOD": "Use X_t minus mean of X strictly before t as predictable instrument, and c_t times Y_t minus mean of all Y strictly after t, on the same t=5,...,16. Divide summed moments by sum(c_t). Standard FOD moment with a specified instrument, not full panel GMM.",
                "FOD_normalization_sum_c": float(c.sum()),
                "FOD_normalization": "A positive design-only scalar ensures constant innovation-covariance response one. It does not change a studentized statistic.",
                "instrument_under_alternative": "Only null forecasts are predictable relative to the current outcome innovation. Current-innovation injections deliberately break the null and are synthetic positive controls.",
                "paired_intervals": "Per-family power differences; pointwise normal Monte Carlo intervals conditional on the realized calibration banks. Primary G400 K11 additionally uses Bonferroni intervals across 3 signals x 3 comparator methods x 3 inference rules = 27 contrasts. No claim of equal realized FDR.",
                "FDR_bounds": "Pointwise 95% Hoeffding bounds use bounded per-family FDP in [0,1]; for supplied-null inference they cover the conditional FDR given the realized independent bank. They do not integrate new-bank uncertainty and are not simultaneous across cells. Conventional Monte Carlo intervals are also retained.",
                "selection": "The new FOD specification and primary comparisons are fixed before executing this new comparison, after the earlier study had been reviewed. This is a retrospective methodological follow-up, not independent discovery confirmation."}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    legacy.write_csv(output / "analytic_expectations.csv", analytic_moments(matrices))
    np.savez_compressed(output / "score_matrices.npz", matrices=matrices, used=used, fod_factors=c)
    started = time.perf_counter()
    null = np.empty((calibration, len(METHODS), len(legacy.GROUPS)))
    rng = np.random.default_rng(prior["calibration_seed"])
    for begin in range(0, calibration, batch_size):
        scores = legacy.draw_scores(rng, batch_size, 1, (0.,), matrices)[:, 0, :, 0]
        for g, count in enumerate(legacy.GROUPS):
            null[begin:begin+batch_size, :, g] = legacy.statistic(scores[..., :count])[2]
    print("Independent calibration bank complete", flush=True)
    shape = (evaluation, len(legacy.SIGNALS), len(METHODS), len(legacy.GROUPS), 11)
    means, ses, ts = (np.empty(shape) for _ in range(3))
    guards = np.zeros(shape, bool)
    rng = np.random.default_rng(prior["evaluation_seed"])
    for begin in range(0, evaluation, batch_size):
        scores = legacy.draw_scores(rng, batch_size, 11, legacy.SIGNALS, matrices)
        for g, count in enumerate(legacy.GROUPS):
            mu, se, t, guard = legacy.statistic(scores[..., :count])
            means[begin:begin+batch_size, :, :, g] = mu
            ses[begin:begin+batch_size, :, :, g] = se
            ts[begin:begin+batch_size, :, :, g] = t
            guards[begin:begin+batch_size, :, :, g] = guard
    pvalues = np.empty(shape+(len(INFERENCE),))
    pvalues[..., 0] = norm.sf(ts)
    for g, count in enumerate(legacy.GROUPS):
        pvalues[:, :, :, g, :, 1] = student_t.sf(ts[:, :, :, g], count-1)
        for m in range(len(METHODS)):
            pvalues[:, :, m, g, :, 2] = legacy.calibration_p(ts[:, :, m, g], null[:, m, g])
    pvalues[guards] = 1.
    verification = verify_original(null, means, ses, ts, pvalues, guards, base_results)
    np.savez_compressed(output / "independent_null_calibration.npz", statistics=null)
    np.savez_compressed(output / "evaluation_statistics.npz", mean=means, standard_error=ses,
                        statistic=ts, pvalue=pvalues, guarded=guards)
    (output / "original_replay_verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    summary, singles, paired, families = [], [], [], []
    crit = norm.ppf(1-.05/(2*27))
    for s, signal in enumerate(legacy.SIGNALS):
        for g, count in enumerate(legacy.GROUPS):
            for ref, label in enumerate(INFERENCE):
                for k in (10, 11):
                    null_mask = np.ones(k, bool)
                    if signal: null_mask[:3] = False
                    decisions = legacy.by_adjust(pvalues[:, s, :, g, :k, ref]) <= .05
                    rejected = decisions.sum(axis=-1)
                    false = decisions[..., null_mask].sum(axis=-1)
                    fdp = false/np.maximum(rejected, 1)
                    power = decisions[..., :3].mean(axis=-1) if signal else np.zeros_like(fdp)
                    for m, method in enumerate(METHODS):
                        common = {"entities": count, "signal": signal, "family_size": k,
                                  "inference": label, "method": method}
                        fdr = float(fdp[:, m].mean())
                        fdr_se = float(fdp[:, m].std(ddof=1)/np.sqrt(evaluation))
                        fdr_lo, fdr_hi = legacy.wilson(fdp[:, m]) if not signal else (max(0., fdr-1.96*fdr_se), min(1., fdr+1.96*fdr_se))
                        hoeffding_radius = np.sqrt(np.log(2/.05)/(2*evaluation))
                        summary.append({**common, "evaluation_families": evaluation,
                            "FDR": fdr, "FDR_mc_se": fdr_se, "FDR_low": fdr_lo, "FDR_high": fdr_hi,
                            "FDR_Hoeffding95_low": max(0., fdr-hoeffding_radius),
                            "FDR_Hoeffding95_high": min(1., fdr+hoeffding_radius),
                            "FDR_Hoeffding95_scope": "pointwise conditional-bank FDR" if ref == 2 else "pointwise fixed-reference FDR",
                            "mean_rejections": float(rejected[:, m].mean()),
                            "power": float(power[:, m].mean()) if signal else "",
                            "power_mc_se": float(power[:, m].std(ddof=1)/np.sqrt(evaluation)) if signal else "",
                            "guarded_fraction": float(guards[:, s, m, g, :k].mean())})
                        for rep in range(evaluation):
                            families.append({**common, "replication": rep, "rejected": int(rejected[rep, m]),
                                "false_rejected": int(false[rep, m]), "FDP": float(fdp[rep, m]),
                                "power": float(power[rep, m]) if signal else ""})
                    if signal:
                        for comparator in range(3):
                            delta = power[:, 3]-power[:, comparator]
                            d, se = float(delta.mean()), float(delta.std(ddof=1)/np.sqrt(evaluation))
                            paired.append({"entities": count, "signal": signal, "family_size": k,
                                "inference": label, "comparison": "FOD_minus_"+METHODS[comparator],
                                "paired_power_difference": d, "mc_se": se,
                                "pointwise_low": d-1.96*se, "pointwise_high": d+1.96*se,
                                "primary_Bonferroni27_low": d-crit*se if count == 400 and k == 11 else "",
                                "primary_Bonferroni27_high": d+crit*se if count == 400 and k == 11 else "",
                                "reference_interpretation": "common nominal level with supplied null law; realized FDR need not match" if ref == 2 else "nominal reference; biased methods are not size-matched"})
                for m, method in enumerate(METHODS):
                    single = pvalues[:, s, m, g, 0, ref] <= .05
                    low, high = legacy.wilson(single)
                    singles.append({"entities": count, "signal": signal, "method": method, "inference": label,
                        "rejections": int(single.sum()), "rejection_rate": float(single.mean()),
                        "wilson_low": low, "wilson_high": high,
                        "mean_score": float(means[:, s, m, g, 0].mean()),
                        "mean_se": float(ses[:, s, m, g, 0].mean()),
                        "empirical_score_sd": float(means[:, s, m, g, 0].std(ddof=1)),
                        "mean_statistic": float(ts[:, s, m, g, 0].mean())})
    for name, rows in (("family_summary.csv", summary), ("single_model_summary.csv", singles),
                       ("paired_power_differences.csv", paired), ("family_replications.csv.gz", families)):
        legacy.write_csv(output/name, rows)
    unchanged = {p.name: digest(p) for p in sorted(base_results.iterdir()) if p.is_file()}
    if unchanged != protocol["legacy_record_hashes"]:
        raise RuntimeError("An original input changed during the comparison.")
    receipt = {"status": "completed", "elapsed_seconds": time.perf_counter()-started,
               "source_sha256": digest(__file__), "numpy_version": np.__version__,
               "family_summary_cells": len(summary), "family_records": len(families),
               "paired_contrasts": len(paired), "guarded_count": int(guards.sum()),
               "all_original_record_hashes_unchanged": True,
               "all_original_BY_decisions_identical": True,
               "minimum_calibrated_p": 1/(calibration+1),
               "first_BY_threshold_K11": .05/(11*np.sum(1/np.arange(1, 12)))}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (output / "SHA256SUMS").write_text("".join(digest(p)+"  "+p.name+"\n" for p in sorted(output.iterdir()) if p.is_file()))
    print(json.dumps(receipt, indent=2), flush=True)


def verify_saved(output, base_study=None, base_results=DEFAULT_BASE_RESULTS):
    """Recompute normal/rank p-values and every saved family-summary count."""
    import csv
    configure_legacy(base_study)
    output, base_results = Path(output), Path(base_results)
    null = np.load(output / "independent_null_calibration.npz")["statistics"]
    arrays = np.load(output / "evaluation_statistics.npz")
    tvalues, guards = arrays["statistic"], arrays["guarded"]
    pvalues = np.empty(tvalues.shape+(len(INFERENCE),))
    pvalues[..., 0] = norm.sf(tvalues)
    for g, count in enumerate(legacy.GROUPS):
        pvalues[:, :, :, g, :, 1] = student_t.sf(tvalues[:, :, :, g], count-1)
        for m in range(len(METHODS)):
            # Deliberately direct inclusive comparisons, independent of searchsorted.
            reference = null[:, m, g]
            flat = tvalues[:, :, m, g].reshape(-1)
            exact = np.array([(1+np.count_nonzero(reference >= value))/(len(reference)+1) for value in flat])
            pvalues[:, :, m, g, :, 2] = exact.reshape(tvalues[:, :, m, g].shape)
    pvalues[guards] = 1.
    np.testing.assert_allclose(pvalues, arrays["pvalue"], atol=1e-14, rtol=1e-12)
    checked = 0
    with (output / "family_summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            g = list(legacy.GROUPS).index(int(row["entities"]))
            s = list(legacy.SIGNALS).index(float(row["signal"]))
            m, ref, k = METHODS.index(row["method"]), INFERENCE.index(row["inference"]), int(row["family_size"])
            p = pvalues[:, s, m, g, :k, ref]
            # Independently apply step-up rejection thresholds without adjusted values.
            order = np.argsort(p, axis=-1)
            ranked = np.take_along_axis(p, order, axis=-1)
            passes = ranked <= np.arange(1, k+1)*.05/(k*np.sum(1/np.arange(1, k+1)))
            cutoff = np.max(np.where(passes, np.arange(1, k+1), 0), axis=-1)
            selected = np.zeros_like(p, dtype=bool)
            np.put_along_axis(selected, order, np.arange(1, k+1)[None, :] <= cutoff[:, None], axis=-1)
            false = selected[:, 3:].sum(axis=-1) if float(row["signal"]) else selected.sum(axis=-1)
            fdr = np.mean(false/np.maximum(selected.sum(axis=-1), 1))
            np.testing.assert_allclose(fdr, float(row["FDR"]), atol=1e-15)
            np.testing.assert_allclose(selected.sum(axis=-1).mean(), float(row["mean_rejections"]), atol=1e-15)
            if float(row["signal"]):
                np.testing.assert_allclose(selected[:, :3].mean(), float(row["power"]), atol=1e-15)
            radius = np.sqrt(np.log(40)/(2*len(p)))
            np.testing.assert_allclose(max(0., fdr-radius), float(row["FDR_Hoeffding95_low"]), atol=1e-15)
            np.testing.assert_allclose(min(1., fdr+radius), float(row["FDR_Hoeffding95_high"]), atol=1e-15)
            checked += 1
    replay = verify_original(null, arrays["mean"], arrays["standard_error"], tvalues,
                             pvalues, guards, base_results)
    result = {"status": "verified", "family_cells_recomputed": checked,
              "direct_tail_ranks_verified": True, "independent_BY_step_up_verified": True,
              "Hoeffding_bounds_verified": True, **replay}
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--base-study", type=Path, default=DEFAULT_BASE_STUDY)
    parser.add_argument("--legacy-results", type=Path, default=DEFAULT_BASE_RESULTS)
    args = parser.parse_args()
    if args.verify:
        verify_saved(args.verify, args.base_study, args.legacy_results)
    else:
        run(args.output, base_study=args.base_study, base_results=args.legacy_results)
