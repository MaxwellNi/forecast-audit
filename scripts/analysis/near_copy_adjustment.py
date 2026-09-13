"""Continuous-control sensitivity under an exactly specified rank null.

Every period has the same unordered baseline grid, randomly assigned to entities.
Independent homogeneous score and outcome noise imply a zero conditional rank
covariance given focal baseline rank. All inputs are generated publicly.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm, rankdata, t as student_t
from sklearn.preprocessing import SplineTransformer

from independent_entity_family import calibration_p, wilson

ENTITIES = 80
PERIODS = 25
QS = (8, 12, 16, 24, 32)
METHODS = ("bins32", "extrapolation_beta1", "extrapolation_beta2", "spline", "linear_plus_spline", "oracle_rank_means")
CONFIGS = ((0., 0.),) + tuple((sigma, rho) for sigma in (.01, .1) for rho in (0., .03, .06, .1, .3, .6))


def rank_standardize(values):
    r = rankdata(values, axis=1, method="average")
    scale = r.std(axis=1, ddof=1, keepdims=True)
    return np.divide(r-r.mean(axis=1, keepdims=True), scale, out=np.zeros_like(r), where=scale > 0)


def baseline_bases():
    pct = (np.arange(ENTITIES)+.5)/ENTITIES
    # Every complete training fold contains each grid point 20 times.
    training_pct = np.tile(pct, PERIODS-PERIODS//5)
    spline = SplineTransformer(knots=np.quantile(training_pct, np.linspace(0, 1, 8))[:, None],
                               degree=3, extrapolation="linear", include_bias=False)
    basis = spline.fit_transform(pct[:, None])
    values = {f"bins{q}": np.eye(q)[np.minimum((q*pct).astype(int), q-1)] for q in QS}
    values["spline"] = basis
    values["linear_plus_spline"] = np.column_stack([pct, basis])
    return values


def fit_additive(values, design, penalty):
    """Training-only least squares after exact entity-effect elimination.

    values: periods x entities x targets; design: periods x entities x features.
    The controls contain no held-out outcomes or forecasts. Entity effects use
    the other four fixed whole-period folds, matching the existing comparison.
    """
    values = np.asarray(values, float); design = np.asarray(design, float)
    penalty = np.asarray(penalty, float)
    if values.shape[:2] != design.shape[:2] or penalty.shape != (design.shape[-1],):
        raise ValueError("Aligned target/design arrays and one penalty per feature are required.")
    folds = np.repeat(np.arange(5), PERIODS//5)
    fitted = np.empty_like(values)
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        bx = design[tr]; vy = values[tr]
        mean_b = bx.mean(axis=0); mean_y = vy.mean(axis=0)
        centered_b = (bx-mean_b).reshape(-1, design.shape[-1])
        centered_y = (vy-mean_y).reshape(-1, values.shape[-1])
        gram = centered_b.T@centered_b + np.diag(penalty)
        rhs = centered_b.T@centered_y
        coefficient = np.linalg.pinv(gram, rcond=1e-12)@rhs
        normal_error = np.max(np.abs(gram@coefficient-rhs))/(1+np.max(np.abs(rhs)))
        if normal_error > 1e-9:
            raise RuntimeError("Training normal equations failed.")
        fitted[te] = (design[te]-mean_b)@coefficient+mean_y
    return fitted


def oracle_mean(grid, loading, noise):
    if noise == 0:
        return (np.arange(ENTITIES)+1-(ENTITIES+1)/2)/np.sqrt(ENTITIES*(ENTITIES+1)/12)
    pair = norm.cdf(loading*(grid[:, None]-grid[None, :])/(np.sqrt(2)*noise))
    # The .5 diagonal is not a peer and is replaced by the focal rank's +1.
    expected_rank = .5+pair.sum(axis=1)
    return (expected_rank-(ENTITIES+1)/2)/np.sqrt(ENTITIES*(ENTITIES+1)/12)


def replication(seed):
    rng = np.random.default_rng(seed)
    permutation = np.stack([rng.permutation(ENTITIES) for _ in range(PERIODS)])
    grid = np.linspace(-1., 1., ENTITIES)
    baseline = grid[permutation]
    ex, ey = rng.normal(size=(2, PERIODS, ENTITIES))
    outcome = .6*baseline + .1*ey
    y = rank_standardize(outcome)
    scores = [.8*baseline + sigma*(rho*ey+np.sqrt(1-rho*rho)*ex) for sigma, rho in CONFIGS]
    x = np.stack([rank_standardize(v) for v in scores], axis=-1)
    values = np.concatenate([x, y[:, :, None]], axis=-1)
    residual = {}
    for key, basis in baseline_bases().items():
        penalty = np.zeros(basis.shape[-1]) if key.startswith("bins") else np.full(basis.shape[-1], .001)
        if key == "linear_plus_spline": penalty[0] = 0.
        residual[key] = values-fit_additive(values, basis[permutation], penalty)
    products = {key: value[:, :, :-1]*value[:, :, -1, None] for key, value in residual.items()}
    series = {"bins32": products["bins32"], "spline": products["spline"], "linear_plus_spline": products["linear_plus_spline"]}
    stack = np.stack([products[f"bins{q}"] for q in QS], axis=-1)
    for beta in (1, 2):
        design = np.column_stack([np.ones(len(QS)), np.array(QS, float)**(-beta)])
        weights = np.linalg.pinv(design)[0]
        series[f"extrapolation_beta{beta}"] = stack@weights
    mx = np.stack([oracle_mean(grid, .8, sigma)[permutation] for sigma, rho in CONFIGS], axis=-1)
    my = oracle_mean(grid, .6, .1)[permutation]
    series["oracle_rank_means"] = (x-mx)*(y-my)[:, :, None]
    means = []; ses = []; stats = []; guarded = []
    baseline_ranks = rank_standardize(baseline)
    exact_copy = np.all(x == baseline_ranks[:, :, None], axis=(0, 1))
    for method in METHODS:
        by_period = series[method].mean(axis=1)
        mean = by_period.mean(axis=0)
        se = by_period.std(axis=0, ddof=1)/np.sqrt(PERIODS)
        guard = exact_copy | (se <= 1e-10)
        means.append(mean); ses.append(se)
        stats.append(np.divide(mean, se, out=np.zeros_like(mean), where=se>1e-10))
        guarded.append(guard)
    return np.array(means), np.array(ses), np.array(stats), np.array(guarded)


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def run(output, calibration=1000, evaluation=1000, workers=2):
    output = Path(output)
    if output.exists(): raise FileExistsError("Recorded runs cannot be overwritten.")
    output.mkdir(parents=True)
    start = time.perf_counter()
    protocol = {"calibration_seed_start": 342719001, "evaluation_seed_start": 876510001,
        "calibration_draws": calibration, "evaluation_draws": evaluation,
        "entities": ENTITIES, "periods": PERIODS, "configs_noise_correlation": CONFIGS,
        "methods": METHODS, "workers": workers, "alpha": .05,
        "truth": "At correlation zero, conditional on the focal baseline rank, the competitor baseline multiset is fixed and score/outcome noises are disjoint homogeneous Gaussian arrays. The whole score/outcome rank distributions then factorize. Thus the conditional rank covariance is exactly zero, including after mixing random grid assignments. Exact copy has zero product variance and is a separate degenerate boundary.",
        "controls": "The current five contiguous whole-period complementary folds; additive entity effects plus baseline-percentile bins or splines. Same rows, forecasts, outcomes and training information for all learned fits. Eight nominal quantile knots, cubic spline, sum-loss ridge .001; the added linear term is explicitly unpenalized. Oracle means use the known Gaussian grid law and are a diagnostic comparator.",
        "guard": "Whole-panel equality of forecast and baseline weak orders, or standard error <=1e-10, assigns p=1. All attempts remain in denominators. No R2 threshold is used.",
        "inference": "Normal statistic uses the historical ddof=0 cluster scale. CR1+t uses sample period SD ddof=1 and t24. Independent supplied-null calibration uses plus-one ranks of the ddof=1 statistic from disjoint simulated panels, separately for each noise scale and method.",
        "calibration_scope": "Known simulated null law and a common nominal 5% level, not equal realized error rates or a calibration law for actual data. Single hypothesis per configuration, not a family FDR study.",
        "coupling": "Shared primitives across methods, noise levels and injected correlations in each replication. Calibration and evaluation seeds are disjoint. No replication or cell is excluded.",
        "followup_disclosure": "The original zero/.3/.6 correlation grid was evaluated first and retained in a separate recorded run. Because its positive controls saturated, this follow-up adds .03/.06/.10 correlations on the same primitive seeds. It retains every original configuration and is an exploratory power extension, not independent confirmation of a prespecified power claim.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    phases = {}
    for label, count, seed in (("calibration", calibration, protocol["calibration_seed_start"]),
                               ("evaluation", evaluation, protocol["evaluation_seed_start"])):
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(replication, range(seed, seed+count), chunksize=10))
        phases[label] = tuple(np.stack([row[j] for row in rows]) for j in range(4))
        np.savez_compressed(output/(label+".npz"), mean=phases[label][0], standard_error=phases[label][1],
                            statistic=phases[label][2], guarded=phases[label][3])
        print(f"{label}: {count} draws complete", flush=True)
    means, ses, stats, guarded = phases["evaluation"]
    pvalues = np.ones(stats.shape+(3,))
    pvalues[..., 0] = norm.sf(stats*np.sqrt(PERIODS/(PERIODS-1)))
    pvalues[..., 1] = student_t.sf(stats, PERIODS-1)
    for m, method in enumerate(METHODS):
        for c, (sigma, rho) in enumerate(CONFIGS):
            null_index = CONFIGS.index((sigma, 0.))
            pvalues[:, m, c, 2] = calibration_p(stats[:, m, c], phases["calibration"][2][:, m, null_index])
    pvalues[guarded] = 1.
    np.savez_compressed(output/"pvalues.npz", pvalue=pvalues)
    summary = []; differences = []
    for c, (sigma, rho) in enumerate(CONFIGS):
        for m, method in enumerate(METHODS):
            for j, inference in enumerate(("historical_normal", "cr1_student_t", "independent_supplied_null")):
                rejection = pvalues[:, m, c, j] <= .05
                low, high = wilson(rejection)
                summary.append({"noise_scale": sigma, "injected_correlation": rho, "method": method,
                    "inference": inference, "replications": evaluation, "rejections": int(rejection.sum()),
                    "rejection_rate": float(rejection.mean()), "wilson_low": low, "wilson_high": high,
                    "guarded_fraction": float(guarded[:, m, c].mean()),
                    "mean_product": float(means[:, m, c].mean()), "mean_reported_se_cr1": float(ses[:, m, c].mean()),
                    "empirical_product_sd": float(means[:, m, c].std(ddof=1)),
                    "mean_statistic_cr1": float(stats[:, m, c].mean()),
                    "target_status": "degenerate exact copy" if sigma == 0 else "proved zero rank target" if rho == 0 else "injected score/outcome noise correlation"})
            if rho > 0 and method != "linear_plus_spline":
                delta = (pvalues[:, 4, c, 2] <= .05).astype(float)-(pvalues[:, m, c, 2] <= .05)
                se = float(delta.std(ddof=1)/np.sqrt(evaluation))
                differences.append({"noise_scale": sigma, "injected_correlation": rho,
                    "comparison": "linear_plus_spline_minus_"+method,
                    "paired_power_difference": float(delta.mean()), "mc_se": se,
                    "interval_low": float(delta.mean()-1.96*se), "interval_high": float(delta.mean()+1.96*se),
                    "scope": "independent supplied-null calibration at the same nominal level"})
    csv_write(output/"summary.csv", summary)
    csv_write(output/"paired_power_differences.csv", differences)
    receipt = {"elapsed_seconds": time.perf_counter()-start, "summary_cells": len(summary),
               "source_sha256": protocol["source_sha256"], "status": "completed"}
    (output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    (output/"SHA256SUMS").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.name+"\n"
        for p in sorted(output.iterdir()) if p.is_file()))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=1000)
    parser.add_argument("--evaluation", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.calibration, args.evaluation, args.workers)
