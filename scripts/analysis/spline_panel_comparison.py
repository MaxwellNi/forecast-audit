"""Frozen paired comparison of an explicitly local spline-GCM implementation.

The normal-reference diagnostics are not finite-sample certificates. Existing
forecasts and extrapolation results were known before this extension was frozen.
Run freeze, then run; all outputs are retained, regardless of comparative result.
"""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer
from statsmodels.stats.multitest import multipletests

import audit_panel_predictions as core
import current_core_sensitivity as sensitivity
from audit_decisions import inspect_control_redundancy, apply_decision_policy

SCRIPT = Path(__file__).resolve()
PACKAGE = SCRIPT.parents[2]
OLD = PACKAGE / "results"
OUTPUT = PACKAGE / "results/spline_panel"
LOCAL = Path("local_spline_residuals")
INPUT_SPECIFICATION = None
RIDGE = .001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def domains():
    if INPUT_SPECIFICATION is None:
        raise ValueError("Supply --input-specification with local public forecast files; raw licensed rows are not packaged")
    specification = json.loads(INPUT_SPECIFICATION.read_text())
    order = [("electricity", 14, "D"), ("ratings", 0, None),
             ("retail", 2, "W-FRI"), ("portfolios", 12, "M")]
    return [(name, lag, frequency, [Path(p) for p in specification[name]],
             OLD / ("monthly_portfolios" if name == "portfolios" else "public_nominal_final/" + name) / "extrapolated_profile.csv")
            for name, lag, frequency in order]


def sources():
    return {p.name: sha(p) for p in [SCRIPT, Path(core.__file__), Path(sensitivity.__file__), SCRIPT.with_name("audit_decisions.py"), PACKAGE / "tests/test_spline_panel_comparison.py"]}


def fit_spline(values, control, entities, folds, clusters):
    """Training-only cubic spline ridge plus unpenalized entity effects.

    Minimize sum squared residuals + .001 times squared spline coefficients.
    Eliminate entity effects exactly by within-training-entity demeaning.
    Unseen entities use the training-weighted centered intercept. Training
    quantile knots are deduplicated; one distinct knot means intercept only.
    """
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    control = np.asarray(control, float)
    folds = np.asarray(folds)
    code, levels = pd.factorize(entities, sort=True)
    if not np.isfinite(values).all() or not np.isfinite(control).all() or np.any(code < 0):
        raise ValueError("nonfinite inputs")
    if pd.DataFrame({"c": clusters, "f": folds}).groupby("c").f.nunique().max() != 1:
        raise ValueError("clusters split across folds")
    if len(np.unique(folds)) < 2:
        raise ValueError("two nonempty folds required")
    predictions = np.empty_like(values)
    diagnostics = []
    for fold in np.unique(folds):
        tr, te = folds != fold, folds == fold
        knots = np.unique(np.quantile(control[tr], np.linspace(0, 1, 8)))
        if len(knots) > 1:
            transformer = SplineTransformer(knots=knots[:, None], degree=3, extrapolation="linear", include_bias=False)
            basis = transformer.fit_transform(control[tr, None])
            test_basis = transformer.transform(control[te, None])
        else:
            basis = np.empty((tr.sum(), 0))
            test_basis = np.empty((te.sum(), 0))
        counts = np.bincount(code[tr], minlength=len(levels))
        known = counts > 0
        mean_y = np.zeros((len(levels), values.shape[1]))
        mean_b = np.zeros((len(levels), basis.shape[1]))
        for j in range(values.shape[1]):
            mean_y[known, j] = np.bincount(code[tr], weights=values[tr, j], minlength=len(levels))[known] / counts[known]
        for j in range(basis.shape[1]):
            mean_b[known, j] = np.bincount(code[tr], weights=basis[:, j], minlength=len(levels))[known] / counts[known]
        centered_y = values[tr] - mean_y[code[tr]]
        centered_b = basis - mean_b[code[tr]]
        coefficient = np.linalg.solve(centered_b.T @ centered_b + RIDGE * np.eye(basis.shape[1]), centered_b.T @ centered_y)
        residual_intercept = values[tr].mean(axis=0) - basis.mean(axis=0) @ coefficient
        intercepts = mean_y - mean_b @ coefficient
        intercepts[~known] = residual_intercept
        predictions[te] = test_basis @ coefficient + intercepts[code[te]]
        normal_residual = centered_b.T @ (centered_y - centered_b @ coefficient) - RIDGE * coefficient
        relative_error = float(np.max(np.abs(normal_residual), initial=0) / (1 + np.max(np.abs(centered_b.T @ centered_y), initial=0)))
        if relative_error > 1e-10:
            raise AssertionError("ridge normal equations failed")
        diagnostics.append({"fold": int(fold), "training_rows": int(tr.sum()), "evaluation_rows": int(te.sum()), "distinct_knots": len(knots), "unseen_evaluation_rows": int((~known[code[te]]).sum()), "normal_equation_error": relative_error})
    return predictions, diagnostics


def spline_audit(frame, lag, frequency):
    fr, _ = core.within_period_rank(frame.prediction, frame.period)
    yr, _ = core.within_period_rank(frame.y, frame.period)
    _, z = core.within_period_rank(frame.baseline, frame.period)
    folds = core.period_folds(frame.period, "contiguous")
    fitted, diagnostics = fit_spline(np.column_stack([fr, yr]), z, frame.entity, folds, frame.period)
    residuals = np.column_stack([fr, yr]) - fitted
    products = residuals[:, 0] * residuals[:, 1]
    mean = float(products.mean())
    if frequency is None:
        if lag != 0:
            raise ValueError("unordered clusters require lag zero")
        scores = pd.DataFrame({"s": products - mean, "c": frame.period}).groupby("c").s.sum().to_numpy()[:, None]
        gaps = 0
    else:
        scores, gaps = core.calendar_scores(products - mean, frame.period, frequency)
    se = float(np.sqrt(core.bartlett_score_covariance(scores, lag)[0, 0]) / len(products))
    zero = any(np.max(np.abs(residuals[:, j])) <= 1e-10 * (1 + np.max(np.abs(v))) for j, v in enumerate([fr, yr]))
    defined = not zero and se > max(core.NUMERICAL_SCORE_FLOOR, 8 * np.finfo(float).eps * np.max(np.abs(products)))
    statistic = mean / se if defined else np.nan
    result = {"fold_scheme": "contiguous", "lag": lag, "n": len(frame), "periods": int(frame.period.nunique()), "calendar_gaps": gaps, "mean_product": mean, "standard_error": se, "statistic": statistic, "p_one_sided": float(norm.sf(statistic)) if defined else 1., "inference_status": "computed" if defined else "undefined_rank_or_scale", "nuisance_r2": float(1 - np.mean(residuals[:, 0] ** 2) / np.var(fr)) if np.var(fr) else np.nan}
    return result, residuals, scores, diagnostics


def simulation_replication(replication):
    primitives = sensitivity.primitive_draw(sensitivity.SEED_START + replication)
    rows, fixtures = [], {}
    for clusters in sensitivity.CLUSTERS:
        for condition, frame in zip(sensitivity.CONDITIONS, sensitivity.frames_from_primitives(primitives, clusters)):
            result, residuals, scores, diagnostics = spline_audit(frame, 0, None)
            rows.append({"replication": replication, "seed": sensitivity.SEED_START + replication, "condition": condition, "clusters": clusters, "reject": result["p_one_sided"] <= .05, **result})
            if replication == 0:
                key = f"M{clusters}_{condition}"
                fixtures[key + "_residuals"] = residuals
                fixtures[key + "_clusters"] = frame.period.to_numpy()
    return rows, fixtures


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = []
    for name, lag, frequency, paths, old in domains():
        if len(paths) != (1 if name == "electricity" else 10):
            raise ValueError("missing input family")
        inputs.append({"domain": name, "lag": lag, "frequency": frequency, "predictions": [{"name": p.name, "sha256": sha(p)} for p in paths], "old_profile_sha256": sha(old)})
    protocol = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "status": "local freeze before new spline outcomes; existing datasets and extrapolation outcomes were known", "sources": sources(), "inputs": inputs, "old_panel_draws_sha256": sha(OLD / "current_core_sensitivity/draws.csv"), "method": "Local spline-GCM score, not an official GCM software implementation", "nuisance": "Training quantile cubic B-spline with eight nominal knots including endpoints, distinct quantiles retained, linear extrapolation, include_bias=False; sum-loss ridge .001 on spline coefficients; unpenalized training-only entity effects; unseen entity uses training residual intercept", "rank_and_folds": "Same average within-period ranks standardized by sample SD, same rank-percentile scalar control and same five contiguous whole-period folds as frozen extrapolation; no forward-only audit claim", "simulation": {"replications": 300, "conditions": list(sensitivity.CONDITIONS), "clusters": list(sensitivity.CLUSTERS), "new_rows": 2700, "paired_comparisons_to_existing_33_cells": 9900, "same_primitive_seeds": True, "lag": 0, "no_multiplicity_simulation": "One fixed hypothesis per condition; cells share primitives and are not independent studies"}, "public": "All 41 frozen model vectors in four domains, no refitting of forecasters or row exclusions; unchanged domain HAC lags and complete calendars; all outcomes retained", "decisions": "Raw one-sided normal p values; fixed numerical scale abstention; same outcome-free observed-copy/order guard; raw and guarded BY separately across each complete domain family; no nuisance R2 exclusion", "limitations": "Cross-fitted shared training, approximation bias and serial dependence are not made valid by this nominal comparison; public positive counts are not ground-truth power", "verification": "Dense penalized dummy-design fixture against eliminated-entity solver; evaluation-target perturbation test; independently sum saved OOF residual products and cluster/HAC scores; statsmodels full-family BY; all 33 paired settings retained"}
    protocol["dependencies"] = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn", "statsmodels"]}
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"frozen": str(output), "protocol_sha256": sha(output / "protocol.json")}))


def verify_scores(products, labels, lag, frequency):
    """Separate loop computation, without production score/covariance helpers."""
    mean = float(np.sum(products) / len(products))
    if frequency is None:
        unique = sorted(set(labels))
        codes = np.asarray([unique.index(v) for v in labels])
        length = len(unique)
    else:
        periods = pd.PeriodIndex(pd.to_datetime(labels), freq=frequency)
        codes = periods.asi8 - periods.asi8.min()
        length = int(codes.max()) + 1
    sums = np.bincount(codes, weights=products - mean, minlength=length)
    variance = sum(float(x) ** 2 for x in sums)
    for k in range(1, min(lag, len(sums) - 1) + 1):
        variance += 2 * (1 - k / (lag + 1)) * sum(float(sums[i]) * float(sums[i - k]) for i in range(k, len(sums)))
    return mean, np.sqrt(variance) / len(products)


def run(output, workers):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["sources"] != sources() or (output / "receipt.json").exists():
        raise ValueError("source drift or completed output")
    for record, (_, _, _, paths, old) in zip(protocol["inputs"], domains()):
        assert record["predictions"] == [{"name": p.name, "sha256": sha(p)} for p in paths]
        assert record["old_profile_sha256"] == sha(old)
    assert protocol["old_panel_draws_sha256"] == sha(OLD / "current_core_sensitivity/draws.csv")
    started = time.time()
    rows, fixtures = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch, fixture in pool.map(simulation_replication, range(300)):
            rows.extend(batch)
            fixtures.update(fixture)
    draws = pd.DataFrame(rows).sort_values(["condition", "clusters", "replication"])
    draws.to_csv(output / "panel_spline_draws.csv", index=False)
    np.savez_compressed(output / "synthetic_oof_fixtures.npz", **fixtures)
    old = pd.read_csv(OLD / "current_core_sensitivity/draws.csv", keep_default_na=False)
    joined = old.merge(draws, on=["replication", "seed", "condition", "clusters"], validate="many_to_one", suffixes=("_extrapolated", "_spline"))
    assert len(joined) == 9900
    joined.to_csv(output / "panel_all_33_comparisons.csv", index=False)
    summaries = []
    for (condition, clusters), group in draws.groupby(["condition", "clusters"]):
        count = int(group.reject.sum())
        lo, hi = sensitivity.wilson(count, len(group))
        summaries.append({"condition": condition, "clusters": int(clusters), "replications": len(group), "rejections": count, "rejection_rate": count / len(group), "wilson95_lower": lo, "wilson95_upper": hi, "mean_T": float(group.statistic.mean()), "sd_T": float(group.statistic.std(ddof=1)), "mean_score": float(group.mean_product.mean()), "mean_se_over_empirical_sd": float(group.standard_error.mean() / group.mean_product.std(ddof=1)), "undefined": int((group.inference_status != "computed").sum())})
    pd.DataFrame(summaries).to_csv(output / "panel_spline_summary.csv", index=False)
    public, public_summaries, verifications = [], [], []
    LOCAL.mkdir(parents=True, exist_ok=True)
    for domain, lag, frequency, paths, old_path in domains():
        frame = pd.concat([pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in paths], ignore_index=True).rename(columns={"meter": "entity", "date": "period"})
        old_profile = pd.read_csv(old_path)
        assert set(frame.model) == set(old_profile.model)
        evidence, results = {}, []
        for j, (name, part) in enumerate(frame.groupby("model", sort=True)):
            part = part.reset_index(drop=True)
            expected_n = old_profile.loc[old_profile.model == name, "n"].unique()
            assert len(expected_n) == 1 and len(part) == expected_n[0]
            result, residuals, scores, diagnostics = spline_audit(part, lag, frequency)
            results.append({"model": name, **result})
            evidence[name] = inspect_control_redundancy(part.prediction, part.baseline, part.period)
            fixture_path = LOCAL / f"{domain}_{j:02d}.npz"
            np.savez_compressed(fixture_path, residuals=residuals, clusters=part.period.to_numpy().astype(str), scores=scores)
            mean, se = verify_scores(residuals[:, 0] * residuals[:, 1], part.period.to_numpy(), lag, frequency)
            np.testing.assert_allclose([mean, se], [result["mean_product"], result["standard_error"]], atol=1e-12, rtol=1e-11)
            verifications.append({"domain": domain, "model": name, "local_fixture": fixture_path.name, "sha256": sha(fixture_path), "mean_abs_difference": abs(mean - result["mean_product"]), "se_abs_difference": abs(se - result["standard_error"]), "fits": diagnostics})
        family = sorted(evidence)
        spline_policy = apply_decision_policy(pd.DataFrame(results), evidence, family_models=family, specification_columns=["fold_scheme", "lag"])
        spline_policy["method"] = "local_spline_GCM"
        old_policy = apply_decision_policy(old_profile, evidence, family_models=family, specification_columns=["fold_scheme", "beta", "lag"])
        old_policy["method"] = "extrapolated_beta_" + old_policy.beta.astype(int).astype(str)
        comparison = pd.concat([spline_policy, old_policy], ignore_index=True)
        comparison["domain"] = domain
        for method, part in comparison.groupby("method"):
            rejected, adjusted, _, _ = multipletests(part.p_policy, alpha=.05, method="fdr_by")
            np.testing.assert_allclose(adjusted, part.policy_by_adjusted_p, atol=1e-14, rtol=1e-12)
            np.testing.assert_array_equal(rejected, part.policy_by_reject)
            public_summaries.append({"domain": domain, "method": method, "models": len(part), "raw_nominal": int(part.raw_nominal_positive.sum()), "raw_BY": int(part.raw_by_reject_recomputed.sum()), "guarded_nominal": int(part.policy_nominal_positive.sum()), "guarded_BY": int(part.policy_by_reject.sum()), "copy_order_abstentions": int(part.redundancy_abstain.sum()), "undefined": int(part.undefined_audit_abstain.sum())})
        public.append(comparison)
    pd.concat(public, ignore_index=True).to_csv(output / "public_all_models.csv", index=False)
    pd.DataFrame(public_summaries).to_csv(output / "public_summary.csv", index=False)
    for condition in sensitivity.CONDITIONS:
        for clusters in sensitivity.CLUSTERS:
            key = f"M{clusters}_{condition}"
            r = fixtures[key + "_residuals"]
            mean, se = verify_scores(r[:, 0] * r[:, 1], fixtures[key + "_clusters"], 0, None)
            row = draws[(draws.replication == 0) & (draws.condition == condition) & (draws.clusters == clusters)].iloc[0]
            np.testing.assert_allclose([mean, se], [row.mean_product, row.standard_error], atol=1e-13, rtol=1e-11)
    write_json(output / "verification.json", {"status": "PASS", "public_full_residual_checks": verifications, "synthetic_full_residual_checks": 9, "complete_BY_families": 12, "comparison_rows": len(joined), "spline_simulation_rows": len(draws), "retained_public_model_rows": 123})
    write_json(output / "receipt.json", {"status": "PASS", "elapsed_seconds": time.time() - started, "protocol_sha256": sha(output / "protocol.json"), "sources": sources(), "panel_summary": summaries, "public_summary": public_summaries, "outputs": {p.name: sha(p) for p in sorted(output.glob("*")) if p.is_file()}})
    print(json.dumps({"status": "PASS", "elapsed_seconds": time.time() - started, "panel_summary": summaries, "public_summary": public_summaries}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input-specification", type=Path, required=True)
    parser.add_argument("--local-residuals", type=Path, required=True,
                        help="New local folder; not for redistribution")
    args = parser.parse_args()
    INPUT_SPECIFICATION = args.input_specification
    LOCAL = args.local_residuals
    freeze(args.output) if args.stage == "freeze" else run(args.output, args.workers)
