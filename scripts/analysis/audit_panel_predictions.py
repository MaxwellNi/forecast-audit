"""Audit panel predictions with complete training-fold nuisance fitting.

The routines implement the separately recorded real-audit protocol. Returned
normal p-values are diagnostics whose validity still requires a sampling model,
negligible nuisance bias, and an adequate dependence approximation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from cluster_covariance_reference import cluster_standard_error

# Scores are products of within-cluster standardized ranks. Below this fixed
# absolute scale, sparse-solver roundoff must not be studentized into evidence.
# This is a numerical abstention rule, not a statistical calibration correction.
NUMERICAL_SCORE_FLOOR = 1e-10


def by_adjust(pvalues, alpha=0.05):
    """Benjamini-Yekutieli over the complete input family, without filtering."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be a nonempty vector in [0, 1]")
    m = len(p)
    harmonic = float(np.sum(1 / np.arange(1, m + 1)))
    order = np.argsort(p, kind="stable")
    adjusted_ordered = np.minimum.accumulate(
        (p[order] * m * harmonic / np.arange(1, m + 1))[::-1]
    )[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted <= alpha, adjusted


def period_folds(periods, scheme="contiguous", n_folds=5):
    codes, unique = pd.factorize(periods, sort=True)
    if (codes < 0).any() or len(unique) < n_folds:
        raise ValueError("at least n_folds nonmissing periods are required")
    if scheme == "contiguous":
        membership = np.empty(len(unique), dtype=int)
        for k, block in enumerate(np.array_split(np.arange(len(unique)), n_folds)):
            membership[block] = k
    elif scheme == "interleaved":
        membership = np.arange(len(unique)) % n_folds
    else:
        raise ValueError("unknown fold scheme")
    return membership[codes]


def within_period_rank(values, periods):
    data = pd.DataFrame({"value": np.asarray(values, float), "period": np.asarray(periods)})
    ranks = data.groupby("period", sort=False)["value"].rank(method="average")
    grouped = ranks.groupby(data.period, sort=False)
    scale = grouped.transform("std").fillna(0).to_numpy()
    centered = (ranks - grouped.transform("mean")).to_numpy()
    standardized = np.divide(centered, scale, out=np.zeros(len(data)), where=scale > 0)
    percentile = (ranks.to_numpy() - 0.5) / grouped.transform("size").to_numpy()
    return standardized, percentile


def crossfit_additive_least_squares(values, groups, folds, clusters):
    """Joint additive least squares using an explicitly checked sparse solve.

    The minimum-norm solver resolves aliased group columns deterministically.
    Observed group effects are centred within training data; an unseen level
    has zero effect. Evaluation targets never enter a training design or solve.
    """
    values = np.asarray(values, float)
    folds = np.asarray(folds)
    check = pd.DataFrame({"cluster":np.asarray(clusters), "fold":folds}).groupby("cluster").fold.nunique()
    if check.max() != 1 or not np.isfinite(values).all():
        raise ValueError("invalid cluster partition or targets")
    encoded = [pd.factorize(group, sort=True) for group in groups]
    if any((codes < 0).any() for codes, _ in encoded):
        raise ValueError("missing group label")
    offsets = np.cumsum([1] + [len(unique) for _, unique in encoded])
    n = len(values)
    row_indices = np.tile(np.arange(n), len(groups) + 1)
    columns = np.concatenate([np.zeros(n, dtype=int)] +
                             [codes + offsets[j] for j, (codes, _) in enumerate(encoded)])
    design = csr_matrix((np.ones(len(columns)), (row_indices, columns)), shape=(n, offsets[-1]))
    fitted = np.empty(n)
    for fold in np.unique(folds):
        training = folds != fold
        evaluation = ~training
        train_design = design[training]
        target = values[training]
        solution = lsmr(train_design, target, atol=1e-12, btol=1e-12, conlim=1e12,
                        maxiter=max(2000, 4 * design.shape[1]))
        if solution[1] not in [0, 1, 2, 4, 5]:
            raise RuntimeError(f"additive sparse solve did not converge: code {solution[1]}")
        coefficient = solution[0].copy()
        residual = target - train_design @ coefficient
        normal_residual = np.asarray(train_design.T @ residual)
        denominator = 1 + np.linalg.norm(train_design.T @ target, ord=np.inf)
        if np.linalg.norm(normal_residual, ord=np.inf) / denominator > 1e-9:
            raise RuntimeError("additive least-squares normal equations failed")
        for j, (codes, unique) in enumerate(encoded):
            count = np.bincount(codes[training], minlength=len(unique))
            effects = coefficient[offsets[j]:offsets[j + 1]]
            centre = float(count @ effects / training.sum())
            effects[count > 0] -= centre
            effects[count == 0] = 0
            coefficient[0] += centre
        fitted[evaluation] = design[evaluation] @ coefficient
    return fitted


def panel_residuals(frame, fold_scheme="contiguous", q=10):
    """Return residuals and fits; required columns are entity, period, y, prediction, baseline."""
    columns = ["entity", "period", "y", "prediction", "baseline"]
    if any(c not in frame for c in columns):
        raise ValueError(f"required columns: {columns}")
    if frame[columns].isna().any().any() or not np.isfinite(frame[["y", "prediction", "baseline"]].to_numpy(float)).all():
        raise ValueError("input contains missing or nonfinite values")
    if frame.duplicated(["entity", "period"]).any():
        raise ValueError("duplicate entity-period rows")
    fr, _ = within_period_rank(frame.prediction, frame.period)
    yr, _ = within_period_rank(frame.y, frame.period)
    _, baseline_percentile = within_period_rank(frame.baseline, frame.period)
    if isinstance(q, bool) or int(q) != q or q < 2:
        raise ValueError("q must be an integer at least two")
    bins = np.minimum((q * baseline_percentile).astype(int), q - 1)
    folds = period_folds(frame.period, scheme=fold_scheme)
    groups = [frame.entity.to_numpy(), bins]
    fit_f = crossfit_additive_least_squares(fr, groups, folds, frame.period)
    fit_y = crossfit_additive_least_squares(yr, groups, folds, frame.period)
    return {"forecast_rank": fr, "outcome_rank": yr, "forecast_fit": fit_f,
            "outcome_fit": fit_y, "forecast_residual": fr - fit_f,
            "outcome_residual": yr - fit_y, "folds": folds, "baseline_bins": bins}


def calendar_scores(scores, periods, frequency):
    """Sum row scores into a complete calendar, preserving gaps as zero sums."""
    labels = pd.PeriodIndex(pd.to_datetime(np.asarray(periods)), freq=frequency)
    array = np.asarray(scores, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) != len(labels) or not np.isfinite(array).all() or labels.isna().any():
        raise ValueError("invalid scores or calendar periods")
    grouped = pd.DataFrame(array, index=labels).groupby(level=0).sum()
    calendar = pd.period_range(labels.min(), labels.max(), freq=frequency)
    return grouped.reindex(calendar, fill_value=0).to_numpy(), len(calendar) - len(grouped)


def bartlett_score_covariance(scores, lag):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or not np.isfinite(scores).all() or isinstance(lag, bool) or int(lag) != lag or lag < 0:
        raise ValueError("invalid score matrix or lag")
    covariance = scores.T @ scores
    for k in range(1, min(lag, len(scores) - 1) + 1):
        cross = scores[k:].T @ scores[:-k]
        covariance += (1 - k / (lag + 1)) * (cross + cross.T)
    return covariance


def audit_panel(frame, lags, frequency, fold_scheme="contiguous"):
    fit = panel_residuals(frame, fold_scheme)
    products = fit["forecast_residual"] * fit["outcome_residual"]
    n = len(products)
    mean = float(products.mean())
    if frequency is None:
        if list(lags) != [0]:
            raise ValueError("unordered clusters support only lag zero")
        scores = pd.DataFrame({"score":products-mean, "period":frame.period}).groupby("period", observed=False).score.sum().to_numpy()[:,None]
        missing_periods = 0
    else:
        scores, missing_periods = calendar_scores(products - mean, frame.period, frequency)
    cluster_se = cluster_standard_error(products, frame.period)
    base_scale = float(fit["forecast_rank"].var())
    r2 = 1 - float(np.mean(fit["forecast_residual"]**2)) / base_scale if base_scale > 0 else None
    numerical_residual_zero = any(
        np.max(np.abs(fit[f"{side}_residual"])) <= 1e-10 * (1 + np.max(np.abs(fit[f"{side}_rank"])))
        for side in ["forecast","outcome"])
    rows = []
    for lag in lags:
        variance = float(bartlett_score_covariance(scores, lag)[0, 0]) / n ** 2
        if variance < 0 or not np.isfinite(variance):
            raise ValueError("negative or nonfinite score variance")
        se = float(np.sqrt(variance))
        if lag == 0 and not np.isclose(se, cluster_se, rtol=1e-10, atol=1e-13):
            raise AssertionError("HAC lag zero disagrees with reference cluster standard error")
        defined = base_scale > 0 and not numerical_residual_zero and se > max(
            NUMERICAL_SCORE_FLOOR, 8*np.finfo(float).eps*np.max(np.abs(products)))
        statistic = mean / se if defined else np.nan
        rows.append({"fold_scheme": fold_scheme, "lag": int(lag), "n": n,
                     "periods": int(frame.period.nunique()), "calendar_gaps": missing_periods,
                     "mean_product": mean, "standard_error": se, "statistic": statistic,
                     "p_one_sided": float(norm.sf(statistic)) if defined else 1.0,
                     "inference_status":"computed" if defined else "undefined_rank_or_scale",
                     "nuisance_r2": r2,
                     "overlap_screen": bool(r2 is not None and r2 > 0.2)})
    return rows


def audit_extrapolated_panel(frame, lag, frequency, qs=(8,12,16,24,32), betas=(1,2)):
    """Fixed-ladder mean extrapolation with its full cross-resolution covariance."""
    products, numerical_zero = [], []
    for q in qs:
        fit = panel_residuals(frame, "contiguous", q=q)
        products.append(fit["forecast_residual"] * fit["outcome_residual"])
        numerical_zero.append(any(
            np.max(np.abs(fit[f"{side}_residual"])) <= 1e-10 * (1 + np.max(np.abs(fit[f"{side}_rank"])))
            for side in ["forecast","outcome"]))
    products = np.column_stack(products)
    results = []
    for beta in betas:
        design = np.column_stack([np.ones(len(qs)), np.asarray(qs, float) ** (-beta)])
        weights = np.linalg.pinv(design)[0]
        np.testing.assert_allclose(weights @ design, [1.,0.], atol=1e-12)
        combined = products @ weights
        mean = float(combined.mean())
        if frequency is None:
            if lag != 0:
                raise ValueError("unordered clusters require lag zero")
            scores = pd.DataFrame({"score":combined-mean,"period":frame.period}).groupby("period", observed=False).score.sum().to_numpy()[:,None]
            gaps = 0
        else:
            scores, gaps = calendar_scores(combined-mean, frame.period, frequency)
        variance = float(bartlett_score_covariance(scores, lag)[0,0]) / len(combined) ** 2
        se = float(np.sqrt(variance))
        if not np.isfinite(se):
            raise ValueError("extrapolated score has a nonfinite scale")
        defined = not all(numerical_zero) and se > max(
            NUMERICAL_SCORE_FLOOR, 8*np.finfo(float).eps*np.max(np.abs(combined)))
        statistic = mean / se if defined else np.nan
        results.append({"fold_scheme":"contiguous", "beta":beta, "lag":lag, "n":len(frame),
                        "periods":int(frame.period.nunique()), "calendar_gaps":gaps,
                        "mean_product":mean, "standard_error":se, "statistic":statistic,
                        "p_one_sided":float(norm.sf(statistic)) if defined else 1.0,
                        "inference_status":"computed" if defined else "undefined_rank_or_scale",
                        **{f"product_mean_q{q}":float(products[:,j].mean()) for j,q in enumerate(qs)}})
    return results


def main():
    """Public entry point for standardized or electricity-format predictions."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV/parquet file or directory of model parquet files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", choices=["M","D","W-FRI","cluster"], required=True)
    parser.add_argument("--lags", required=True, help="comma-separated fixed nonnegative lags")
    parser.add_argument("--entity-column", default="entity")
    parser.add_argument("--period-column", default="period")
    parser.add_argument("--outcome-column", default="y")
    parser.add_argument("--forecast-column", default="prediction")
    parser.add_argument("--baseline-model", help="construct baseline column from this named model")
    parser.add_argument("--extrapolation-betas", help="comma-separated positive exponents; requires exactly one lag")
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.parquet")) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error("no input files")
    raw = pd.concat([pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in paths], ignore_index=True)
    raw = raw.rename(columns={args.entity_column:"entity",args.period_column:"period",
                              args.outcome_column:"y",args.forecast_column:"prediction"})
    if "model" not in raw:
        parser.error("input requires model column")
    if args.frequency != "cluster":
        raw["period"] = pd.to_datetime(raw.period)
    if args.baseline_model:
        baseline = raw.loc[raw.model == args.baseline_model,["entity","period","prediction"]].rename(columns={"prediction":"baseline"})
        raw = raw.merge(baseline,on=["entity","period"],how="left",validate="many_to_one")
    lags = [int(value) for value in args.lags.split(",")]
    if any(lag < 0 for lag in lags):
        parser.error("lags must be nonnegative")
    frequency = None if args.frequency == "cluster" else args.frequency
    betas = [float(value) for value in args.extrapolation_betas.split(",")] if args.extrapolation_betas else None
    if betas is not None and (len(lags) != 1 or any(beta <= 0 for beta in betas)):
        parser.error("positive betas and exactly one lag required for extrapolation")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    rows = []
    for model, group in raw.groupby("model",sort=True):
        if betas is not None:
            rows.extend({"model":model, **row} for row in audit_extrapolated_panel(group,lags[0],frequency,betas=betas))
        else:
            for scheme in ["contiguous","interleaved"]:
                rows.extend({"model":model, **row} for row in audit_panel(group,lags,frequency,scheme))
    result = pd.DataFrame(rows)
    keys = ["fold_scheme","lag"] + (["beta"] if betas is not None else [])
    from statsmodels.stats.multitest import multipletests
    for _, group in result.groupby(keys):
        decision, adjusted = by_adjust(group.p_one_sided)
        oracle = multipletests(group.p_one_sided,method="fdr_by",alpha=.05)
        np.testing.assert_array_equal(decision,oracle[0]); np.testing.assert_allclose(adjusted,oracle[1])
        result.loc[group.index,"by_reject"] = decision
        result.loc[group.index,"by_adjusted_p"] = adjusted
    result.to_csv(args.output_dir / "profile.csv",index=False)
    (args.output_dir / "specification.json").write_text(json.dumps({
        "family_size":int(raw.model.nunique()),"frequency":args.frequency,"lags":lags,"betas":betas,
        "normal_pvalue_is_conditional_diagnostic":True,"full_family_by_statsmodels_match":True},indent=2)+"\n")


if __name__ == "__main__":
    main()
