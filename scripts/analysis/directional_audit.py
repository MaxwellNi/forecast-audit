"""Compare nuisance-training directions on one complete forecast family.

Only model aggregates are exported. Baseline bins and the original full-cohort
abstention policy stay fixed. Calendar direction describes an offline audit;
ordering unordered cluster identifiers does not create a temporal guarantee.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import audit_panel_predictions as core
import forecast_audit_cli as cli


def additive_prediction(values, entities, bins, training, evaluation, *, use_entities=True):
    """Fit additive entity/bin means with centred effects and an unseen-level fallback.

    The bin assignments are supplied and held fixed. Missing entity effects
    use the training-weighted average; absent bins use the intercept. When
    use_entities is false, prediction is by pooled training-bin means.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    entities = np.asarray(entities, dtype=int)
    bins = np.asarray(bins, dtype=int)
    training = np.asarray(training, dtype=bool)
    evaluation = np.asarray(evaluation, dtype=bool)
    n = len(values)
    if values.ndim != 2 or any(len(x) != n for x in (entities, bins, training, evaluation)):
        raise ValueError("inconsistent input lengths")
    if not training.any() or np.any(training & evaluation):
        raise ValueError("training must be nonempty and disjoint from evaluation")
    if not np.isfinite(values).all() or (entities < 0).any() or (bins < 0).any():
        raise ValueError("values and group codes must be finite and valid")
    q = int(bins.max()) + 1
    b, y = bins[training], values[training]
    bin_count = np.bincount(b, minlength=q)
    bin_sum = np.zeros((q, values.shape[1]))
    np.add.at(bin_sum, b, y)
    if not use_entities:
        fitted = np.tile(y.mean(axis=0), (q, 1))
        fitted[bin_count > 0] = bin_sum[bin_count > 0] / bin_count[bin_count > 0, None]
        return fitted[bins[evaluation]]
    ne = int(entities.max()) + 1
    f = entities[training]
    counts = np.bincount(f, minlength=ne)
    seen = counts > 0
    cell_counts = np.zeros((ne, q))
    np.add.at(cell_counts, (f, b), 1)
    entity_means = np.zeros((ne, values.shape[1]))
    np.add.at(entity_means, f, y)
    entity_means[seen] /= counts[seen, None]
    frequencies = np.zeros_like(cell_counts)
    frequencies[seen] = cell_counts[seen] / counts[seen, None]
    normal = np.diag(bin_count) - cell_counts.T @ frequencies
    rhs = bin_sum - cell_counts.T @ entity_means
    bin_effects = np.linalg.lstsq(normal, rhs, rcond=1e-12)[0]
    entity_effects = np.zeros_like(entity_means)
    entity_effects[seen] = entity_means[seen] - frequencies[seen] @ bin_effects
    entity_center = counts @ entity_effects / len(y)
    bin_center = bin_count @ bin_effects / len(y)
    entity_effects[seen] -= entity_center
    bin_effects[bin_count > 0] -= bin_center
    bin_effects[bin_count == 0] = 0
    return entity_center + bin_center + entity_effects[entities[evaluation]] + bin_effects[bins[evaluation]]


def directional_residuals(values, entities, bins, folds, side):
    """Use earlier folds for forecasts and later folds for outcomes.

    A globally empty side falls back to pooled-bin fitting on complementary
    folds. This is a reported boundary convention, not a time-order guarantee.
    """
    if side not in ("forecast", "outcome"):
        raise ValueError("side must be forecast or outcome")
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    folds = np.asarray(folds, int)
    entities = np.asarray(entities, int)
    if len(folds) != len(values) or not np.array_equal(np.unique(folds), np.arange(5)):
        raise ValueError("five nonempty folds labelled 0 through 4 are required")
    result = np.empty_like(values)
    fallback = np.zeros(len(values), dtype=bool)
    unseen = np.zeros(len(values), dtype=bool)
    for fold in range(5):
        evaluation = folds == fold
        training = folds < fold if side == "forecast" else folds > fold
        if training.any():
            result[evaluation] = values[evaluation] - additive_prediction(
                values, entities, bins, training, evaluation)
            unseen[evaluation] = ~np.isin(entities[evaluation], entities[training])
        else:
            fallback[evaluation] = True
            result[evaluation] = values[evaluation] - additive_prediction(
                values, entities, bins, folds != fold, evaluation, use_entities=False)
    return result, fallback, unseen


def score_summary(values, periods, frequency, lag):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("score values must be finite and nonempty")
    mean = float(values.mean())
    if frequency == "cluster":
        sums = pd.DataFrame({"score": values - mean, "cluster": np.asarray(periods)}).groupby(
            "cluster", observed=True, sort=True).score.sum().to_numpy()[:, None]
        gaps = 0
    else:
        sums, gaps = core.calendar_scores(values - mean, periods, frequency)
    if lag >= len(sums):
        raise ValueError("lag must be smaller than every reported evaluation support")
    se = float(np.sqrt(core.bartlett_score_covariance(sums, lag)[0, 0]) / len(values))
    raw_statistic = mean / se if se > 0 else float("nan")
    standard_error_abstain = not np.isfinite(se) or se <= core.NUMERICAL_SCORE_FLOOR
    statistic = float("nan") if standard_error_abstain else raw_statistic
    return {"mean_product": mean, "standard_error": se, "statistic": statistic,
            "raw_statistic": raw_statistic, "standard_error_abstain": standard_error_abstain,
            "calendar_gaps": gaps, "observed_groups": int(pd.Series(np.asarray(periods)).nunique())}


def audit_directional_frame(frame, *, frequency, lag, beta, ladder, alpha=.05):
    ladder, weights = cli.validate_specification(frequency, lag, beta, ladder, alpha)
    frame, input_cohorts = cli.validate_frame(frame, frequency, lag)
    original = cli.audit_frame(frame.copy(), frequency=frequency, lag=lag, beta=beta,
                               ladder=ladder, alpha=alpha)["profile"].set_index("model")
    rows = []
    for name, part in frame.groupby("model", sort=True, observed=True):
        part = part.reset_index(drop=True)
        entities = pd.factorize(part.entity.astype(str), sort=True)[0]
        products, standard = [], []
        for q in ladder:
            fit = core.panel_residuals(part, "contiguous", q=q)
            folds = np.asarray(fit["folds"], dtype=int)
            bins = np.asarray(fit["baseline_bins"], dtype=int)
            xr, xf, xu = directional_residuals(fit["forecast_rank"], entities, bins, folds, "forecast")
            yr, yf, yu = directional_residuals(fit["outcome_rank"], entities, bins, folds, "outcome")
            products.append(xr[:, 0] * yr[:, 0])
            standard.append(fit["forecast_residual"] * fit["outcome_residual"])
        directional = np.column_stack(products) @ weights
        complementary = np.column_stack(standard) @ weights
        middle = np.isin(folds, (1, 2, 3))
        configurations = (("complementary_all", complementary, np.ones(len(part), bool)),
                          ("complementary_middle", complementary, middle),
                          ("directional_middle", directional, middle),
                          ("directional_all", directional, np.ones(len(part), bool)))
        for configuration, values, mask in configurations:
            selected_periods = part.loc[mask, "period"].reset_index(drop=True)
            summary = score_summary(values[mask], selected_periods, frequency, lag)
            directional_fit = configuration.startswith("directional")
            rows.append({"model": name, "configuration": configuration, "evaluation_rows": int(mask.sum()),
                         "family_size": len(original), "beta": beta, "lag": lag, "frequency": frequency,
                         "calendar_direction": frequency != "cluster", **summary,
                         "original_policy_abstain": bool(original.loc[name, "policy_abstain"]),
                         "original_guard_reason": original.loc[name, "guard_reason"],
                         "forecast_pooled_fallback_rows": int((xf & mask).sum()) if directional_fit else 0,
                         "outcome_pooled_fallback_rows": int((yf & mask).sum()) if directional_fit else 0,
                         "forecast_unseen_entity_rows": int((xu & mask).sum()) if directional_fit else 0,
                         "outcome_unseen_entity_rows": int((yu & mask).sum()) if directional_fit else 0,
                         "cohort_sha256": input_cohorts[name]})
    profile = pd.DataFrame(rows)
    for configuration, group in profile.groupby("configuration", sort=False):
        invalid = ~np.isfinite(group.statistic.to_numpy(float))
        numerical = group.standard_error_abstain.to_numpy(bool)
        guarded = group.original_policy_abstain.to_numpy(bool) | numerical | invalid
        profile.loc[group.index, "guard_reason"] = np.where(
            group.original_policy_abstain.to_numpy(bool), group.original_guard_reason.to_numpy(),
            np.where(numerical, "standard_error_at_or_below_numerical_floor",
                     np.where(invalid, "undefined_statistic", "no_guard")))
        p = np.where(guarded, 1., norm.sf(group.statistic.to_numpy(float)))
        _, adjusted = core.by_adjust(p, alpha=alpha)
        profile.loc[group.index, "normal_p_one_sided"] = norm.sf(group.statistic.to_numpy(float))
        profile.loc[group.index, "guarded_p"] = p
        profile.loc[group.index, "BY_adjusted_p"] = adjusted
        profile.loc[group.index, "final_label"] = np.where(guarded, "ABSTAIN", np.where(
            adjusted <= alpha, "RETAIN", "NOT_RETAINED"))
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency", choices=cli.FREQUENCIES, required=True)
    parser.add_argument("--lag", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--ladder", required=True)
    parser.add_argument("--alpha", type=float, default=.05)
    for column in cli.REQUIRED:
        parser.add_argument(f"--{column}-column", default=column)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("output directory already exists; refusing to overwrite")
        ladder = tuple(int(q) for q in args.ladder.split(","))
        mapping = {name: getattr(args, f"{name}_column") for name in cli.REQUIRED}
        frame = cli.read_input(args.input, mapping)
        profile = audit_directional_frame(frame, frequency=args.frequency, lag=args.lag,
                                          beta=args.beta, ladder=ladder, alpha=args.alpha)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        destination = args.output_dir / "profile.csv"
        profile.to_csv(destination, index=False, float_format="%.17g", na_rep="")
        sources = [Path(__file__), Path(core.__file__), Path(cli.__file__),
                   Path(__file__).with_name("audit_decisions.py"),
                   Path(__file__).with_name("cluster_covariance_reference.py")]
        receipt = {"status": "COMPLETED_DIRECTIONAL_SENSITIVITY", "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "input_sha256": cli.sha256(args.input), "family_size": int(profile.model.nunique()),
                   "settings": {"frequency": args.frequency, "lag": args.lag, "beta": args.beta,
                                "ladder": list(ladder), "alpha": args.alpha},
                   "counts": {name: group.final_label.value_counts().to_dict() for name, group in profile.groupby("configuration")},
                   "scope": "Diagnostic comparison, not a calibrated test or a historically available forecasting strategy.",
                   "bin_scope": "Original within-group baseline-rank bins are held fixed; the baseline control channel is not refitted for time direction.",
                   "guard_scope": "The complementary-fold full-cohort abstention policy is held fixed; new undefined statistics and standard errors at or below 1e-10 additionally abstain. Raw statistics remain diagnostic only. Every model remains in every family.",
                   "ordering_scope": "Calendar periods are ordered in time; frequency=cluster orders identifiers and does not imply time.",
                   "fallback_scope": "An empty training side uses pooled bin means on complementary folds. An unseen entity receives the training-weighted mean entity effect. These fallbacks are outside the exact centering guarantee.",
                   "comparison_scope": "Middle-fold and all-fold results have different evaluation supports; complementary-fold comparisons on both supports are supplied.",
                   "sources": {p.name: cli.sha256(p) for p in sources},
                   "files": {destination.name: cli.sha256(destination)}}
        (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    except (ValueError, TypeError, KeyError, AssertionError, RuntimeError, OSError, csv.Error) as error:
        parser.exit(2, f"Directional comparison not issued: {error}\n")
    print(json.dumps({"status": receipt["status"], "family_size": receipt["family_size"], "counts": receipt["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
