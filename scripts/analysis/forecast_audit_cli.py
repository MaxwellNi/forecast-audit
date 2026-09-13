"""Audit one fixed family of forecasts and retain every model in its BY screen.

Inputs are model/entity/period/y/prediction/baseline columns in a CSV file.
Outputs contain model and resolution aggregates, never individual observations.
The labels describe guarded normal-reference screens, not finite certificates.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import sys
import re
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import audit_panel_predictions as core
from audit_decisions import apply_decision_policy, inspect_control_redundancy

REQUIRED = ("model", "entity", "period", "y", "prediction", "baseline")
FREQUENCIES = ("D", "M", "W-FRI", "cluster")
MINIMUM_CLUSTERS = 5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_specification(frequency, lag, beta, ladder, alpha):
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}")
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError("lag must be a nonnegative integer")
    if frequency == "cluster" and lag != 0:
        raise ValueError("unordered clusters require lag zero")
    if isinstance(beta, (bool, np.bool_)) or not np.isscalar(beta) or not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    ladder = tuple(ladder)
    if len(ladder) < 2 or any(isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, np.integer)) or q < 2 for q in ladder):
        raise ValueError("ladder must contain at least two integer resolutions of at least two")
    if any(right <= left for left, right in zip(ladder, ladder[1:])):
        raise ValueError("ladder must be strictly increasing with no repeated resolution")
    x = np.asarray(ladder, dtype=float) ** -float(beta)
    design = np.column_stack([np.ones(len(x)), x])
    if not np.isfinite(design).all() or np.linalg.matrix_rank(design) != 2:
        raise ValueError("beta and ladder produce a numerically rank-deficient extrapolation")
    weights = np.linalg.pinv(design)[0]
    if not np.allclose(weights @ design, [1., 0.], rtol=1e-12, atol=1e-12):
        raise ValueError("extrapolation constraints cannot be resolved numerically")
    return ladder, weights


def cohort_sha256(frame):
    """Hash common target/control rows, without retaining those rows in output."""
    view = frame[["entity", "period", "y", "baseline"]].copy()
    view["entity"] = view.entity.astype(str)
    view["period"] = view.period.astype(str)
    view = view.sort_values(list(view.columns), kind="stable")
    return hashlib.sha256(view.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def ordered_cluster_identifiers(values):
    """Keep exact identifiers; numerical ordering never merges equivalent spellings."""
    labels = values.astype(str)
    unique = labels.unique().tolist()
    numeric = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
    if all(numeric.fullmatch(label) for label in unique):
        ordered = sorted(unique, key=lambda label: (Decimal(label), label))
    else:
        ordered = sorted(unique)
    return pd.Series(pd.Categorical(labels, categories=ordered, ordered=True), index=values.index)


def validate_frame(frame, frequency, lag):
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.columns.duplicated().any():
        raise ValueError("input must be a nonempty table with unique column names")
    if any(column not in frame for column in REQUIRED):
        raise ValueError(f"required columns: {REQUIRED}")
    frame = frame[list(REQUIRED)].copy()
    if frame.isna().any().any():
        raise ValueError("missing values are not allowed")
    for column in ("model", "entity", "period"):
        labels = frame[column].astype(str)
        if labels.str.strip().eq("").any() or labels.ne(labels.str.strip()).any():
            raise ValueError(f"{column} labels must be nonempty and have no outer whitespace")
        if any(isinstance(x, bool) for x in frame[column]):
            raise ValueError(f"boolean {column} labels are not supported")
        if pd.api.types.is_numeric_dtype(frame[column]) and not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"numeric {column} labels must be finite")
    frame["model"] = frame.model.astype(str)
    for column in ("y", "prediction", "baseline"):
        if any(isinstance(x, (bool, complex)) for x in frame[column]):
            raise ValueError(f"{column} must be real numeric values")
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[["y", "prediction", "baseline"]].to_numpy()).all():
        raise ValueError("outcomes, predictions and baselines must be finite")
    if frame.duplicated(["model", "entity", "period"]).any():
        raise ValueError("duplicate model/entity/period observations")
    supplied_hashes = {str(name): cohort_sha256(part) for name, part in frame.groupby("model", sort=True)}
    if len(set(supplied_hashes.values())) != 1:
        raise ValueError("all family models must use exactly the same entities, periods, outcomes and baseline")
    if frequency != "cluster":
        frame["period"] = pd.to_datetime(frame.period, errors="raise")
        if frame.period.isna().any() or getattr(frame.period.dt, "tz", None) is not None:
            raise ValueError("periods must be nonmissing timezone-free calendar labels")
        periods = pd.PeriodIndex(frame.period, freq=frequency)
        mapping = pd.DataFrame({"period": frame.period, "calendar": periods}).drop_duplicates()
        if mapping.calendar.duplicated().any():
            raise ValueError("distinct period labels map to one declared calendar period")
        calendar_length = int(periods.asi8.max() - periods.asi8.min() + 1)
        if lag >= calendar_length:
            raise ValueError("lag must be smaller than the complete calendar length")
    elif lag != 0:
        raise ValueError("unordered clusters require lag zero")
    else:
        frame["period"] = ordered_cluster_identifiers(frame.period)
    if frame.duplicated(["model", "entity", "period"]).any():
        raise ValueError("period parsing produced duplicate observations")
    counts = frame.groupby("model").period.nunique()
    if (counts < MINIMUM_CLUSTERS).any():
        raise ValueError("at least five observed periods are required for five whole-period folds")
    first = frame[frame.model == sorted(supplied_hashes)[0]]
    core.period_folds(first.period, scheme="contiguous", n_folds=5)
    return frame, supplied_hashes


def by_trace(profile, alpha):
    """Expose the complete sorted step-up calculation for both p-value vectors."""
    rows = []
    m = len(profile)
    harmonic = float(np.sum(1. / np.arange(1, m + 1)))
    for kind, column, decision_column in (
        ("raw", "raw_p_one_sided", "raw_by_reject_recomputed"),
        ("guarded", "p_policy", "policy_by_reject"),
    ):
        ordered = profile.sort_values([column, "model"], kind="stable").reset_index(drop=True)
        thresholds = alpha * np.arange(1, m + 1) / (m * harmonic)
        passing = np.flatnonzero(ordered[column].to_numpy() <= thresholds)
        k = int(passing[-1] + 1) if len(passing) else 0
        boundary = float(thresholds[k - 1]) if k else 0.
        decisions = ordered[column].to_numpy() <= boundary if k else np.zeros(m, dtype=bool)
        if not np.array_equal(decisions, ordered[decision_column].to_numpy(bool)):
            raise AssertionError("explicit ordered BY trace disagrees with the maintained decision routine")
        for i, row in ordered.iterrows():
            rows.append({"screen": kind, "model": row.model, "ordered_rank": i + 1,
                         "family_size": m, "harmonic_factor": harmonic,
                         "pvalue": row[column], "rank_cutoff": thresholds[i],
                         "passes_own_rank_cutoff": bool(row[column] <= thresholds[i]),
                         "step_up_last_passing_rank": k, "step_up_cutoff": boundary,
                         "retained_by_step_up": bool(decisions[i]),
                         "final_label": row.final_label})
    return pd.DataFrame(rows)


def audit_frame(frame, *, frequency, lag, beta, ladder, alpha=.05):
    ladder, weights = validate_specification(frequency, lag, beta, ladder, alpha)
    frame, supplied_hashes = validate_frame(frame, frequency, lag)
    rows, evidence, traces, cohorts = [], {}, [], []
    for name, part in frame.groupby("model", sort=True):
        part = part.reset_index(drop=True)
        evidence[name] = inspect_control_redundancy(part.prediction, part.baseline, part.period)
        result = core.audit_extrapolated_panel(part, lag, None if frequency == "cluster" else frequency,
                                               qs=ladder, betas=(beta,))[0]
        rows.append({"model": name, **result})
        means = np.array([result[f"product_mean_q{q}"] for q in ladder])
        np.testing.assert_allclose(means @ weights, result["mean_product"], rtol=1e-10, atol=1e-12)
        for q, weight, mean in zip(ladder, weights, means):
            traces.append({"model": name, "resolution_q": q, "beta": beta, "weight": weight,
                           "mean_residual_product": mean, "weighted_mean_contribution": weight * mean,
                           "combined_mean": result["mean_product"], "standard_error": result["standard_error"],
                           "statistic": result["statistic"], "normal_p_one_sided": result["p_one_sided"]})
        folds = core.period_folds(part.period)
        fold_counts = pd.DataFrame({"period": part.period, "fold": folds}).drop_duplicates().groupby("fold").size()
        cohorts.append({"model": name, "rows": len(part), "observed_periods": part.period.nunique(),
                        "supplied_cohort_sha256": supplied_hashes[name],
                        "parsed_cohort_sha256": cohort_sha256(part),
                        "ordered_fold_sha256": hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest(),
                        **{f"fold_{k}_periods": int(fold_counts.loc[k]) for k in range(5)}})
    family = sorted(evidence)
    profile = apply_decision_policy(pd.DataFrame(rows), evidence, family_models=family,
                                    specification_columns=["fold_scheme", "lag", "beta"], alpha=alpha)
    profile["final_label"] = np.where(profile.policy_abstain, "ABSTAIN",
                                       np.where(profile.policy_by_reject, "RETAIN", "NOT_RETAINED"))
    profile["label_scope"] = "operational normal-reference screen; calibration not established"
    profile["finite_certificate_issued"] = False
    profile["target"] = "additive-control residual association of within-period ranks; fixed resolution contrast"
    profile["control_representation"] = "additive entity effects and declared baseline rank bins; not full joint conditioning"
    profile["short_panel_empirical_concern"] = profile.periods <= 50
    profile["guard_reason"] = np.where(profile.redundancy_abstain, profile.redundancy_reason,
                                        np.where(profile.undefined_audit_abstain, "undefined_rank_or_scale", "none"))
    ordered = by_trace(profile, alpha)
    redundancy = pd.DataFrame([{"model": name, **evidence[name].to_dict()} for name in family])
    return {"profile": profile, "resolution_trace": pd.DataFrame(traces), "by_trace": ordered,
            "guard_evidence": redundancy, "cohorts": pd.DataFrame(cohorts)}


def write_outputs(tables, output, *, frequency, lag, beta, ladder, alpha=.05, input_hash=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.17g", na_rep="")
    sources = [Path(__file__), Path(core.__file__), Path(__file__).with_name("audit_decisions.py"),
               Path(__file__).with_name("cluster_covariance_reference.py")]
    profile = tables["profile"]
    receipt = {
        "status": "COMPLETED_DIAGNOSTIC_SCREEN", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hash,
        "family_size": len(profile), "models": profile.model.tolist(),
        "specification": {"frequency": frequency, "lag": lag, "beta": beta, "ladder": list(ladder),
                          "alpha": alpha, "folds": 5, "fold_scheme": "contiguous whole observed periods",
                          "minimum_clusters": MINIMUM_CLUSTERS},
        "counts": profile.final_label.value_counts().to_dict(),
        "finite_certificate_issued": False,
        "target": "additive-control residual association of within-period ranks; fixed resolution contrast",
        "control_representation": "additive entity effects and declared baseline rank bins; not full joint conditioning",
        "guarantee_scope": "The five-period minimum allows the five-fold calculation; it does not establish inference validity. The M<=50 flag records an empirical concern, not a validity threshold. Larger panels also require justified sampling, dependence and nuisance conditions. Observed-copy/order guards and full-family BY do not calibrate the underlying normal-reference values.",
        "selection": "The CLI uses exactly the supplied exponent, ladder, lag and family. It performs no outcome-based choice or tuning.",
        "fold_scope": "Complete observed periods stay in a single fold. Training may use periods later than an evaluation period; this is not a forward-only forecast-training procedure. Forecasts must already respect the intended evaluation protocol.",
        "guard_scope": "Exact observed copy, constant within-period ranks, same or reversed weak ordering, and numerical undefined-scale rules. Observed agreement does not prove population measurability. Undefined statistics are blank, with guarded p=1 and ABSTAIN.",
        "output_scope": "Model and resolution aggregates only; individual outcomes, forecasts, residuals and entity/period labels are not written. Aggregate output may still require publication permission.",
        "sources": {path.name: sha256(path) for path in sources},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")},
        "files": {path.name: sha256(path) for path in sorted(output.glob("*.csv"))},
    }
    with (output / "receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


def read_input(path, mapping):
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("column mappings must be distinct")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        for record, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV record {record} does not have exactly {len(header)} fields")
    if not header or len(set(header)) != len(header):
        raise ValueError("CSV header must be nonempty and unique")
    if any(value not in header for value in mapping.values()):
        raise ValueError("mapped input column is missing")
    # Identifiers are strings before pandas sees them. In particular, 01, 1,
    # 1.0 and 1e0 are distinct labels, and NA/NULL are literal identifiers.
    # Empty identifiers are rejected by validate_frame; numeric NA is invalid.
    identifiers = {mapping[name]: str for name in ("model", "entity", "period")}
    frame = pd.read_csv(path, dtype=identifiers, keep_default_na=False,
                        na_filter=False, float_precision="round_trip")
    return frame[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV or CSV.gz containing the complete fixed model family")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for aggregate outputs")
    parser.add_argument("--frequency", choices=FREQUENCIES, required=True)
    parser.add_argument("--lag", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--ladder", required=True, help="strictly increasing comma-separated integer resolutions")
    parser.add_argument("--alpha", type=float, default=.05)
    for column in REQUIRED:
        parser.add_argument(f"--{column}-column", default=column)
    args = parser.parse_args(argv)
    try:
        ladder = tuple(int(q) for q in args.ladder.split(","))
        validate_specification(args.frequency, args.lag, args.beta, ladder, args.alpha)
        if args.output_dir.exists():
            raise ValueError("output directory already exists; refusing to overwrite")
        mapping = {name: getattr(args, f"{name}_column") for name in REQUIRED}
        frame = read_input(args.input, mapping)
        tables = audit_frame(frame, frequency=args.frequency, lag=args.lag, beta=args.beta,
                             ladder=ladder, alpha=args.alpha)
        receipt = write_outputs(tables, args.output_dir, frequency=args.frequency, lag=args.lag,
                                beta=args.beta, ladder=ladder, alpha=args.alpha, input_hash=sha256(args.input))
    except (ValueError, TypeError, KeyError, AssertionError, RuntimeError, OSError, csv.Error) as error:
        parser.exit(2, f"Audit not issued: {error}\n")
    print(json.dumps({"status": receipt["status"], "models": receipt["family_size"],
                      "counts": receipt["counts"], "finite_certificate_issued": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
