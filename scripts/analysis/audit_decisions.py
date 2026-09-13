"""Conservative decisions layered over unchanged normal-reference diagnostics.

This module detects observed prediction/control redundancy without outcomes.
Observed equality or order agreement does not prove population measurability.
The policy only replaces p-values by one, retains every family member, and
recomputes BY. It cannot make an uncalibrated diagnostic into a valid p-value.
No approximate-collinearity or fitted-overlap threshold is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

from audit_panel_predictions import by_adjust


@dataclass(frozen=True)
class RedundancyEvidence:
    row_count: int
    cluster_count: int
    singleton_clusters: int
    minimum_cluster_size: int
    maximum_cluster_size: int
    nonconstant_forecast_clusters: int
    nonconstant_baseline_clusters: int
    same_weak_order_clusters: int
    reversed_weak_order_clusters: int
    exact_baseline_copy: bool
    zero_forecast_rank_variation: bool
    same_weak_order: bool
    reversed_weak_order: bool
    abstain: bool
    reason: str

    def to_dict(self):
        return asdict(self)


def _finite_binary64_vector(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty real numeric vector")
    # The underlying auditor likewise interprets these numeric inputs as
    # binary64. Equality below is exact in that declared representation.
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite binary64 values")
    return array


def _twice_average_ranks(values):
    """Exact integer twice-midranks, with ties compared without a tolerance."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    start = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stop = np.r_[start[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.repeat(start + stop + 1, stop - start)
    return ranks


def inspect_control_redundancy(prediction, baseline, clusters):
    """Inspect exact sample redundancy using no outcome or fitted nuisance.

    The rule abstains for an exact raw baseline copy, zero forecast variation
    within every cluster, or the same/reversed weak ordering (including the
    entire tie partition) in every cluster. One common orientation must work
    across all clusters. Singleton clusters satisfy both order comparisons;
    other clusters must still satisfy the rule. All-singleton inputs abstain.

    A nonlinear increasing/decreasing baseline transform can meet this rule.
    Agreement seen only on the evaluated sample is not an out-of-sample law.
    The function deliberately does not infer an arbitrary control-to-forecast
    map: distinct observed controls would make that criterion vacuous.
    """
    forecast = _finite_binary64_vector(prediction, "prediction")
    control = _finite_binary64_vector(baseline, "baseline")
    labels = np.asarray(clusters)
    if labels.ndim != 1 or len(labels) != len(forecast) or len(control) != len(forecast):
        raise ValueError("prediction, baseline and cluster vectors must align")
    if pd.isna(labels).any():
        raise ValueError("cluster labels must be nonmissing")
    groups = pd.Series(np.arange(len(labels))).groupby(labels, sort=False).indices
    sizes, same, reverse, nonconstant_f, nonconstant_b = [], 0, 0, 0, 0
    for index in groups.values():
        size = len(index)
        fr = _twice_average_ranks(forecast[index])
        br = _twice_average_ranks(control[index])
        sizes.append(size)
        same += int(np.array_equal(fr, br))
        reverse += int(np.array_equal(fr + br, np.full(size, 2 * (size + 1))))
        nonconstant_f += int(np.any(fr != size + 1))
        nonconstant_b += int(np.any(br != size + 1))
    count = len(sizes)
    raw_copy = bool(np.array_equal(forecast, control))
    zero = nonconstant_f == 0
    same_all, reverse_all = same == count, reverse == count
    if raw_copy:
        reason = "exact_observed_baseline_copy"
    elif zero:
        reason = "zero_within_cluster_forecast_rank_variation"
    elif same_all:
        reason = "same_observed_baseline_weak_order"
    elif reverse_all:
        reason = "reversed_observed_baseline_weak_order"
    else:
        reason = "no_exact_sample_redundancy_detected"
    return RedundancyEvidence(
        len(forecast), count, int(np.sum(np.asarray(sizes) == 1)),
        min(sizes), max(sizes), nonconstant_f, nonconstant_b, same, reverse,
        raw_copy, zero, same_all, reverse_all, bool(raw_copy or zero or same_all or reverse_all), reason,
    )


def apply_decision_policy(profile, evidence, *, family_models, specification_columns,
                          alpha=0.05):
    """Retain raw diagnostics and apply abstention within each complete family.

    If the original p-values are super-uniform, replacing selected ones by one
    preserves that property even for a data-dependent selection event. BY then
    has its usual guarantee on the unchanged full family. When the original
    normal-reference values are uncalibrated, these remain nominal decisions.
    Rejection-set containment alone is not a proof of FDR control.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    family = list(family_models)
    if not family or len(set(family)) != len(family) or set(evidence) != set(family):
        raise ValueError("a complete unique family and aligned evidence are required")
    keys = list(specification_columns)
    required = ["model", "p_one_sided", "statistic", "inference_status"] + keys
    if any(column not in profile for column in required) or profile.empty:
        raise ValueError("profile lacks required diagnostic/specification fields")
    if profile[["model"] + keys].isna().any().any():
        raise ValueError("model and specification keys must be nonmissing")
    if profile.duplicated(["model"] + keys).any():
        raise ValueError("duplicate model/specification rows")
    out = profile.copy().reset_index(drop=True)
    raw_p = out.p_one_sided.to_numpy(float)
    if not np.isfinite(raw_p).all() or np.any((raw_p < 0) | (raw_p > 1)):
        raise ValueError("raw p-values must be finite and in [0, 1]")
    if not out.inference_status.isin(["computed", "undefined_rank_or_scale"]).all():
        raise ValueError("unknown raw inference status")
    computed = out.inference_status.eq("computed").to_numpy()
    if not np.isfinite(out.loc[computed, "statistic"].to_numpy(float)).all():
        raise ValueError("computed diagnostics require finite statistics")
    redundancy = np.asarray([evidence[name].abstain for name in out.model], bool)
    reasons = [evidence[name].reason for name in out.model]
    undefined = ~computed
    abstain = redundancy | undefined
    out["raw_p_one_sided"] = raw_p
    out["raw_statistic"] = out.statistic
    out["raw_nominal_positive"] = raw_p < alpha
    out["redundancy_abstain"] = redundancy
    out["redundancy_reason"] = reasons
    out["undefined_audit_abstain"] = undefined
    out["policy_abstain"] = abstain
    out["p_policy"] = np.where(abstain, 1., raw_p)
    out["policy_nominal_positive"] = out.p_policy < alpha
    out["policy_decision"] = np.where(
        redundancy, "ABSTAIN_OBSERVED_CONTROL_REDUNDANCY",
        np.where(undefined, "ABSTAIN_UNDEFINED_AUDIT",
                 np.where(out.policy_nominal_positive, "NOMINAL_POSITIVE", "NOT_POSITIVE")),
    )
    grouped = out.groupby(keys, sort=False, dropna=False) if keys else [((), out)]
    for _, group in grouped:
        if set(group.model) != set(family) or len(group) != len(family):
            raise ValueError("every specification must retain the complete fixed family")
        raw_by, raw_adjusted = by_adjust(group.p_one_sided, alpha=alpha)
        policy_by, policy_adjusted = by_adjust(group.p_policy, alpha=alpha)
        if np.any(policy_by & ~raw_by):
            raise AssertionError("abstention expanded the BY rejection set")
        out.loc[group.index, "raw_by_reject_recomputed"] = raw_by
        out.loc[group.index, "raw_by_adjusted_p_recomputed"] = raw_adjusted
        out.loc[group.index, "policy_by_reject"] = policy_by
        out.loc[group.index, "policy_by_adjusted_p"] = policy_adjusted
        out.loc[group.index, "policy_family_size"] = len(family)
    for name in ["raw_by_reject_recomputed", "policy_by_reject"]:
        out[name] = out[name].astype(bool)
    out["policy_family_size"] = out.policy_family_size.astype(int)
    return out
