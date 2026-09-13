"""Descriptive ranking quality on all frozen public audit cohorts; no fitting.

Supply --input-specification and --output for both stages: freeze, then run.
Use --help for the command-line arguments.
Outputs are write-once. The metric is specified before calculation on existing
audit cohorts; this retrospective comparison is not a prospective preregistration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

SCRIPT = Path(__file__).resolve()
OUTPUT = None
PACKAGE = SCRIPT.parents[2]
sys.path.insert(0, str(PACKAGE / "scripts/analysis"))
import spline_panel_comparison as spline_source
from spline_panel_comparison import domains

PROFILE = PACKAGE / "results/spline_panel/public_all_models.csv"
SOURCE = PACKAGE / "scripts/analysis/spline_panel_comparison.py"
EXPECTED = {"electricity": (11, 34992, 729), "ratings": (10, 12088, 279),
            "retail": (10, 112000, 28), "portfolios": (10, 27272, 132)}
UNITS = {"electricity": "daily mean in original UCI load units",
         "ratings": "rating points", "retail": "weekly sales units",
         "portfolios": "percentage points of return"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def csv_new(frame, name):
    frame.to_csv(OUTPUT / name, index=False, mode="x", float_format="%.17g")


def public_cluster_labels(values):
    ordered = sorted(set(str(v) for v in values), key=lambda value: float(value))
    return {value: f'group_{i+1:03d}' for i, value in enumerate(ordered)}


def frame_for(paths):
    return pd.concat([pd.read_parquet(p) if p.suffix == ".parquet" else
                      pd.read_csv(p, float_precision="round_trip") for p in paths],
                     ignore_index=True).rename(columns={"date": "period", "meter": "entity"})


def freeze():
    inputs = []
    for domain, lag, frequency, paths, old in domains():
        frame = frame_for(paths)
        rating_labels = public_cluster_labels(frame['period'].unique()) if domain == 'ratings' else {}
        roster = sorted(frame.model.unique().tolist())
        assert len(roster) == EXPECTED[domain][0]
        inputs.append({"domain": domain, "lag": lag, "frequency": frequency,
                       "models": roster, "files": [{"path": p.name,
                       "sha256": sha(p)} for p in paths]})
    json_new(OUTPUT / "protocol.json", {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "status": "Retrospective descriptive comparison; the metric was specified before its calculation, after the existing audit results were known.",
        "metric": "Within each actual audit cluster, Spearman(prediction,y), using average ranks and the unchanged admissible audit rows. Primary summary is the equally weighted arithmetic mean over defined cluster correlations. No pooled ranking or row weighting of correlations.",
        "undefined": "Fewer than two rows, constant prediction, or constant outcome makes cluster correlation undefined. Keep all model rows and report undefined group counts by reason; never replace undefined correlations by zero. A model with no defined groups has undefined mean.",
        "secondary": "Pooled-row original-scale MAE for finite scores that predict the original target. Exclude ratings SVD interaction because its interaction-only score omits the rating intercept. Baseline MAE and correlation are reported on each model's identical cohort. MAE units are domain-specific and not cross-domain comparable.",
        "scope": "All 41 models, no refits, tuning, fresh downloads, row exclusions, private cases or model selection. Metrics are descriptive of these held-out cohorts and do not establish generalization, incremental information or conditional independence. Extrapolated beta=2 normal-reference audit labels remain operational screens, not validated certificates.",
        "verification": "Independent hand-built average ranks and centered inner products versus scipy rankdata/spearmanr for every cluster; all 41 cohort rows and model counts versus existing audit output; math.erfc normal tails versus scipy norm.sf and saved values; independent full-family BY versus statsmodels and saved values.",
        "script_sha256": sha(SCRIPT), "domain_source_sha256": sha(SOURCE),
        "audit_profile_sha256": sha(PROFILE), "inputs": inputs,
        "dependencies": {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scipy", "statsmodels"]}})
    print("FROZEN: protocol.json; no new quality metrics calculated.", flush=True)


def manual_ranks(x):
    """Independent midranks from sorting and exact tie-run boundaries."""
    order = np.argsort(x, kind="stable")
    ordered = x[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], len(x)]
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.repeat((starts + stops + 1) / 2., stops - starts)
    return ranks


def correlation(rx, ry):
    a, b = rx - rx.mean(), ry - ry.mean()
    scale = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / scale if scale else np.nan


def cohort_hash(frame):
    canonical = frame[["entity", "period", "y", "baseline"]].copy()
    canonical["entity"] = canonical.entity.astype(str)
    canonical["period"] = canonical.period.astype(str)
    canonical = canonical.sort_values(list(canonical.columns), kind="stable")
    return hashlib.sha256(canonical.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def reason(x, y):
    if len(x) < 2:
        return "fewer_than_two_rows"
    cx, cy = np.all(x == x[0]), np.all(y == y[0])
    if cx and cy:
        return "constant_prediction_and_outcome"
    if cx:
        return "constant_prediction"
    if cy:
        return "constant_outcome"
    return "defined"


def by_manual(p):
    order = np.argsort(p, kind="stable")
    m = len(p)
    harmonic = sum(1. / i for i in range(1, m + 1))
    sorted_adjusted = np.minimum.accumulate((p[order] * m * harmonic / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1., sorted_adjusted)
    return adjusted


def run():
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    assert protocol["script_sha256"] == sha(SCRIPT)
    assert protocol["domain_source_sha256"] == sha(SOURCE)
    assert protocol["audit_profile_sha256"] == sha(PROFILE)
    if (OUTPUT / "receipt.json").exists():
        raise ValueError("Completed output cannot be overwritten")
    start = time.monotonic()
    profile = pd.read_csv(PROFILE, float_precision="round_trip")
    profile = profile.loc[profile.method.eq("extrapolated_beta_2")].copy()
    assert len(profile) == 41 and not profile.duplicated(["domain", "model"]).any()
    models, clusters, checks, family_checks = [], [], [], []
    max_rank_difference, max_rho_difference, max_tail_difference = 0., 0., 0.
    for frozen, (domain, lag, frequency, paths, old) in zip(protocol["inputs"], domains()):
        assert frozen["domain"] == domain
        assert frozen["files"] == [{"path": p.name, "sha256": sha(p)} for p in paths]
        frame = frame_for(paths)
        rating_labels = public_cluster_labels(frame['period'].unique()) if domain == 'ratings' else {}
        assert sorted(frame.model.unique().tolist()) == frozen["models"]
        family = profile.loc[profile.domain.eq(domain)].copy()
        assert set(frame.model) == set(family.model)
        reference_hash = None
        for model, part in frame.groupby("model", sort=True):
            audit = family.loc[family.model.eq(model)].iloc[0]
            nmodels, expected_n, expected_groups = EXPECTED[domain]
            assert len(part) == int(audit.n) == expected_n
            assert part.period.nunique() == int(audit.periods) == expected_groups
            assert np.isfinite(part[["prediction", "y", "baseline"]].to_numpy(float)).all()
            assert not part[["entity", "period"]].isna().any().any()
            chash = cohort_hash(part)
            if reference_hash is None:
                reference_hash = chash
            assert chash == reference_hash, (domain, model, "cohort mismatch")
            rows = []
            for cluster, group in part.groupby("period", sort=True):
                pred, y, baseline = (group[k].to_numpy(float) for k in ["prediction", "y", "baseline"])
                rp, ry, rb = [manual_ranks(v) for v in [pred, y, baseline]]
                for raw, ranks in zip([pred, y, baseline], [rp, ry, rb]):
                    delta = float(np.max(np.abs(rankdata(raw, method="average") - ranks)))
                    max_rank_difference = max(max_rank_difference, delta)
                    assert delta == 0.
                r, br = reason(pred, y), reason(baseline, y)
                rho = correlation(rp, ry) if r == "defined" else np.nan
                base_rho = correlation(rb, ry) if br == "defined" else np.nan
                scipy_rho = float(spearmanr(pred, y).statistic) if r == "defined" else np.nan
                scipy_base = float(spearmanr(baseline, y).statistic) if br == "defined" else np.nan
                for a, b in [(rho, scipy_rho), (base_rho, scipy_base)]:
                    if np.isfinite(a):
                        max_rho_difference = max(max_rho_difference, abs(a-b))
                        np.testing.assert_allclose(a, b, atol=1e-14, rtol=1e-13)
                rows.append({"domain": domain, "model": model, "cluster": rating_labels[str(cluster)] if domain == "ratings" else str(cluster), "n": len(group),
                             "spearman_manual": rho, "spearman_scipy": scipy_rho, "status": r,
                             "baseline_spearman_manual": base_rho, "baseline_spearman_scipy": scipy_base,
                             "baseline_status": br})
            group_table = pd.DataFrame(rows)
            clusters.extend(rows)
            defined = group_table.status.eq("defined")
            bdefined = group_table.baseline_status.eq("defined")
            paired = defined & bdefined
            statistic = float(audit.raw_statistic)
            raw_p = .5 * math.erfc(statistic / math.sqrt(2)) if np.isfinite(statistic) else 1.
            if np.isfinite(statistic):
                np.testing.assert_allclose(raw_p, norm.sf(statistic), atol=1e-300, rtol=1e-12)
            max_tail_difference = max(max_tail_difference, abs(raw_p - float(audit.raw_p_one_sided)))
            np.testing.assert_allclose(raw_p, audit.raw_p_one_sided, atol=1e-300, rtol=2e-12)
            mae_allowed = not (domain == "ratings" and model == "SVD interaction")
            row = {"domain": domain, "model": model, "n_rows": len(part), "clusters_total": len(rows),
                   "clusters_defined": int(defined.sum()), "clusters_undefined": int((~defined).sum()),
                   "mean_cluster_spearman": float(group_table.loc[defined, "spearman_manual"].mean()),
                   "baseline_clusters_defined": int(bdefined.sum()),
                   "baseline_mean_cluster_spearman": float(group_table.loc[bdefined, "baseline_spearman_manual"].mean()),
                   "paired_clusters_defined": int(paired.sum()),
                   "paired_mean_spearman_difference": float((group_table.loc[paired, "spearman_manual"] - group_table.loc[paired, "baseline_spearman_manual"]).mean()),
                   "original_scale_mae": float(np.abs(part.prediction-part.y).mean()) if mae_allowed else np.nan,
                   "mae_status": "computed" if mae_allowed else "interaction_only_score_not_rating_prediction",
                   "mae_units": UNITS[domain], "baseline_original_scale_mae": float(np.abs(part.baseline-part.y).mean()),
                   "exact_baseline_copy": bool(np.array_equal(part.prediction, part.baseline)),
                   "audit_beta": 2, "audit_raw_T": statistic, "audit_raw_p": raw_p,
                   "audit_raw_BY_p": float(audit.raw_by_adjusted_p_recomputed),
                   "audit_guarded_p": float(audit.p_policy), "audit_guarded_BY_p": float(audit.policy_by_adjusted_p),
                   "audit_final_label": "abstain" if audit.policy_abstain else ("retain" if audit.policy_by_reject else "null"),
                   "audit_label_scope": "operational screen; calibration not established",
                   "audit_abstention_reason": str(audit.redundancy_reason) if audit.redundancy_abstain else ("undefined_audit_scale" if audit.undefined_audit_abstain else "none"),
                   "cohort_sha256": chash, "audit_family_size": int(audit.policy_family_size)}
            for status in ["fewer_than_two_rows", "constant_prediction", "constant_outcome", "constant_prediction_and_outcome"]:
                row["undefined_" + status] = int(group_table.status.eq(status).sum())
            models.append(row)
            checks.append({"domain": domain, "model": model, "n": len(part), "clusters": len(rows),
                           "cohort_matches_complete_family": True, "cohort_sha256": chash,
                           "rank_and_spearman_checks": "PASS", "normal_tail_check": "PASS"})
        for column, saved, flag in [("p_policy", "policy_by_adjusted_p", "policy_by_reject"),
                                    ("raw_p_one_sided", "raw_by_adjusted_p_recomputed", "raw_by_reject_recomputed")]:
            p = family[column].to_numpy(float)
            adjusted = by_manual(p)
            reject, sm, _, _ = multipletests(p, alpha=.05, method="fdr_by")
            np.testing.assert_allclose(adjusted, sm, atol=1e-15, rtol=1e-12)
            np.testing.assert_allclose(adjusted, family[saved], atol=1e-15, rtol=1e-12)
            np.testing.assert_array_equal(adjusted <= .05, reject)
            np.testing.assert_array_equal(reject, family[flag].to_numpy(bool))
            family_checks.append({"domain": domain, "kind": column, "models": len(family),
                                  "rejects": int(reject.sum()), "status": "PASS"})
        print("Checked", domain, len(family), "models", flush=True)
    result = pd.DataFrame(models).sort_values(["domain", "model"])
    assert len(result) == 41
    csv_new(result, "public_all_41_models.csv")
    csv_new(pd.DataFrame(clusters), "public_all_cluster_correlations.csv")
    csv_new(result.loc[result.exact_baseline_copy], "public_baseline_copies.csv")
    json_new(OUTPUT / "verification.json", {"status": "PASS", "models_checked": len(checks),
        "model_cluster_pairs_checked": len(clusters), "rank_vectors_checked": 3*len(clusters),
        "max_rank_absolute_difference": max_rank_difference, "max_spearman_absolute_difference": max_rho_difference,
        "max_saved_normal_tail_absolute_difference": max_tail_difference,
        "cohorts": checks, "full_family_BY_checks": family_checks})
    json_new(OUTPUT / "receipt.json", {"status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic()-start, "protocol_sha256": sha(OUTPUT / "protocol.json"),
        "files": {p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}})
    print(result[["domain", "model", "mean_cluster_spearman", "clusters_defined", "clusters_undefined", "original_scale_mae", "audit_raw_T", "audit_final_label"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run"])
    parser.add_argument("--input-specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT = args.output
    spline_source.INPUT_SPECIFICATION = args.input_specification
    if args.stage == "freeze":
        OUTPUT.mkdir(parents=True, exist_ok=False)
        freeze()
    else:
        run()
