"""Freeze and apply observed-redundancy abstention to saved public forecasts.

The input specification supplies local file paths. Published protocols retain
file names and hashes rather than machine-specific paths. No forecaster is
refitted and no raw normal-reference diagnostic is changed by this runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from audit_decisions import inspect_control_redundancy, apply_decision_policy


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    directory = Path(__file__).resolve().parent
    return {name: sha(directory / name) for name in [
        "audit_decisions.py", "reproduce_guarded_decisions.py",
        "audit_panel_predictions.py", "cluster_covariance_reference.py"]}


def input_records(specification):
    records = []
    for domain in specification["domains"]:
        profiles = []
        families = []
        for name, item in domain["profiles"].items():
            path = Path(item["path"])
            models = sorted(pd.read_csv(path, usecols=["model"]).model.unique().tolist())
            families.append(models)
            profiles.append({"name": name, "file": path.name, "bytes": path.stat().st_size,
                "sha256": sha(path), "specification_columns": item["specification_columns"]})
        if not families or any(family != families[0] for family in families):
            raise ValueError("all raw profiles must describe the same complete model family")
        inputs = [{"file": Path(p).name, "bytes": Path(p).stat().st_size, "sha256": sha(p)}
                  for p in domain["predictions"]]
        records.append({"domain": domain["domain"], "prediction_inputs": inputs,
            "rename": domain.get("rename", {}), "profiles": profiles, "family_models": families[0]})
    if len({record["domain"] for record in records}) != len(records):
        raise ValueError("domain names must be unique")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run"])
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.specification.read_text())
    records = input_records(spec)
    protocol_path = args.output / "protocol.json"
    if args.stage == "freeze":
        args.output.mkdir(parents=True, exist_ok=False)
        protocol = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "frozen_before_policy_counts",
            "motivation_disclosure": "The rule was chosen after examining earlier raw baseline-copy failures. This is a new deterministic abstention policy, not an external preregistration or a calibration experiment.",
            "rules_in_priority_order": ["exact observed raw baseline equality", "zero forecast rank variation in every cluster",
                "same weak ordering with identical tie partitions in every cluster", "reversed weak ordering with identical tie partitions in every cluster"],
            "one_common_order_orientation_across_clusters": True,
            "comparison_representation": "Finite binary64 input equality; exact integer twice-average ranks, signed zeros tied, no approximate comparison",
            "equality_tolerance": 0, "fitted_overlap_threshold": None,
            "minimum_cluster_cutoff": None,
            "small_cluster_policy": "Record cluster sizes, singleton and informative-cluster counts; do not claim a universal normal-approximation cutoff.",
            "outcomes_used_by_redundancy_guard": False,
            "undefined_raw_audit_policy": "abstain with p_policy=1; retain original diagnostic fields",
            "nonabstained_policy": "p_policy=p_raw; NOMINAL_POSITIVE iff p_policy<alpha; otherwise NOT_POSITIVE",
            "multiplicity": "BY recomputed for every unchanged complete family and fixed specification, including p=1 abstentions",
            "alpha": .05, "source_hashes": source_hashes(), "inputs": records,
            "scope": "Observed sample redundancy only. Neither population measurability nor real-data normal calibration is established. No optional witness functions or inferred mappings."}
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        print(json.dumps({"protocol_sha256": sha(protocol_path), "family_sizes": [len(x["family_models"]) for x in records]}))
        return
    protocol = json.loads(protocol_path.read_text())
    if protocol["source_hashes"] != source_hashes() or protocol["inputs"] != records:
        raise ValueError("source or input bytes changed after protocol freeze")
    if (args.output / "receipt.json").exists():
        raise ValueError("completed policy outputs cannot be overwritten")
    summaries, evidence_rows, output_hashes = [], [], {}
    for domain, record in zip(spec["domains"], records):
        directory = args.output / domain["domain"]
        directory.mkdir(exist_ok=False)
        panels = [pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
                  for path in domain["predictions"]]
        frame = pd.concat(panels, ignore_index=True).rename(columns=domain.get("rename", {}))
        if frame[["model", "entity", "period"]].isna().any().any() or frame.duplicated(["model", "entity", "period"]).any():
            raise ValueError("invalid saved forecast model/entity/period keys")
        if set(frame.model) != set(record["family_models"]):
            raise ValueError("saved forecast family differs from frozen raw profiles")
        evidence = {}
        for name, group in frame.groupby("model", sort=True):
            item = inspect_control_redundancy(group.prediction, group.baseline, group.period)
            evidence[name] = item
            evidence_rows.append({"domain": domain["domain"], "model": name, **item.to_dict()})
        for profile_name, item in domain["profiles"].items():
            raw = pd.read_csv(item["path"])
            result = apply_decision_policy(raw, evidence, family_models=record["family_models"],
                specification_columns=item["specification_columns"], alpha=protocol["alpha"])
            for _, group in result.groupby(item["specification_columns"], sort=False):
                summaries.append({"domain": domain["domain"], "profile": profile_name,
                    **{name: group[name].iloc[0].item() if isinstance(group[name].iloc[0], np.generic) else group[name].iloc[0]
                       for name in item["specification_columns"]},
                    "family_size": len(group), "raw_nominal": int(group.raw_nominal_positive.sum()),
                    "raw_BY": int(group.raw_by_reject_recomputed.sum()),
                    "redundancy_abstentions": int(group.redundancy_abstain.sum()),
                    "undefined_abstentions": int(group.undefined_audit_abstain.sum()),
                    "union_abstentions": int(group.policy_abstain.sum()),
                    "policy_nominal": int(group.policy_nominal_positive.sum()),
                    "policy_BY": int(group.policy_by_reject.sum())})
            path = directory / (profile_name + "_guarded.csv")
            result.to_csv(path, index=False)
            output_hashes[str(path.relative_to(args.output))] = sha(path)
    pd.DataFrame(evidence_rows).to_csv(args.output / "redundancy_evidence.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    output_hashes.update({name: sha(args.output / name) for name in ["redundancy_evidence.csv", "summary.csv"]})
    if protocol["source_hashes"] != source_hashes() or protocol["inputs"] != input_records(spec):
        raise ValueError("source or inputs changed during the policy run")
    receipt = {"status": "PASS", "protocol_sha256": sha(protocol_path),
        "source_hashes": source_hashes(), "models": len(evidence_rows),
        "raw_diagnostic_fields_preserved": True, "full_family_recomputed_BY": True,
        "outcomes_read_by_guard": False, "forecasters_refitted": False,
        "output_sha256": output_hashes, "summary": summaries}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
