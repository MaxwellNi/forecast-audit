"""Replay a declared public forecast family through the guarded CSV auditor.

This verifies an existing evaluation; it performs no forecast training,
selection, new hypothesis search or new statistical experiment.
"""
import argparse
import hashlib
import json
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_audit_cli as cli


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run(protocol_path, root, domain, output):
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol["sources"].items():
        if cli.sha256(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f"Source changed after specification was recorded: {name}")
    specification = next(row for row in protocol["families"] if row["domain"] == domain)
    frames = []
    for source in specification["files"]:
        path = root / source["path"]
        if cli.sha256(path) != source["sha256"]:
            raise RuntimeError("Input file does not match the recorded hash")
        frames.append(pd.read_parquet(path) if path.suffix == ".parquet" else
                      pd.read_csv(path, float_precision="round_trip", keep_default_na=False))
    frame = pd.concat(frames, ignore_index=True).rename(columns=specification["rename_columns"])
    # Exercise the actual CSV parser, including exact identifier preservation.
    # Temporary public observations are never written into release outputs.
    with tempfile.TemporaryDirectory(prefix="forecast_csv_replay_") as folder:
        local_csv = Path(folder) / "public_forecasts.csv"
        frame[list(cli.REQUIRED)].to_csv(local_csv, index=False, float_format="%.17g")
        frame = cli.read_input(local_csv, {name: name for name in cli.REQUIRED})
    if sorted(frame.model.unique()) != specification["models"]:
        raise AssertionError("Model family differs from the frozen registry")
    reference_path = root / protocol["reference_registry"]["path"]
    if cli.sha256(reference_path) != protocol["reference_registry"]["sha256"]:
        raise RuntimeError("Reference registry changed")
    reference = pd.read_csv(reference_path, keep_default_na=False, float_precision="round_trip")
    reference = reference[reference.domain == domain].set_index("model")
    begin = time.perf_counter()
    tables = cli.audit_frame(frame, frequency=specification["frequency"], lag=specification["lag"],
                             beta=protocol["beta"], ladder=protocol["ladder"], alpha=protocol["alpha"])
    result = tables["profile"].set_index("model")
    assert set(result.index) == set(reference.index)
    errors = []
    for name, row in result.iterrows():
        expected = reference.loc[name]
        assert row.final_label.lower() == expected.audit_final_label
        assert int(row.n) == int(expected.n_rows)
        assert row.policy_family_size == expected.audit_family_size
        assert cli.cohort_sha256(frame[frame.model == name]) == expected.cohort_sha256
        if row.inference_status == "computed":
            error = abs(float(row.statistic) - float(expected.audit_raw_T))
            np.testing.assert_allclose(row.statistic, float(expected.audit_raw_T), atol=2e-9, rtol=2e-9)
        else:
            assert expected.audit_raw_T == ""
            error = 0.
        np.testing.assert_allclose([row.raw_p_one_sided, row.p_policy, row.policy_by_adjusted_p],
                                   [float(expected.audit_raw_p), float(expected.audit_guarded_p), float(expected.audit_guarded_BY_p)],
                                   rtol=2e-8, atol=2e-11)
        errors.append(error)
    receipt = cli.write_outputs(tables, output, frequency=specification["frequency"], lag=specification["lag"],
                                 beta=protocol["beta"], ladder=protocol["ladder"], alpha=protocol["alpha"],
                                 input_hash=hashlib.sha256(json.dumps(specification["files"], sort_keys=True).encode()).hexdigest())
    verification = {"status": "PASS", "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "domain": domain, "models": len(result), "rows": len(frame),
                    "seconds": time.perf_counter() - begin,
                    "all_frozen_labels_match": True, "all_cohort_hashes_match": True,
                    "actual_CSV_parser_exercised": True,
                    "maximum_statistic_absolute_error": max(errors), "counts": receipt["counts"],
                    "protocol_sha256": cli.sha256(protocol_path),
                    "reference_registry_sha256": cli.sha256(reference_path),
                    "cli_receipt_sha256": cli.sha256(output / "receipt.json"),
                    "scope": "Replay of known public forecasts and diagnostics, without refitting or claiming new calibration evidence."}
    write_json(output / "verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.data_root, args.domain, args.output)
