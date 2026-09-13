"""Render and verify the public model tables against their full-precision CSV.

Original null/NULL decision codes mean non-rejection. Display them as
NOT_RETAINED and keep them distinct from an undefined number. CSV fields are read as strings so that a label is never parsed as NaN.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def display_label(value: str) -> str:
    return {"retain": "RETAIN", "null": "NOT_RETAINED", "abstain": "ABSTAIN"}[value.lower()]


def number(value: str, *, probability: bool = False) -> str:
    if value == "":
        return "undefined"
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite numerical fields must be represented as empty CSV fields")
    return format(parsed, ".6g" if probability else ".6f")


def render_roster(source: Path, markdown: Path) -> tuple[str, dict]:
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = {(row["domain"], row["model"]): row for row in rows}
    if len(rows) != 41 or len(records) != 41:
        raise ValueError("Expected the complete 41-model comparison without duplicate keys")
    if any(row["audit_final_label"] not in {"retain", "null", "abstain"} for row in rows):
        raise ValueError("Unknown decision label in source CSV")
    expected_copies = {key for key, row in records.items() if row["exact_baseline_copy"].lower() == "true"}
    domain = None
    seen = set()
    copies = set()
    changes = []
    lines = []
    for line_number, line in enumerate(markdown.read_text().splitlines(), 1):
        if line.startswith("### "):
            candidate = line[4:].split(":", 1)[0].strip().lower()
            if candidate in {key[0] for key in records}:
                domain = candidate
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            key = None
            copy = False
            if len(cells) == 9 and (cells[0], cells[1]) in records:
                key, copy = (cells[0], cells[1]), True
            elif len(cells) == 8 and (domain, cells[0]) in records:
                key = (domain, cells[0])
            if key is not None:
                row = records[key]
                group_count = f'{row["clusters_defined"]}/{row["clusters_total"]}'
                common = [number(row["mean_cluster_spearman"]), group_count]
                if copy:
                    expected = [*key, *common, number(row["original_scale_mae"]),
                                number(row["audit_raw_T"]), number(row["audit_raw_p"], probability=True),
                                number(row["audit_guarded_BY_p"], probability=True), display_label(row["audit_final_label"])]
                    if key in copies:
                        raise ValueError(f"Duplicate baseline-copy row: {key}")
                    copies.add(key)
                else:
                    expected = [row["model"], *common, row["clusters_undefined"],
                                number(row["original_scale_mae"]), number(row["audit_raw_T"]),
                                number(row["audit_guarded_BY_p"], probability=True), display_label(row["audit_final_label"])]
                    if key in seen:
                        raise ValueError(f"Duplicate model row: {key}")
                    seen.add(key)
                if cells != expected:
                    changes.append({"line": line_number, "domain": key[0], "model": key[1],
                                    "actual": cells, "expected": expected})
                line = "| " + " | ".join(expected) + " |"
        lines.append(line)
    if seen != set(records) or copies != expected_copies:
        raise ValueError("Markdown must contain every model once and every exact baseline copy once")
    counts = {label: sum(row["audit_final_label"] == label for row in rows)
              for label in ("retain", "null", "abstain")}
    return "\n".join(lines) + "\n", {"models": len(seen), "baseline_copies": len(copies),
            "decision_counts": counts, "mismatches": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write regenerated tables; omit to check the supplied Markdown")
    args = parser.parse_args()
    rendered, report = render_roster(args.source, args.markdown)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(rendered)
    print(json.dumps(report, indent=2))
    if not args.output and report["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
