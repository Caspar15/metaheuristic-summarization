"""Measure reference word lengths on one frozen validation partition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.data.partitions import iter_partition_rows, selected_ids_sha256
from src.data.policy import sha256_file
from src.data.schemas import extract_references
from src.utils.io import read_jsonl


def build_reference_length_report(
    input_path: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    partition: str,
) -> dict:
    actual_manifest_sha256 = sha256_file(str(manifest_path))
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"partition manifest SHA-256 is {actual_manifest_sha256}, "
            f"expected {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_optimization_scores":
        raise ValueError("partition manifest was not frozen before optimization scores")
    if manifest.get("input_sha256") != sha256_file(str(input_path)):
        raise ValueError("input SHA-256 does not match the frozen partition manifest")
    partition_entry = (manifest.get("partitions") or {}).get(partition)
    if not isinstance(partition_entry, dict):
        raise ValueError(f"partition manifest has no {partition!r} entry")
    selected_ids = partition_entry.get("selected_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError(f"partition {partition!r} has no selected IDs")
    if selected_ids_sha256(selected_ids) != partition_entry.get("selected_ids_sha256"):
        raise ValueError(f"partition {partition!r} selected-ID digest is inconsistent")
    partition_preflight = {
        "selected_ids": selected_ids,
        "rows": partition_entry.get("rows"),
    }

    lengths: list[int] = []
    for row_number, row in enumerate(
        iter_partition_rows(read_jsonl(str(input_path)), partition_preflight),
        start=1,
    ):
        if row.get("split") != "validation":
            raise ValueError(f"selected row {row_number} is not validation")
        references = extract_references(row)
        if len(references) != 1:
            raise ValueError(
                f"row {row.get('id')!r} has {len(references)} references; "
                "length protocol requires an explicit multi-reference rule"
            )
        lengths.append(len(references[0].split()))
    values = np.asarray(lengths, dtype=float)
    percentiles = {
        f"p{percentile}": float(np.percentile(values, percentile))
        for percentile in (10, 25, 50, 75, 90)
    }
    return {
        "evidence_schema_version": "1.0",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "reference_lengths_only_no_system_scores",
        "test_split_accessed": False,
        "dataset": manifest.get("dataset"),
        "base_split": manifest.get("base_split"),
        "partition": partition,
        "input_path": input_path.as_posix(),
        "input_sha256": manifest["input_sha256"],
        "partition_manifest_path": manifest_path.as_posix(),
        "partition_manifest_sha256": actual_manifest_sha256,
        "selected_ids_sha256": partition_entry["selected_ids_sha256"],
        "rows": len(lengths),
        "word_count_rule": "len(reference.split()) on the single canonical reference",
        "distribution": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            **percentiles,
            "min": int(values.min()),
            "max": int(values.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest_sha256", required=True)
    parser.add_argument("--partition", required=True, choices=("dev", "dev-test"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = build_reference_length_report(
        Path(args.input),
        Path(args.manifest),
        expected_manifest_sha256=args.manifest_sha256,
        partition=args.partition,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(report["distribution"], ensure_ascii=False))


if __name__ == "__main__":
    main()
