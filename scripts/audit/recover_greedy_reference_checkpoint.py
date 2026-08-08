"""Archive a corrupted greedy-reference checkpoint and retain its exact prefix.

This recovery utility has no split argument. It is limited to the preregistered
frozen-dev Gate 2 study and records the invalidated rows in evidence/search log.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from scripts.audit.run_gate2_greedy_reference import (
    OUTPUT_ROOT,
    _checkpoint_prefix_length,
    _checkpoint_rows,
    _exclusive_run_lock,
    _load_frozen_rows,
    _load_protocol,
)
from scripts.audit.run_length_contract_study import (
    _append_search_log,
    _canonical_sha256,
    _relative,
    _utc_now,
    _write_json,
)
from src.data.policy import sha256_file


def recover_checkpoint(dataset: str, target: str) -> dict[str, Any]:
    protocol = _load_protocol()
    spec = protocol["datasets"][dataset]
    frozen_rows, partition = _load_frozen_rows(spec)
    ordered_ids = [str(row["id"]) for row in frozen_rows]
    root = OUTPUT_ROOT / dataset / "dev" / target
    rows_path = root / "rows.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    if (root / "evidence.json").exists():
        raise ValueError("refusing to recover a completed greedy-reference run")

    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    config_hash = _canonical_sha256(config)
    checkpoint = _checkpoint_rows(rows_path)
    valid_prefix = _checkpoint_prefix_length(checkpoint, ordered_ids, target)
    if valid_prefix == len(checkpoint):
        raise ValueError("checkpoint is already an exact frozen prefix")

    attempts_root = root / "attempts"
    attempt_number = len(list(attempts_root.glob("attempt_*_checkpoint_recovery"))) + 1
    attempt_id = f"attempt_{attempt_number:02d}_checkpoint_recovery"
    attempt_root = attempts_root / attempt_id
    attempt_root.mkdir(parents=True, exist_ok=False)
    archived_path = attempt_root / "rows_corrupt.jsonl"
    shutil.copy2(rows_path, archived_path)

    raw_lines = rows_path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != len(checkpoint):
        raise ValueError("raw checkpoint line count differs from parsed row count")
    temporary = rows_path.with_suffix(".recovery.tmp")
    with temporary.open("wb") as stream:
        stream.writelines(raw_lines[:valid_prefix])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, rows_path)

    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "invalidated_and_recovered",
        "finding": "F-56",
        "reason": "concurrent Windows process trees wrote the same checkpoint",
        "dataset": spec["label"],
        "partition": "dev",
        "optimization_target": target,
        "original_rows": len(checkpoint),
        "valid_prefix_rows": valid_prefix,
        "discarded_rows": len(checkpoint) - valid_prefix,
        "original_rows_sha256": sha256_file(str(archived_path)),
        "archived_rows_path": _relative(archived_path),
        "recovered_rows_path": _relative(rows_path),
        "recovered_rows_sha256": sha256_file(str(rows_path)),
        "scientific_config_sha256": config_hash,
        "partition_selected_ids_sha256": partition["selected_ids_sha256"],
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    evidence_path = attempt_root / "recovery_evidence.json"
    _write_json(evidence_path, evidence)
    _append_search_log(
        {
            "logged_at_utc": _utc_now(),
            "study_id": "gate2-greedy-reference-v1",
            "dataset": spec["label"],
            "partition": "dev",
            "family": "greedy_reference",
            "candidate": target,
            "method": "metric_specific_greedy_reference",
            "config_hash": config_hash,
            "run_attempt": attempt_id,
            "dev_score": None,
            "dev_test_score": None,
            "status": "failed",
            "promoted": False,
            "reason": evidence["reason"],
            "completed_prefix_rows": valid_prefix,
            "discarded_rows": len(checkpoint) - valid_prefix,
            "comparison_family_size": 6,
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("multinews", "govreport"))
    parser.add_argument("--target", required=True, choices=("rouge1", "rouge2", "rougeLsum"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lock_path = OUTPUT_ROOT / ".locks" / f"{args.dataset}-{args.target}.lock"
    with _exclusive_run_lock(lock_path):
        evidence = recover_checkpoint(args.dataset, args.target)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
