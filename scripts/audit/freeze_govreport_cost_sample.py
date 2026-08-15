"""Freeze the reference-blind GovReport E2 cost/scaling sample.

Only source sentence counts and canonical IDs are inspected.  References,
predictions, and ROUGE artifacts are deliberately outside this script's
interface so the timing sample cannot be selected for favorable quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.data.schemas import flatten_sentence_records
from src.utils.io import read_jsonl


QUANTILES = (0.10, 0.50, 0.90)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def select_cost_rows(
    counts: Mapping[str, int], *, per_stratum: int = 10
) -> tuple[list[dict], dict[str, float]]:
    """Select unique IDs closest to q10/q50/q90 log sentence counts."""

    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    if len(counts) < per_stratum * len(QUANTILES):
        raise ValueError("not enough rows for disjoint cost strata")
    for row_id, count in counts.items():
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("every row must have a non-empty string ID")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"invalid sentence count for {row_id!r}: {count!r}")

    log_counts = {row_id: math.log1p(count) for row_id, count in counts.items()}
    values = np.asarray(list(log_counts.values()), dtype=float)
    targets = {
        f"q{int(quantile * 100):02d}": float(np.quantile(values, quantile))
        for quantile in QUANTILES
    }
    selected: list[dict] = []
    used: set[str] = set()
    for label, target in targets.items():
        available = [row_id for row_id in counts if row_id not in used]
        ranked = sorted(
            available,
            key=lambda row_id: (abs(log_counts[row_id] - target), row_id),
        )
        chosen = ranked[:per_stratum]
        if len(chosen) != per_stratum:
            raise ValueError(f"could not fill {label} cost stratum")
        for row_id in chosen:
            used.add(row_id)
            selected.append(
                {
                    "id": row_id,
                    "stratum": label,
                    "source_sentences": counts[row_id],
                    "log1p_source_sentences": log_counts[row_id],
                    "target_log1p_source_sentences": target,
                    "absolute_target_distance": abs(log_counts[row_id] - target),
                }
            )
    return selected, targets


def build_manifest(
    rows: Iterable[Mapping],
    *,
    dev_ids: Sequence[str],
    input_path: Path,
    input_sha256: str,
    partition_manifest_path: Path,
    partition_manifest_sha256: str,
    per_stratum: int = 10,
) -> dict:
    dev_set = set(dev_ids)
    if len(dev_set) != len(dev_ids):
        raise ValueError("dev partition contains duplicate IDs")
    counts: dict[str, int] = {}
    for line_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if row_id not in dev_set:
            continue
        if row_id in counts:
            raise ValueError(f"duplicate canonical ID {row_id!r}")
        # This is the only content-bearing field read for sampling.
        counts[str(row_id)] = len(flatten_sentence_records(row))
    if set(counts) != dev_set:
        missing = sorted(dev_set - set(counts))
        raise ValueError(f"canonical input is missing dev IDs: {missing[:5]}")

    selected, targets = select_cost_rows(counts, per_stratum=per_stratum)
    selected_ids = [row["id"] for row in selected]
    return {
        "manifest_schema_version": "1.0",
        "status": "frozen_before_any_e2_timing",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "reference_blind_govreport_dev_cost_scaling_sample",
        "dataset": "GovReport",
        "base_split": "validation",
        "partition": "frozen dev",
        "input_path": input_path.as_posix(),
        "input_sha256": input_sha256,
        "partition_manifest_path": partition_manifest_path.as_posix(),
        "partition_manifest_sha256": partition_manifest_sha256,
        "partition_rows": len(dev_ids),
        "references_or_predictions_read_for_sampling": False,
        "selection_algorithm": (
            "on log1p(source sentence count), visit q10/q50/q90 in that order; "
            "take the 10 nearest still-unselected IDs, tie by canonical ID"
        ),
        "quantile_method": "numpy.quantile default linear",
        "per_stratum": per_stratum,
        "sample_rows": len(selected_ids),
        "quantile_targets_log1p_sentences": targets,
        "selected_ids_sha256": _ids_sha256(selected_ids),
        "selected_ids": selected_ids,
        "rows": selected,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-stratum", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    partition_path = Path(args.partition_manifest)
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    dev_ids = partition["partitions"]["dev"]["selected_ids"]
    manifest = build_manifest(
        read_jsonl(str(input_path)),
        dev_ids=dev_ids,
        input_path=input_path,
        input_sha256=_sha256_file(input_path),
        partition_manifest_path=partition_path,
        partition_manifest_sha256=_sha256_file(partition_path),
        per_stratum=args.per_stratum,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    print(json.dumps({"status": manifest["status"], "rows": manifest["sample_rows"]}))


if __name__ == "__main__":
    main()
