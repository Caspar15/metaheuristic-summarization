"""Reference-blind scaling projection for a frozen validation-dev partition.

This audit deliberately reads source sentence counts only.  It rejects every
partition except ``dev`` and has no route to a dataset test split.  The
quadratic proxy is appropriate for the pre-F-30 greedy implementation, whose
facility-coverage loop rebuilt an ``N x |selected|`` slice for every one of N
candidate extensions at each step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _sentence_count(row: dict) -> int:
    return sum(
        len(section.get("sentences", []))
        for document in row.get("documents", [])
        for section in document.get("sections", [])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--partition", required=True, choices=("dev",))
    parser.add_argument("--partial", required=True)
    parser.add_argument("--anchor-cpu-seconds", required=True, type=float)
    args = parser.parse_args()

    input_path = Path(args.input)
    manifest_path = Path(args.manifest)
    if "test" in input_path.name.lower() or "test" in manifest_path.name.lower():
        raise ValueError("scaling audit refuses paths whose filename contains 'test'")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_split") != "validation":
        raise ValueError("scaling audit accepts validation manifests only")
    partition = manifest["partitions"][args.partition]
    ordered_ids = list(partition["selected_ids"])
    selected_ids = set(ordered_ids)

    counts_by_id: dict[str, int] = {}
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in selected_ids:
                counts_by_id[row["id"]] = _sentence_count(row)
    if len(counts_by_id) != int(partition["rows"]):
        raise ValueError(
            f"partition row mismatch: expected {partition['rows']}, "
            f"got {len(counts_by_id)}"
        )
    counts = [counts_by_id[identifier] for identifier in ordered_ids]

    partial_path = Path(args.partial)
    partial_ids: list[str] = []
    with partial_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                partial_ids.append(str(json.loads(line).get("id")))
    if not partial_ids or len(partial_ids) >= len(ordered_ids):
        raise ValueError("partial artifact must contain a nonempty strict prefix")
    if partial_ids != ordered_ids[: len(partial_ids)]:
        raise ValueError("partial artifact IDs are not the frozen dev ordered prefix")
    if args.anchor_cpu_seconds <= 0:
        raise ValueError("anchor CPU seconds must be positive")

    processed_rows = len(partial_ids)
    processed_n_squared = sum(count * count for count in counts[:processed_rows])
    total_n_squared = sum(count * count for count in counts)
    ratio = total_n_squared / float(processed_n_squared)
    report = {
        "audit": "pre_f30_greedy_quadratic_scaling_projection",
        "input": input_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "base_split": "validation",
        "partition": "dev",
        "rows": len(counts),
        "references_read": False,
        "test_split_accessed": False,
        "sentence_counts": {
            "mean": sum(counts) / len(counts),
            "max": max(counts),
            "ge_1000": sum(count >= 1000 for count in counts),
            "ge_2000": sum(count >= 2000 for count in counts),
        },
        "anchor": {
            "partial_path": partial_path.as_posix(),
            "partial_bytes": partial_path.stat().st_size,
            "processed_rows": processed_rows,
            "last_processed_id": partial_ids[-1],
            "observed_cpu_seconds_lower_bound": args.anchor_cpu_seconds,
        },
        "processed_n_squared_share": processed_n_squared / total_n_squared,
        "sum_n_squared_ratio_total_to_processed_prefix": ratio,
        "projected_cpu_hours_lower_bound": ratio
        * args.anchor_cpu_seconds
        / 3600.0,
        "interpretation": (
            "prefix-calibrated complexity proxy for the interrupted pre-F-30 "
            "implementation; not a post-optimization runtime claim"
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
