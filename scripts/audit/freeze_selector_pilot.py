"""Freeze a reference-blind, deterministic validation pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.utils.io import read_jsonl


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_manifest(input_path: Path, *, sample_size: int, salt: str) -> dict:
    """Choose a uniform hash sample without reading reference contents."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    row_ids = []
    seen = set()
    for line_number, row in enumerate(read_jsonl(str(input_path)), start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"input row {line_number} has no valid id")
        if row_id in seen:
            raise ValueError(f"duplicate input id: {row_id}")
        seen.add(row_id)
        row_ids.append(row_id)
    if sample_size > len(row_ids):
        raise ValueError("sample_size exceeds the input row count")

    ranked = sorted(
        row_ids,
        key=lambda row_id: (
            hashlib.sha256(f"{salt}\0{row_id}".encode("utf-8")).hexdigest(),
            row_id,
        ),
    )
    selected_set = set(ranked[:sample_size])
    # Preserve canonical input order.  This makes sequential artifacts easy
    # to audit while membership remains a reference-blind hash sample.
    selected_ids = [row_id for row_id in row_ids if row_id in selected_set]
    selection_digest = hashlib.sha256(
        "\n".join(selected_ids).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_schema_version": "1.0",
        "status": "frozen_before_selector_pilot_scores",
        "purpose": "reference_blind_uniform_hash_validation_pilot",
        "input_path": input_path.as_posix(),
        "input_sha256": file_sha256(input_path),
        "input_rows": len(row_ids),
        "selection_algorithm": "lowest_sha256(salt\\0row_id), then canonical input order",
        "salt": salt,
        "sample_size": sample_size,
        "selected_ids_sha256": selection_digest,
        "selected_ids": selected_ids,
        "interpretation": (
            "Diagnostic validation pilot only; it cannot replace the governed "
            "5,621-row Multi-News primary validation analysis."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--salt", default="multinews-selector-pilot-v1")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = freeze_manifest(
        Path(args.input), sample_size=args.sample_size, salt=args.salt
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    print(f"Wrote frozen manifest to {output}")


if __name__ == "__main__":
    main()
