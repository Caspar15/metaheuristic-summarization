"""Create the score-free Multi-News test U+FFFD manifest.

The command reads only canonical source/reference text, records damaged rows,
and refuses to overwrite its output.  It never generates predictions or scores.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from src.data.freeze_multinews_policy import replacement_character_counts
from src.utils.io import read_jsonl, write_jsonl_atomic


EXPECTED_ROWS = 70
EXPECTED_CHARACTERS = 733


def materialize(input_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite replacement manifest: {output_path}")
    manifest: list[dict[str, object]] = []
    ids_digest = hashlib.sha256()
    for line_number, row in enumerate(read_jsonl(str(input_path)), start=1):
        source_count, reference_count = replacement_character_counts(row)
        total = source_count + reference_count
        if not total:
            continue
        row_id = str(row["id"])
        ids_digest.update(row_id.encode("utf-8"))
        ids_digest.update(b"\n")
        manifest.append(
            {
                "id": row_id,
                "source_row_index": row.get("metadata", {}).get("source_row_index"),
                "canonical_line_number": line_number,
                "data_fingerprint": row["data_fingerprint"],
                "source_replacement_characters": source_count,
                "reference_replacement_characters": reference_count,
                "total_replacement_characters": total,
                "main_analysis_action": "retain_row_without_text_repair",
            }
        )
    characters = sum(int(row["total_replacement_characters"]) for row in manifest)
    if len(manifest) != EXPECTED_ROWS or characters != EXPECTED_CHARACTERS:
        raise ValueError(
            f"replacement identity drifted: {len(manifest)} rows/{characters} chars"
        )
    write_jsonl_atomic(str(output_path), manifest)
    return {
        "rows": len(manifest),
        "characters": characters,
        "row_ids_sha256": ids_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/processed/multi_news_test_canonical.jsonl"
    )
    parser.add_argument(
        "--output",
        default="configs/data_policies/multinews_test_replacement_rows_v1.jsonl",
    )
    args = parser.parse_args()
    print(materialize(Path(args.input), Path(args.output)))


if __name__ == "__main__":
    main()
