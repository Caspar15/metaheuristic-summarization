"""Verify write-only self-referential provenance hashes against their files.

``dataset_preflight.json`` and ``partition_preflight.json`` are written once
per governed run, and their own SHA-256 is copied into the sibling
``evidence.json`` as ``dataset_preflight_sha256`` / ``partition_preflight_sha256``.
Nothing in this project ever reads those two fields back and re-hashes the
file they describe: the PR #16 CRLF pin audit found ~490 such copies whose
recorded digest no longer matched the committed file (the file was written
with CRLF line endings on Windows, checked into git as LF), and none of them
tripped a single fail-loud check anywhere in the pipeline, because nothing
was checking. This script closes that gap: it walks a directory tree, and
for every record that declares one of these fields, re-hashes the sibling
file it names and compares. It is meant to be run by hand or in CI as an
audit step, not imported into the governed run path itself.

Usage::

    python -m scripts.audit.verify_provenance --root runs_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from src.data.policy import sha256_file

# Record field name -> the sibling file (in the same directory as the
# record) whose SHA-256 that field is supposed to hold. Extend this mapping
# rather than guessing a target path from the field name: a fixed, explicit
# mapping cannot silently start checking the wrong file if a future field
# happens to share a naming convention by coincidence.
KNOWN_SELF_REFERENTIAL_FIELDS: dict[str, str] = {
    "dataset_preflight_sha256": "dataset_preflight.json",
    "partition_preflight_sha256": "partition_preflight.json",
    "config_used_json_sha256": "config_used.json",
    "feasibility_report_json_sha256": "feasibility_report.json",
    "baseline_run_json_sha256": "baseline_run.json",
}

# Record file names this script opens looking for the fields above. A record
# is any JSON file that might declare one of ``KNOWN_SELF_REFERENTIAL_FIELDS``;
# evidence.json is the only place this project currently writes them, but the
# list is kept explicit rather than "every *.json" to avoid silently trying to
# parse large non-record artifacts (predictions.jsonl-adjacent files, etc.).
RECORD_FILENAMES: tuple[str, ...] = ("evidence.json",)


class ProvenanceMismatch(RuntimeError):
    """Raised when one or more self-referential provenance hashes drift."""


def find_record_files(root: Path) -> Iterator[Path]:
    for name in RECORD_FILENAMES:
        yield from root.rglob(name)


def _load_record(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceMismatch(f"{path}: not valid JSON ({exc})") from exc


def check_record(record_path: Path) -> list[dict[str, Any]]:
    """Return one result dict per known self-referential field present."""

    record = _load_record(record_path)
    if not isinstance(record, dict):
        return []

    results: list[dict[str, Any]] = []
    for field, sibling_name in KNOWN_SELF_REFERENTIAL_FIELDS.items():
        if field not in record:
            continue
        expected = record[field]
        if not (isinstance(expected, str) and len(expected) == 64):
            raise ProvenanceMismatch(
                f"{record_path}: {field!r} is not a 64-character SHA-256 "
                f"string: {expected!r}"
            )
        sibling_path = record_path.parent / sibling_name
        if not sibling_path.is_file():
            raise ProvenanceMismatch(
                f"{record_path}: declares {field!r} but the file it "
                f"identifies is missing: {sibling_path}"
            )
        actual = sha256_file(str(sibling_path))
        results.append(
            {
                "record": str(record_path),
                "field": field,
                "target": str(sibling_path),
                "expected": expected,
                "actual": actual,
                "match": actual == expected,
            }
        )
    return results


def verify_root(root: Path) -> int:
    """Check every known self-referential field under ``root``.

    Returns the number of fields checked. Raises :class:`ProvenanceMismatch`
    listing every drifted field if any check fails; it does not stop at the
    first one, so a single run reports the full extent of drift instead of
    requiring one invocation per broken file.
    """

    if not root.is_dir():
        raise ProvenanceMismatch(f"root is not a directory: {root}")

    checked = 0
    mismatches: list[dict[str, Any]] = []
    for record_path in sorted(find_record_files(root)):
        for result in check_record(record_path):
            checked += 1
            if not result["match"]:
                mismatches.append(result)

    if mismatches:
        lines = [f"{len(mismatches)} provenance hash(es) drifted:"]
        for m in mismatches:
            lines.append(f"  record: {m['record']}")
            lines.append(f"    field:    {m['field']}")
            lines.append(f"    target:   {m['target']}")
            lines.append(f"    expected: {m['expected']}")
            lines.append(f"    actual:   {m['actual']}")
        raise ProvenanceMismatch("\n".join(lines))

    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="runs_v2",
        help="directory to scan recursively for evidence.json records",
    )
    args = parser.parse_args()
    root = Path(args.root)
    checked = verify_root(root)
    print(f"verified {checked} self-referential provenance hash(es) under {root}")


if __name__ == "__main__":
    main()
