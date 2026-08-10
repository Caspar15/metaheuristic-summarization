"""Verify write-only self-referential provenance hashes against their files.

``dataset_preflight.json`` and ``partition_preflight.json`` are written once
per governed run, and their own SHA-256 is copied into the sibling
``evidence.json`` as ``dataset_preflight_sha256`` / ``partition_preflight_sha256``.
Nothing in this project ever reads those two fields back and re-hashes the
file they describe: the PR #16 CRLF pin audit found ~490 such copies whose
recorded digest no longer matched the committed file (the file was written
with CRLF line endings on Windows, checked into git as LF), and none of them
tripped a single fail-loud check anywhere in the pipeline, because nothing
was checking. This script closes that gap.

Every recorded field is classified into exactly one of three buckets, using
:func:`src.data.policy.verify_pin`:

- ``pass``   -- the sibling file's current SHA-256 equals the recorded value
  directly.
- ``legacy`` -- the recorded value is a known CRLF-era pin (see
  ``configs/pin_errata_lf_normalization.json``), and the sibling file's
  current content matches the errata's LF-canonical hash for it. This is
  the expected, benign state for run records produced before PR #16's
  ``sha256_file`` fix (~490 records at the time of writing): they honestly
  recorded the CRLF-era hash they saw when they ran, and nothing ever read
  them back, so there was never a reason to update them.
- ``fail``   -- neither of the above. The sibling file's content does not
  match what the record claims, under any known identity. This is the only
  bucket that indicates an actual problem.

A malformed hash field or a record naming a sibling file that does not
exist is not part of this classification -- those are schema-level defects
in the record itself, not a hash disagreement, and still fail loud
immediately (:class:`ProvenanceMismatch`) rather than being counted as
``fail``.

Usage::

    python -m scripts.audit.verify_provenance --root runs_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from src.data.policy import DEFAULT_PIN_ERRATA_PATH, verify_pin

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
    """Raised for a schema-level provenance defect, not a hash disagreement.

    A missing sibling file, a malformed hash field, invalid JSON, or a
    non-directory root all indicate the record itself is broken, which is
    a different kind of problem from "this hash is a legacy CRLF-era pin"
    or "this hash is genuinely wrong" -- both of the latter are reported
    through the pass/legacy/fail classification instead.
    """


def find_record_files(root: Path) -> Iterator[Path]:
    for name in RECORD_FILENAMES:
        yield from root.rglob(name)


def _load_record(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceMismatch(f"{path}: not valid JSON ({exc})") from exc


def check_record(
    record_path: Path, *, errata_path: str = DEFAULT_PIN_ERRATA_PATH
) -> list[dict[str, Any]]:
    """Return one classified result dict per known field present.

    Each result's ``status`` is ``"pass"``, ``"legacy"``, or ``"fail"``,
    from :func:`src.data.policy.verify_pin`. A malformed field or a missing
    sibling file raises :class:`ProvenanceMismatch` immediately instead of
    being classified.
    """

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
        try:
            status = verify_pin(str(sibling_path), expected, errata_path=errata_path)
            detail = None
        except ValueError as exc:
            status = "fail"
            detail = str(exc)
        results.append(
            {
                "record": str(record_path),
                "field": field,
                "target": str(sibling_path),
                "expected": expected,
                "status": status,
                "detail": detail,
            }
        )
    return results


def verify_root(
    root: Path, *, errata_path: str = DEFAULT_PIN_ERRATA_PATH
) -> dict[str, Any]:
    """Classify every known self-referential field under ``root``.

    Returns ``{"counts": {"pass": int, "legacy": int, "fail": int},
    "fail_details": [...]}``. This never raises for a hash disagreement --
    ``fail`` entries are collected, not thrown -- so a single run always
    reports the full pass/legacy/fail picture. It still raises
    :class:`ProvenanceMismatch` immediately for a schema-level defect (see
    :func:`check_record`) or a root that is not a directory, since those
    are not something the three-way classification is meant to describe.
    """

    if not root.is_dir():
        raise ProvenanceMismatch(f"root is not a directory: {root}")

    counts = {"pass": 0, "legacy": 0, "fail": 0}
    fail_details: list[dict[str, Any]] = []
    for record_path in sorted(find_record_files(root)):
        for result in check_record(record_path, errata_path=errata_path):
            counts[result["status"]] += 1
            if result["status"] == "fail":
                fail_details.append(result)

    return {"counts": counts, "fail_details": fail_details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="runs_v2",
        help="directory to scan recursively for evidence.json records",
    )
    parser.add_argument(
        "--errata",
        default=DEFAULT_PIN_ERRATA_PATH,
        help="path to the LF-normalization pin errata JSON",
    )
    args = parser.parse_args()
    root = Path(args.root)
    report = verify_root(root, errata_path=args.errata)
    counts = report["counts"]
    total = counts["pass"] + counts["legacy"] + counts["fail"]
    print(
        f"{root}: {total} checked -- "
        f"{counts['pass']} pass / {counts['legacy']} legacy / {counts['fail']} fail"
    )
    if counts["fail"]:
        print(f"{counts['fail']} FAILURE(S):")
        for result in report["fail_details"]:
            print(f"  record: {result['record']}")
            print(f"    field:    {result['field']}")
            print(f"    target:   {result['target']}")
            print(f"    expected: {result['expected']}")
            print(f"    detail:   {result['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
