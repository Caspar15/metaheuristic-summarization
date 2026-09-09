"""Verify the review snapshot and export final score tables without datasets/models.

Uses only the Python standard library. This checks archived evidence, not a new
benchmark execution. The manifest is a dated inventory, not a pre-test freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "docs/research/evidence/submission_snapshot_2026_09_09.json"
DATASETS = {"govreport": 973, "multinews": 5621}


def verify_inventory(root: Path, manifest: dict) -> int:
    """Check every inventoried file, rejecting empty/unsafe manifests and drift."""
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("snapshot must contain a nonempty files mapping")
    root = root.resolve()
    for relative, pin in entries.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
            raise ValueError(f"unsafe snapshot path: {relative}")
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"missing or external snapshot file: {relative}")
        payload = target.read_bytes()
        # Source/configuration text may have CRLF; result evidence is byte-pinned.
        mode = pin.get("mode", "raw")
        if mode == "lf":
            payload = payload.replace(b"\r\n", b"\n")
        elif mode != "raw":
            raise ValueError(f"unknown snapshot hash mode: {mode}")
        if hashlib.sha256(payload).hexdigest() != pin["sha256"]:
            raise ValueError(f"snapshot hash mismatch: {relative}")
    return len(entries)


def final_rows(root: Path) -> list[dict]:
    """Read published corpus scores; do not recompute paired differences from them."""
    rows = []
    for dataset, expected_rows in DATASETS.items():
        source = f"runs_v2/{dataset}_final_test_v1/analysis.json"
        analysis = json.loads((root / source).read_text(encoding="utf-8"))
        if analysis["status"] != "completed" or analysis["rows"] != expected_rows:
            raise ValueError(f"unexpected final population/status: {dataset}")
        ranking = analysis["corpus_ranking"]
        if len(ranking) != 9 or len({row["system"] for row in ranking}) != 9:
            raise ValueError(f"expected nine distinct systems: {dataset}")
        for scores in ranking:
            rows.append({"dataset": dataset, "rows": expected_rows, **scores, "source": source})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, help="new directory for JSON/CSV score tables")
    args = parser.parse_args()
    manifest = json.loads((REPO_ROOT / MANIFEST).read_text(encoding="utf-8"))
    count = verify_inventory(REPO_ROOT, manifest)
    rows = final_rows(REPO_ROOT)
    report = {
        "status": "pass", "files_verified": count,
        "datasets": DATASETS, "score_rows": len(rows),
        "verification_scope": "archived source and evidence; no inference or evaluation rerun",
        "manifest_sha256": hashlib.sha256((REPO_ROOT / MANIFEST).read_bytes()).hexdigest(),
    }
    if args.export_dir:
        # Never overwrite a frozen run or an earlier export.
        args.export_dir.mkdir(parents=True, exist_ok=False)
        (args.export_dir / "verification.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (args.export_dir / "final_scores.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        with (args.export_dir / "final_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
