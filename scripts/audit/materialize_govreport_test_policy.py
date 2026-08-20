"""Materialize the authorized GovReport official-test policy before any score.

This Stage-A command canonicalizes the official test split, records structural
health and immutable identities, and writes no predictions or ROUGE scores.
It refuses to overwrite an existing policy package.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

from src.data.policy import sha256_binary_file, sha256_file
from src.data.preprocess_govreport import (
    DATASET_NAME,
    DATASET_REVISION,
    EXPECTED_TEST_ROWS,
    OFFICIAL_ARCHIVE_SHA256,
    PREPROCESSOR_VERSION,
    TEST_ID_MEMBERS,
    iter_test_examples,
)
from src.data.schemas import flatten_sentence_records
from src.data.validate_dataset import validate_jsonl
from src.utils.io import read_jsonl, write_jsonl_atomic


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION = Path("configs/data_policies/govreport_test_authorization_v1.json")
AUTHORIZATION_SHA256 = (
    "64184a537f25bbdbad7c2701ef32735304b6ee8b659a59342ad8b46780f4e277"
)
DEFAULT_ARCHIVE = Path("data/raw/gov-report.tar.gz")
DEFAULT_CANONICAL = Path("data/processed/govreport_test_canonical.jsonl")
DEFAULT_EXCLUSIONS = Path("configs/data_policies/govreport_test_exclusions_v1.json")
DEFAULT_REPLACEMENTS = Path(
    "configs/data_policies/govreport_test_replacement_characters_v1.json"
)
DEFAULT_HEALTH = Path("docs/research/evidence/govreport_test_canonical_health_v1.json")
DEFAULT_POLICY = Path("configs/data_policies/govreport_test_v1.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _membership_identity(archive_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    globally_seen: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for agency, member_name in TEST_ID_MEMBERS.items():
            stream = archive.extractfile(member_name)
            if stream is None:
                raise ValueError(f"cannot read official membership {member_name}")
            raw = stream.read()
            ids = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
            if len(ids) != EXPECTED_TEST_ROWS[agency] or len(ids) != len(set(ids)):
                raise ValueError(f"official {agency} test membership drifted")
            overlap = globally_seen.intersection(ids)
            if overlap:
                raise ValueError(f"cross-agency test ID collision: {sorted(overlap)[:5]}")
            globally_seen.update(ids)
            result[agency] = {
                "member_path": member_name,
                "rows": len(ids),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "ordered_ids_sha256": hashlib.sha256(
                    ("\n".join(ids) + "\n").encode("utf-8")
                ).hexdigest(),
            }
    return result


def materialize(
    *,
    archive_path: Path,
    canonical_path: Path,
    exclusions_path: Path,
    replacements_path: Path,
    health_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    authorization_path = REPO_ROOT / AUTHORIZATION
    if sha256_file(str(authorization_path)) != AUTHORIZATION_SHA256:
        raise ValueError("GovReport test authorization record drifted")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("stage_A", {}).get("authorized") is not True:
        raise ValueError("Stage A is not authorized")
    if authorization.get("protected_state_at_authorization", {}).get(
        "test_scores_observed"
    ) is not False:
        raise ValueError("authorization does not precede test scores")

    resolved = [
        REPO_ROOT / canonical_path,
        REPO_ROOT / exclusions_path,
        REPO_ROOT / replacements_path,
        REPO_ROOT / health_path,
        REPO_ROOT / policy_path,
    ]
    existing = [str(path.relative_to(REPO_ROOT)) for path in resolved if path.exists()]
    if existing:
        raise ValueError(f"refusing to overwrite Stage-A artifacts: {existing}")
    final_run_root = REPO_ROOT / "runs_v2/govreport_final_test_v1"
    if final_run_root.exists():
        raise ValueError("final-test output exists before policy materialization")

    archive = REPO_ROOT / archive_path
    if sha256_binary_file(str(archive)) != OFFICIAL_ARCHIVE_SHA256:
        raise ValueError("official GovReport archive SHA-256 drifted")
    membership = _membership_identity(archive)
    exclusions: list[dict[str, Any]] = []
    write_jsonl_atomic(
        str(REPO_ROOT / canonical_path),
        iter_test_examples(archive, exclusion_records=exclusions),
    )
    exclusion_manifest = {
        "manifest_schema_version": "1.0",
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "official_split": "test",
        "official_membership_rows": sum(EXPECTED_TEST_ROWS.values()),
        "canonical_rows": sum(EXPECTED_TEST_ROWS.values()) - len(exclusions),
        "rule": "exclude only rows whose official reference has zero paragraphs; never fabricate a target",
        "excluded_rows": exclusions,
        "created_before_predictions_or_scores": True,
    }
    _write_json(REPO_ROOT / exclusions_path, exclusion_manifest)

    replacement_rows: list[dict[str, Any]] = []
    for row in read_jsonl(str(REPO_ROOT / canonical_path)):
        source_count = sum(
            record["text"].count("\ufffd") for record in flatten_sentence_records(row)
        )
        reference_count = sum(reference.count("\ufffd") for reference in row["references"])
        if source_count or reference_count:
            replacement_rows.append(
                {
                    "id": row["id"],
                    "source_replacement_characters": source_count,
                    "reference_replacement_characters": reference_count,
                    "total_replacement_characters": source_count + reference_count,
                }
            )
    replacement_manifest = {
        "manifest_schema_version": "1.0",
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "split": "test",
        "rule": "retain and record every canonical row containing U+FFFD without text repair",
        "rows": replacement_rows,
        "row_count": len(replacement_rows),
        "character_count": sum(
            row["total_replacement_characters"] for row in replacement_rows
        ),
        "created_before_predictions_or_scores": True,
    }
    _write_json(REPO_ROOT / replacements_path, replacement_manifest)

    expected_rows = exclusion_manifest["canonical_rows"]
    report = validate_jsonl(
        str(REPO_ROOT / canonical_path),
        expected_split="test",
        expected_rows=expected_rows,
        expected_dataset_revision=DATASET_REVISION,
        expected_dataset_name=DATASET_NAME,
        expected_replacement_rows=replacement_manifest["row_count"],
        expected_replacement_characters=replacement_manifest["character_count"],
        allow_replacement_character=True,
    )
    if not report["valid"]:
        raise ValueError(f"canonical test health failed: {report['errors'][:5]}")
    recorded_at = _utc_now()
    health_evidence = {
        "evidence_schema_version": "1.0",
        "study_id": "govreport-test-policy-materialization-v1",
        "recorded_at_utc": recorded_at,
        "status": "completed_before_predictions_or_scores",
        "authorization_path": AUTHORIZATION.as_posix(),
        "authorization_sha256": AUTHORIZATION_SHA256,
        "source_archive": {
            "path": archive_path.as_posix(),
            "binary_sha256": OFFICIAL_ARCHIVE_SHA256,
        },
        "membership": membership,
        "canonical": {
            "path": canonical_path.as_posix(),
            "file_sha256": sha256_file(str(REPO_ROOT / canonical_path)),
            "dataset_fingerprint": report["dataset_fingerprint"],
            "rows": report["rows"],
        },
        "exclusion_manifest": {
            "path": exclusions_path.as_posix(),
            "file_sha256": sha256_file(str(REPO_ROOT / exclusions_path)),
            "rows": len(exclusions),
        },
        "replacement_character_manifest": {
            "path": replacements_path.as_posix(),
            "file_sha256": sha256_file(str(REPO_ROOT / replacements_path)),
            "rows": replacement_manifest["row_count"],
            "characters": replacement_manifest["character_count"],
        },
        "health": report,
        "test_predictions_generated": False,
        "test_scores_observed": False,
    }
    _write_json(REPO_ROOT / health_path, health_evidence)

    policy = {
        "policy_schema_version": "1.0",
        "policy_id": "govreport-test-v1",
        "status": "frozen_before_test_results",
        "frozen_at_utc": recorded_at,
        "authorization": {
            "path": AUTHORIZATION.as_posix(),
            "file_sha256": AUTHORIZATION_SHA256,
        },
        "test_split_accessed_for_policy_materialization": True,
        "test_predictions_generated": False,
        "test_scores_observed": False,
        "dataset": {
            "name": DATASET_NAME,
            "dataset_id": "official-govreport-author-archive",
            "dataset_revision": DATASET_REVISION,
            "preprocessor_version": PREPROCESSOR_VERSION,
            "split": "test",
            "license": "CC-BY-4.0",
            "official_dataset_page": "https://gov-report-data.github.io/",
        },
        "source_archive": {
            "path": archive_path.as_posix(),
            "file_sha256": OFFICIAL_ARCHIVE_SHA256,
            "official_test_membership_rows": sum(EXPECTED_TEST_ROWS.values()),
            "crs_test_ids": membership["crs"],
            "gao_test_ids": membership["gao"],
        },
        "canonical_exclusions": {
            "expected_rows": len(exclusions),
            "reason": "official references with zero paragraphs cannot be scored and are excluded before any prediction",
            "manifest_path": exclusions_path.as_posix(),
            "manifest_file_sha256": health_evidence["exclusion_manifest"]["file_sha256"],
        },
        "replacement_character_manifest": {
            "path": replacements_path.as_posix(),
            "rows": replacement_manifest["row_count"],
            "characters": replacement_manifest["character_count"],
            "file_sha256": health_evidence["replacement_character_manifest"]["file_sha256"],
        },
        "canonical_health_evidence": {
            "path": health_path.as_posix(),
            "file_sha256": sha256_file(str(REPO_ROOT / health_path)),
        },
        "analyses": {
            "main": {
                "role": "primary_test",
                "artifact_path": canonical_path.as_posix(),
                "row_policy": "exclude_only_pre_score_empty_reference_rows",
                "text_repair": "forbidden",
                "expected_rows": expected_rows,
                "expected_dataset_fingerprint": report["dataset_fingerprint"],
                "expected_file_sha256": health_evidence["canonical"]["file_sha256"],
                "expected_replacement_rows": replacement_manifest["row_count"],
                "expected_replacement_characters": replacement_manifest["character_count"],
                "allow_replacement_character": True,
            }
        },
        "section_policy": {
            "preserve_nested_section_order": True,
            "preserve_paragraph_position": True,
            "gao_letter_rule": "exclude top-level Letter paragraphs but retain Letter subsections, matching the official preprocessing README",
            "empty_section_nodes": "record and omit nodes that contribute no source sentences",
        },
        "interpretation_rule": "Evaluate every structurally valid official-test row; do not change exclusions after predictions or scores.",
    }
    _write_json(REPO_ROOT / policy_path, policy)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize_policy", action="store_true")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--replacements", type=Path, default=DEFAULT_REPLACEMENTS)
    parser.add_argument("--health", type=Path, default=DEFAULT_HEALTH)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    args = parser.parse_args()
    if not args.initialize_policy:
        raise SystemExit("--initialize_policy is required for the one-time Stage-A write")
    result = materialize(
        archive_path=args.archive,
        canonical_path=args.canonical,
        exclusions_path=args.exclusions,
        replacements_path=args.replacements,
        health_path=args.health,
        policy_path=args.policy,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["analyses"]["main"]["expected_rows"],
                "test_predictions_generated": result["test_predictions_generated"],
                "test_scores_observed": result["test_scores_observed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
