"""Build governed canonical GovReport rows from the authors' archive.

The requested split's two official membership files and only their referenced
payloads are read. Report section/paragraph structure is retained in canonical
sections and sentence metadata. Validation remains the default for backward
compatibility; official test requires an explicit ``--split test`` invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable, Mapping, Sequence

from src.data.policy import sha256_binary_file
from src.data.preprocess_multinews import segment_document
from src.data.schemas import (
    SCHEMA_VERSION,
    compute_data_fingerprint,
    validate_document_example,
)
from src.utils.io import write_jsonl_atomic


DATASET_NAME = "GovReport"
DATASET_REVISION = "official-author-archive-2021-04-01"
PREPROCESSOR_VERSION = "govreport-canonical-v1"
OFFICIAL_ARCHIVE_SHA256 = (
    "bedf7a780910afbaa8bb64e794742722f3faea06d3c156fd1cedc1a7b5c4cd3c"
)
VALIDATION_ID_MEMBERS = {
    "crs": "gov-report/split_ids/crs_valid.ids",
    "gao": "gov-report/split_ids/gao_valid.ids",
}
EXPECTED_VALIDATION_ROWS = {"crs": 362, "gao": 612}
TEST_ID_MEMBERS = {
    "crs": "gov-report/split_ids/crs_test.ids",
    "gao": "gov-report/split_ids/gao_test.ids",
}
EXPECTED_TEST_ROWS = {"crs": 362, "gao": 611}
SPLIT_ID_MEMBERS = {
    "validation": VALIDATION_ID_MEMBERS,
    "test": TEST_ID_MEMBERS,
}
EXPECTED_SPLIT_ROWS = {
    "validation": EXPECTED_VALIDATION_ROWS,
    "test": EXPECTED_TEST_ROWS,
}
KNOWN_CANONICAL_EXCLUSIONS = {
    ("crs", "98-228"): {
        "reason": "empty_official_reference",
        "raw_json_sha256": "246a634dfe0eb44a73fe8916d9ae7db553fc8396aedee349a570b131ae50902c",
    }
}


class GovReportPreprocessingError(ValueError):
    """Raised when official GovReport content violates the frozen contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise GovReportPreprocessingError(f"{field} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise GovReportPreprocessingError(f"{field} must be non-empty")
    return normalized


def _read_member_bytes(archive: tarfile.TarFile, member_name: str) -> bytes:
    try:
        member = archive.getmember(member_name)
    except KeyError as error:
        raise GovReportPreprocessingError(
            f"official archive is missing {member_name}"
        ) from error
    if not member.isfile():
        raise GovReportPreprocessingError(f"archive member is not a file: {member_name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise GovReportPreprocessingError(f"cannot read archive member: {member_name}")
    return stream.read()


def load_split_ids(
    archive: tarfile.TarFile, split: str
) -> dict[str, list[str]]:
    """Read one official split membership and validate exact counts."""

    if split not in SPLIT_ID_MEMBERS:
        raise GovReportPreprocessingError(
            f"unsupported GovReport split {split!r}; choose one of {sorted(SPLIT_ID_MEMBERS)}"
        )

    result: dict[str, list[str]] = {}
    globally_seen: set[str] = set()
    for agency, member_name in SPLIT_ID_MEMBERS[split].items():
        raw = _read_member_bytes(archive, member_name)
        row_ids = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(row_ids) != EXPECTED_SPLIT_ROWS[split][agency]:
            raise GovReportPreprocessingError(
                f"{agency} {split} IDs contain {len(row_ids)} rows; "
                f"expected {EXPECTED_SPLIT_ROWS[split][agency]}"
            )
        if len(set(row_ids)) != len(row_ids):
            raise GovReportPreprocessingError(f"duplicate {agency} {split} ID")
        overlap = globally_seen & set(row_ids)
        if overlap:
            raise GovReportPreprocessingError(
                f"cross-agency {split} ID collision: {sorted(overlap)[:5]}"
            )
        globally_seen.update(row_ids)
        result[agency] = row_ids
    return result


def load_validation_ids(archive: tarfile.TarFile) -> dict[str, list[str]]:
    """Backward-compatible validation-only membership wrapper."""

    return load_split_ids(archive, "validation")


def _child_sections(node: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    children = node.get("subsections", [])
    if children is None:
        return []
    if not isinstance(children, list) or not all(isinstance(item, Mapping) for item in children):
        raise GovReportPreprocessingError("report subsections must be a list of objects")
    return children


def _walk_report_nodes(
    node: Mapping[str, Any],
    *,
    path: tuple[int, ...],
    parent_headings: tuple[str, ...],
) -> Iterable[tuple[Mapping[str, Any], tuple[int, ...], tuple[str, ...]]]:
    heading_raw = node.get("section_title", "")
    if not isinstance(heading_raw, str):
        raise GovReportPreprocessingError("section_title must be a string")
    heading = " ".join(heading_raw.split())
    heading_path = (*parent_headings, heading) if heading else parent_headings
    yield node, path, heading_path
    for child_position, child in enumerate(_child_sections(node)):
        yield from _walk_report_nodes(
            child,
            path=(*path, child_position),
            parent_headings=heading_path,
        )


def _top_report_nodes(payload: Mapping[str, Any], agency: str) -> list[Mapping[str, Any]]:
    field = "reports" if agency == "crs" else "report"
    raw_report = payload.get(field)
    if agency == "crs":
        if not isinstance(raw_report, Mapping):
            raise GovReportPreprocessingError("CRS reports must be a structured object")
        return [raw_report]
    if not isinstance(raw_report, list) or not raw_report:
        raise GovReportPreprocessingError("GAO report must be a non-empty section list")
    if not all(isinstance(item, Mapping) for item in raw_report):
        raise GovReportPreprocessingError("GAO report sections must be objects")
    return list(raw_report)


def _summary_paragraphs(payload: Mapping[str, Any], agency: str) -> list[str]:
    raw_summary = payload.get("summary" if agency == "crs" else "highlight")
    paragraphs: list[str] = []
    if agency == "crs":
        if not isinstance(raw_summary, list):
            raise GovReportPreprocessingError("CRS summary must be a list of paragraphs")
        paragraphs = [
            _normalize_text(paragraph, f"summary[{index}]")
            for index, paragraph in enumerate(raw_summary)
        ]
    else:
        if not isinstance(raw_summary, list) or not all(
            isinstance(item, Mapping) for item in raw_summary
        ):
            raise GovReportPreprocessingError("GAO highlight must be a section list")
        for top_position, top in enumerate(raw_summary):
            for node, path, _heading_path in _walk_report_nodes(
                top, path=(top_position,), parent_headings=()
            ):
                raw_paragraphs = node.get("paragraphs", [])
                if not isinstance(raw_paragraphs, list):
                    raise GovReportPreprocessingError("highlight paragraphs must be a list")
                for paragraph_position, paragraph in enumerate(raw_paragraphs):
                    paragraphs.append(
                        _normalize_text(
                            paragraph,
                            f"highlight{path}.paragraphs[{paragraph_position}]",
                        )
                    )
    if not paragraphs:
        raise GovReportPreprocessingError("reference summary contains no paragraphs")
    return paragraphs


def process_example(
    payload: Mapping[str, Any],
    *,
    agency: str,
    source_bytes: bytes,
    split: str = "validation",
) -> dict[str, Any]:
    """Convert one official payload to canonical structured form."""

    if agency not in {"crs", "gao"}:
        raise GovReportPreprocessingError(f"unknown GovReport agency {agency!r}")
    if split not in SPLIT_ID_MEMBERS:
        raise GovReportPreprocessingError(f"unsupported GovReport split {split!r}")
    source_id = payload.get("id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise GovReportPreprocessingError("payload id must be a non-empty string")
    example_id = f"{split}_{agency}_{source_id}"
    canonical_sections: list[dict[str, Any]] = []
    document_position = 0
    ignored_empty_nodes = 0
    excluded_letter_paragraphs = 0

    for top_position, top in enumerate(_top_report_nodes(payload, agency)):
        for node, path, heading_path in _walk_report_nodes(
            top, path=(top_position,), parent_headings=()
        ):
            raw_paragraphs = node.get("paragraphs", [])
            if not isinstance(raw_paragraphs, list):
                raise GovReportPreprocessingError("report paragraphs must be a list")
            heading_raw = node.get("section_title", "")
            heading = " ".join(heading_raw.split()) if isinstance(heading_raw, str) else ""
            exclude_paragraphs = agency == "gao" and len(path) == 1 and heading.casefold() == "letter"
            if exclude_paragraphs:
                excluded_letter_paragraphs += len(raw_paragraphs)
                raw_paragraphs = []

            sentences: list[dict[str, Any]] = []
            section_position = 0
            for paragraph_position, paragraph in enumerate(raw_paragraphs):
                normalized_paragraph = _normalize_text(
                    paragraph, f"report{path}.paragraphs[{paragraph_position}]"
                )
                paragraph_sentences, mappings = segment_document(normalized_paragraph)
                for sentence, mapping in zip(paragraph_sentences, mappings):
                    sentences.append(
                        {
                            "sentence_id": (
                                f"{example_id}:d000:s{document_position:06d}"
                            ),
                            "text": sentence,
                            "document_position": document_position,
                            "section_position": section_position,
                            "metadata": {
                                **mapping,
                                "paragraph_position": paragraph_position,
                                "section_tree_path": list(path),
                                "heading_path": list(heading_path),
                            },
                        }
                    )
                    document_position += 1
                    section_position += 1
            if not sentences:
                ignored_empty_nodes += 1
                continue
            section_index = len(canonical_sections)
            canonical_sections.append(
                {
                    "section_id": f"{example_id}:d000:section:{section_index:04d}",
                    "heading": heading or None,
                    "sentences": sentences,
                    "metadata": {
                        "section_tree_path": list(path),
                        "heading_path": list(heading_path),
                        "tree_depth": len(path) - 1,
                    },
                }
            )

    if not canonical_sections:
        raise GovReportPreprocessingError("report produced no non-empty canonical sections")
    reference_paragraphs = _summary_paragraphs(payload, agency)
    reference = "\n".join(reference_paragraphs)
    replacement_counts = {
        "source": sum(
            sentence["text"].count("\ufffd")
            for section in canonical_sections
            for sentence in section["sentences"]
        ),
        "reference": reference.count("\ufffd"),
    }
    metadata = {
        "dataset_id": "official-govreport-author-archive",
        "dataset_revision": DATASET_REVISION,
        "agency": agency.upper(),
        "source_report_id": source_id,
        "title": payload.get("title"),
        "url": payload.get("url"),
        "released_date": payload.get("released_date"),
        "published_date": payload.get("published_date"),
        "preprocessor_version": PREPROCESSOR_VERSION,
        "raw_json_sha256": _sha256_bytes(source_bytes),
        "n_sections": len(canonical_sections),
        "n_reference_paragraphs": len(reference_paragraphs),
        "ignored_empty_section_nodes": ignored_empty_nodes,
        "excluded_gao_letter_paragraphs": excluded_letter_paragraphs,
        "contains_replacement_character": sum(replacement_counts.values()) > 0,
        "replacement_character_count": replacement_counts,
    }
    example: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": example_id,
        "split": split,
        "documents": [
            {
                "document_id": f"{example_id}:d000",
                "source_order": 0,
                "sections": canonical_sections,
                "metadata": {
                    "agency": agency.upper(),
                    "source_report_id": source_id,
                    "raw_json_sha256": _sha256_bytes(source_bytes),
                },
            }
        ],
        "references": [reference],
        "task_profile": {
            "input_mode": "single_document",
            "output_mode": "multi_sentence",
        },
        "dataset_name": DATASET_NAME,
        "metadata": metadata,
    }
    example["data_fingerprint"] = compute_data_fingerprint(example)
    validate_document_example(example)
    return example


def iter_split_examples(
    archive_path: Path,
    *,
    split: str,
    exclusion_records: list[dict[str, Any]] | None = None,
    allow_unpinned_empty_reference_exclusions: bool = False,
) -> Iterable[dict[str, Any]]:
    """Yield one governed split without reading any other split's membership.

    Validation's one historical exclusion is pinned by raw hash. During the
    separately authorized Stage-A test-policy materialization only, an empty
    official reference may be recorded before any model score; every other
    structural problem fails loud.
    """

    if split not in SPLIT_ID_MEMBERS:
        raise GovReportPreprocessingError(f"unsupported GovReport split {split!r}")

    actual_sha256 = sha256_binary_file(str(archive_path))
    if actual_sha256 != OFFICIAL_ARCHIVE_SHA256:
        raise GovReportPreprocessingError(
            f"official archive SHA-256 is {actual_sha256}, "
            f"expected {OFFICIAL_ARCHIVE_SHA256}"
        )
    # Random access in a gzip-compressed tar replays decompression from the
    # beginning for every member (974 near-quadratic scans in the first
    # implementation). Read membership once, then collect desired payloads in
    # one sequential archive pass.
    with tarfile.open(archive_path, mode="r:gz") as archive:
        split_ids = load_split_ids(archive, split)
    wanted = {
        f"gov-report/{agency}/{source_id}.json": (agency, source_id)
        for agency in ("crs", "gao")
        for source_id in split_ids[agency]
    }
    payload_bytes: dict[tuple[str, str], bytes] = {}
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            key = wanted.get(member.name)
            if key is None:
                continue
            if not member.isfile():
                raise GovReportPreprocessingError(
                    f"{split} member is not a file: {member.name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise GovReportPreprocessingError(
                    f"cannot read {split} member: {member.name}"
                )
            payload_bytes[key] = stream.read()
    missing_members = set(wanted.values()) - set(payload_bytes)
    if missing_members:
        raise GovReportPreprocessingError(
            f"archive is missing {len(missing_members)} {split} payloads; "
            f"first={sorted(missing_members)[:5]}"
        )

    for agency in ("crs", "gao"):
        for source_id in split_ids[agency]:
            member_name = f"gov-report/{agency}/{source_id}.json"
            source_bytes = payload_bytes[(agency, source_id)]
            try:
                payload = json.loads(source_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise GovReportPreprocessingError(
                    f"invalid JSON for {agency}/{source_id}: {error}"
                ) from error
            if not isinstance(payload, Mapping):
                raise GovReportPreprocessingError(
                    f"{agency}/{source_id} payload must be an object"
                )
            if payload.get("id") != source_id:
                raise GovReportPreprocessingError(
                    f"member {member_name} contains id {payload.get('id')!r}"
                )
            exclusion = (
                KNOWN_CANONICAL_EXCLUSIONS.get((agency, source_id))
                if split == "validation"
                else None
            )
            if exclusion is not None:
                raw_sha256 = _sha256_bytes(source_bytes)
                if raw_sha256 != exclusion["raw_json_sha256"]:
                    raise GovReportPreprocessingError(
                        f"{agency}/{source_id} exclusion raw SHA-256 changed: "
                        f"{raw_sha256}"
                    )
                if payload.get("summary") != []:
                    raise GovReportPreprocessingError(
                        f"{agency}/{source_id} no longer has the anomaly "
                        "declared by the exclusion policy"
                    )
                if exclusion_records is not None:
                    exclusion_records.append(
                        {
                            "agency": agency.upper(),
                            "source_report_id": source_id,
                            "official_split": split,
                            "reason": exclusion["reason"],
                            "raw_json_sha256": raw_sha256,
                        }
                    )
                continue
            try:
                yield process_example(
                    payload,
                    agency=agency,
                    source_bytes=source_bytes,
                    split=split,
                )
            except GovReportPreprocessingError as error:
                if (
                    split == "test"
                    and allow_unpinned_empty_reference_exclusions
                    and str(error) == "reference summary contains no paragraphs"
                ):
                    if exclusion_records is not None:
                        exclusion_records.append(
                            {
                                "agency": agency.upper(),
                                "source_report_id": source_id,
                                "official_split": split,
                                "reason": "empty_official_reference",
                                "raw_json_sha256": _sha256_bytes(source_bytes),
                            }
                        )
                    continue
                raise GovReportPreprocessingError(
                    f"{agency}/{source_id}: {error}"
                ) from error


def iter_validation_examples(
    archive_path: Path,
    *,
    exclusion_records: list[dict[str, Any]] | None = None,
) -> Iterable[dict[str, Any]]:
    """Backward-compatible governed validation iterator."""

    yield from iter_split_examples(
        archive_path,
        split="validation",
        exclusion_records=exclusion_records,
    )


def iter_test_examples(
    archive_path: Path,
    *,
    exclusion_records: list[dict[str, Any]] | None = None,
) -> Iterable[dict[str, Any]]:
    """Authorized Stage-A test iterator with a pre-score empty-reference rule."""

    yield from iter_split_examples(
        archive_path,
        split="test",
        exclusion_records=exclusion_records,
        allow_unpinned_empty_reference_exclusions=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclusion_manifest", required=True)
    parser.add_argument("--split", choices=sorted(SPLIT_ID_MEMBERS), default="validation")
    args = parser.parse_args()
    exclusion_records: list[dict[str, Any]] = []
    write_jsonl_atomic(
        args.output,
        iter_split_examples(
            Path(args.archive),
            split=args.split,
            exclusion_records=exclusion_records,
            allow_unpinned_empty_reference_exclusions=args.split == "test",
        ),
    )
    expected_validation_exclusions = [
        {
            "agency": "CRS",
            "source_report_id": "98-228",
            "official_split": "validation",
            "reason": "empty_official_reference",
            "raw_json_sha256": "246a634dfe0eb44a73fe8916d9ae7db553fc8396aedee349a570b131ae50902c",
        }
    ]
    if args.split == "validation" and exclusion_records != expected_validation_exclusions:
        raise GovReportPreprocessingError(
            f"unexpected exclusion set after preprocessing: {exclusion_records}"
        )
    exclusion_manifest = {
        "manifest_schema_version": "1.0",
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "official_split": args.split,
        "official_membership_rows": sum(EXPECTED_SPLIT_ROWS[args.split].values()),
        "canonical_rows": sum(EXPECTED_SPLIT_ROWS[args.split].values())
        - len(exclusion_records),
        "rule": "exclude only rows with a pinned empty official reference; never fabricate a target",
        "excluded_rows": exclusion_records,
    }
    manifest_path = Path(args.exclusion_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(exclusion_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_manifest.replace(manifest_path)
    print(f"Wrote canonical GovReport {args.split} to {args.output}")


if __name__ == "__main__":
    main()
