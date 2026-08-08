"""Audit the canonical GovReport validation layer without scoring systems."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data.policy import sha256_file
from src.data.schemas import (
    extract_references,
    flatten_sentence_records,
    validate_document_example,
)
from src.utils.io import read_jsonl


def _distribution(values: Iterable[int]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "mean": float(array.mean()),
        "min": int(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def build_report(
    input_path: Path,
    *,
    archive_path: Path,
    exclusion_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exclusion_manifest = json.loads(
        exclusion_manifest_path.read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    agency_counts: Counter[str] = Counter()
    section_counts: list[int] = []
    source_word_counts: list[int] = []
    reference_word_counts: list[int] = []
    sentence_word_counts: list[int] = []
    rows_missing_title = 0
    rows_missing_url = 0
    rows_with_unnamed_sections = 0
    unnamed_sections = 0
    sentences_missing_section_path = 0
    sentences_missing_paragraph_position = 0
    ignored_empty_nodes = 0
    excluded_letter_paragraphs = 0
    replacement_rows: list[dict[str, Any]] = []
    longest_sentence = {"words": -1, "row_id": None, "sentence_id": None}
    dataset_digest = hashlib.sha256()

    for row in read_jsonl(str(input_path)):
        dataset_digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        dataset_digest.update(b"\n")
        validate_document_example(row)
        if row.get("split") != "validation":
            raise ValueError(f"non-validation row encountered: {row.get('id')!r}")
        if row.get("dataset_name") != "GovReport":
            raise ValueError(f"unexpected dataset name: {row.get('dataset_name')!r}")
        rows.append(row)
        metadata = row["metadata"]
        agency_counts[str(metadata["agency"])] += 1
        rows_missing_title += not bool(metadata.get("title"))
        rows_missing_url += not bool(metadata.get("url"))
        ignored_empty_nodes += int(metadata["ignored_empty_section_nodes"])
        excluded_letter_paragraphs += int(
            metadata["excluded_gao_letter_paragraphs"]
        )

        sections = row["documents"][0]["sections"]
        section_counts.append(len(sections))
        unnamed = sum(section.get("heading") is None for section in sections)
        unnamed_sections += unnamed
        rows_with_unnamed_sections += unnamed > 0

        records = flatten_sentence_records(row)
        source_words = 0
        row_replacements = 0
        for record in records:
            words = len(record["text"].split())
            source_words += words
            sentence_word_counts.append(words)
            row_replacements += record["text"].count("\ufffd")
            sentence_metadata = record.get("metadata") or {}
            sentences_missing_section_path += not isinstance(
                sentence_metadata.get("section_tree_path"), list
            )
            sentences_missing_paragraph_position += (
                "paragraph_position" not in sentence_metadata
            )
            if words > longest_sentence["words"]:
                longest_sentence = {
                    "words": words,
                    "row_id": row["id"],
                    "sentence_id": record["sentence_id"],
                }
        source_word_counts.append(source_words)

        references = extract_references(row)
        if len(references) != 1:
            raise ValueError(f"row {row['id']!r} must contain one reference")
        reference_word_counts.append(len(references[0].split()))
        row_replacements += references[0].count("\ufffd")
        if row_replacements:
            replacement_rows.append(
                {"id": row["id"], "replacement_characters": row_replacements}
            )

    if len(rows) != 973:
        raise ValueError(f"canonical GovReport contains {len(rows)} rows, expected 973")
    dataset_fingerprint = dataset_digest.hexdigest()
    replacement_manifest = {
        "manifest_schema_version": "1.0",
        "dataset": "GovReport",
        "dataset_revision": "official-author-archive-2021-04-01",
        "split": "validation",
        "rule": "record every canonical row containing U+FFFD without text repair",
        "rows": replacement_rows,
        "row_count": len(replacement_rows),
        "character_count": sum(
            row["replacement_characters"] for row in replacement_rows
        ),
    }
    report = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "data_layer_only_no_system_scores",
        "test_split_accessed": False,
        "dataset": "GovReport",
        "dataset_revision": "official-author-archive-2021-04-01",
        "official_validation_membership_rows": 974,
        "canonical_validation_rows": len(rows),
        "canonical_input": {
            "path": input_path.as_posix(),
            "file_sha256": sha256_file(str(input_path)),
            "dataset_fingerprint": dataset_fingerprint,
        },
        "official_archive": {
            "path": archive_path.as_posix(),
            "file_sha256": sha256_file(str(archive_path)),
            "expected_file_sha256": "bedf7a780910afbaa8bb64e794742722f3faea06d3c156fd1cedc1a7b5c4cd3c",
        },
        "license": {
            "name": "Creative Commons Attribution 4.0 International",
            "spdx": "CC-BY-4.0",
            "official_dataset_page": "https://gov-report-data.github.io/",
        },
        "official_sources": {
            "paper": "https://aclanthology.org/2021.naacl-main.112/",
            "dataset_page": "https://gov-report-data.github.io/",
            "code": "https://github.com/luyang-huang96/LongDocSum",
        },
        "exclusion_manifest": {
            "path": exclusion_manifest_path.as_posix(),
            "file_sha256": sha256_file(str(exclusion_manifest_path)),
            "excluded_rows": len(exclusion_manifest["excluded_rows"]),
            "rule": exclusion_manifest["rule"],
        },
        "agency_rows": dict(sorted(agency_counts.items())),
        "section_metadata": {
            "sections_per_row": _distribution(section_counts),
            "unnamed_sections": unnamed_sections,
            "rows_with_unnamed_sections": rows_with_unnamed_sections,
            "sentences_missing_section_tree_path": sentences_missing_section_path,
            "sentences_missing_paragraph_position": sentences_missing_paragraph_position,
            "ignored_empty_section_nodes": ignored_empty_nodes,
            "excluded_gao_letter_paragraphs": excluded_letter_paragraphs,
        },
        "text_lengths": {
            "source_words_per_row": _distribution(source_word_counts),
            "reference_words_per_row": _distribution(reference_word_counts),
            "sentence_words": _distribution(sentence_word_counts),
            "longest_sentence": longest_sentence,
        },
        "metadata_completeness": {
            "rows_missing_title": rows_missing_title,
            "rows_missing_url": rows_missing_url,
        },
        "replacement_characters": {
            "rows": replacement_manifest["row_count"],
            "characters": replacement_manifest["character_count"],
            "text_repair": "forbidden",
        },
    }
    return report, replacement_manifest


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--exclusion_manifest", required=True)
    parser.add_argument("--replacement_manifest_out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report, replacement_manifest = build_report(
        Path(args.input),
        archive_path=Path(args.archive),
        exclusion_manifest_path=Path(args.exclusion_manifest),
    )
    _write_json(Path(args.replacement_manifest_out), replacement_manifest)
    _write_json(Path(args.out), report)
    print(json.dumps({"rows": report["canonical_validation_rows"], "agency_rows": report["agency_rows"], "reference_words": report["text_lengths"]["reference_words_per_row"]}))


if __name__ == "__main__":
    main()
