import json

import pytest

from src.data.preprocess_govreport import (
    GovReportPreprocessingError,
    process_example,
)
from src.data.schemas import flatten_sentence_records, validate_document_example


def test_crs_preserves_nested_sections_paragraph_metadata_and_summary():
    payload = {
        "id": "R1",
        "title": "Title",
        "released_date": "2020-01-01",
        "summary": ["First summary.", "Second summary."],
        "reports": {
            "section_title": "",
            "paragraphs": [],
            "subsections": [
                {
                    "section_title": "Introduction",
                    "paragraphs": ["First sentence. Second sentence."],
                    "subsections": [
                        {
                            "section_title": "Details",
                            "paragraphs": ["Nested detail."],
                            "subsections": [],
                        }
                    ],
                }
            ],
        },
    }
    row = process_example(
        payload,
        agency="crs",
        source_bytes=json.dumps(payload).encode("utf-8"),
    )
    validate_document_example(row)
    assert row["id"] == "validation_crs_R1"
    assert [section["heading"] for section in row["documents"][0]["sections"]] == [
        "Introduction",
        "Details",
    ]
    records = flatten_sentence_records(row)
    assert [record["text"] for record in records] == [
        "First sentence.",
        "Second sentence.",
        "Nested detail.",
    ]
    assert records[2]["metadata"]["heading_path"] == ["Introduction", "Details"]
    assert row["references"] == ["First summary.\nSecond summary."]


def test_gao_excludes_letter_paragraphs_but_keeps_letter_subsections():
    payload = {
        "id": "GAO-1",
        "title": "Title",
        "highlight": [
            {"section_title": "What GAO Found", "paragraphs": ["Finding."], "subsections": []}
        ],
        "report": [
            {
                "section_title": "Letter",
                "paragraphs": ["Excluded letter prose."],
                "subsections": [
                    {
                        "section_title": "Background",
                        "paragraphs": ["Included background."],
                        "subsections": [],
                    }
                ],
            },
            {"section_title": "Results", "paragraphs": ["Included result."], "subsections": []},
        ],
    }
    row = process_example(
        payload,
        agency="gao",
        source_bytes=json.dumps(payload).encode("utf-8"),
    )
    texts = [record["text"] for record in flatten_sentence_records(row)]
    assert texts == ["Included background.", "Included result."]
    assert row["metadata"]["excluded_gao_letter_paragraphs"] == 1
    assert row["references"] == ["Finding."]


def test_explicit_test_split_changes_only_governed_identity_fields():
    payload = {
        "id": "R-test",
        "summary": ["Reference."],
        "reports": {
            "section_title": "Body",
            "paragraphs": ["Source sentence."],
            "subsections": [],
        },
    }
    row = process_example(
        payload,
        agency="crs",
        source_bytes=json.dumps(payload).encode("utf-8"),
        split="test",
    )
    validate_document_example(row)
    assert row["id"] == "test_crs_R-test"
    assert row["split"] == "test"
    assert row["documents"][0]["document_id"] == "test_crs_R-test:d000"
    assert row["references"] == ["Reference."]


def test_unknown_split_fails_before_building_a_row():
    payload = {"id": "R", "summary": ["Reference."], "reports": {}}
    with pytest.raises(GovReportPreprocessingError, match="unsupported GovReport split"):
        process_example(
            payload,
            agency="crs",
            source_bytes=json.dumps(payload).encode("utf-8"),
            split="private",
        )


@pytest.mark.parametrize(
    "payload, agency, message",
    [
        ({"id": "R", "summary": [], "reports": {}}, "crs", "no non-empty"),
        ({"id": "G", "highlight": [], "report": []}, "gao", "non-empty section"),
        ({"id": "R", "summary": ["Summary"], "reports": []}, "crs", "structured object"),
    ],
)
def test_malformed_rows_fail_loud(payload, agency, message):
    with pytest.raises(GovReportPreprocessingError, match=message):
        process_example(
            payload,
            agency=agency,
            source_bytes=json.dumps(payload).encode("utf-8"),
        )
