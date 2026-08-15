"""Regression tests for frozen dataset policy generation and enforcement."""

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pytest

from src.data.freeze_multinews_policy import (
    freeze_policy,
    materialize_clean_sensitivity,
)
from src.data.policy import (
    load_frozen_policy,
    sha256_file,
    validate_dataset_policy_request,
    verify_pin,
)
from src.data.preprocess_multinews import DATASET_REVISION
from src.data.schemas import build_document_example
from src.utils.io import write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_load_frozen_policy_accepts_test_policy_frozen_before_scores(policy_tmp_dir):
    path = policy_tmp_dir / "test_policy.json"
    path.write_text(
        json.dumps(
            {
                "policy_schema_version": "1.0",
                "status": "frozen_before_test_results",
                "analyses": {"main": {}},
            }
        ),
        encoding="utf-8",
    )
    assert load_frozen_policy(str(path))["status"] == "frozen_before_test_results"


@pytest.fixture
def policy_tmp_dir():
    # The managed Windows test host can deny pytest's mode-0700 ``tmp_path``
    # directories.  A unique ignored directory created with normal workspace
    # permissions is equivalent and keeps the regression portable.
    path = ROOT / "data" / "processed" / f"policy_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _row(example_id: str, text: str):
    return build_document_example(
        example_id=example_id,
        split="validation",
        documents=[[text]],
        references=["Reference."],
        input_mode="multi_document",
        output_mode="multi_sentence",
        dataset_name="Multi-News",
        metadata={
            "dataset_revision": DATASET_REVISION,
            "source_row_index": int(example_id.rsplit("_", 1)[-1]),
            "n_source_documents": 1,
        },
    )


@pytest.fixture
def frozen_fixture(policy_tmp_dir):
    main_path = policy_tmp_dir / "main.jsonl"
    clean_path = policy_tmp_dir / "clean.jsonl"
    manifest_path = policy_tmp_dir / "replacement.jsonl"
    policy_path = policy_tmp_dir / "policy.json"
    write_jsonl(
        str(main_path),
        [
            _row("validation_0", "Clean source sentence."),
            _row("validation_1", "Damaged \ufffd source sentence."),
            _row("validation_2", "Another clean source sentence."),
        ],
    )
    policy = freeze_policy(
        input_path=str(main_path),
        clean_output_path=str(clean_path),
        replacement_manifest_path=str(manifest_path),
        policy_output_path=str(policy_path),
        exclusion_manifest_path=None,
        expected_main_rows=3,
        expected_replacement_rows=1,
        expected_replacement_characters=1,
    )
    return {
        "main": main_path,
        "clean": clean_path,
        "manifest": manifest_path,
        "policy_path": policy_path,
        "policy": policy,
    }


def _config(policy_path, analysis):
    return {
        "experiment": {
            "status": "validation_pilot_only",
            "dataset": "multi_news",
        },
        "data_policy": {
            "policy_path": str(policy_path),
            "policy_sha256": sha256_file(str(policy_path)),
            "analysis": analysis,
        },
    }


def test_freeze_creates_exact_paired_clean_subset(frozen_fixture):
    policy = frozen_fixture["policy"]
    manifest_rows = [
        json.loads(line)
        for line in frozen_fixture["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    clean_ids = [
        json.loads(line)["id"]
        for line in frozen_fixture["clean"].read_text(encoding="utf-8").splitlines()
    ]

    assert clean_ids == ["validation_0", "validation_2"]
    assert [row["id"] for row in manifest_rows] == ["validation_1"]
    assert manifest_rows[0]["total_replacement_characters"] == 1
    assert policy["analyses"]["main"]["expected_rows"] == 3
    assert policy["analyses"]["clean_sensitivity"]["expected_rows"] == 2
    assert policy["analyses"]["main"]["text_repair"] == "forbidden"


def test_runtime_accepts_only_the_artifact_bound_to_each_analysis(frozen_fixture):
    main_report = validate_dataset_policy_request(
        _config(frozen_fixture["policy_path"], "main"),
        str(frozen_fixture["main"]),
        "validation",
    )
    clean_report = validate_dataset_policy_request(
        _config(frozen_fixture["policy_path"], "clean_sensitivity"),
        str(frozen_fixture["clean"]),
        "validation",
    )

    assert main_report["rows"] == 3
    assert main_report["replacement_character_rows"] == 1
    assert clean_report["rows"] == 2
    assert clean_report["replacement_character_rows"] == 0
    with pytest.raises(ValueError, match="frozen policy"):
        validate_dataset_policy_request(
            _config(frozen_fixture["policy_path"], "main"),
            str(frozen_fixture["clean"]),
            "validation",
        )


def test_runtime_rejects_experiment_policy_dataset_mismatch(frozen_fixture):
    cfg = _config(frozen_fixture["policy_path"], "main")
    cfg["experiment"]["dataset"] = "GovReport"
    with pytest.raises(ValueError, match="does not match frozen policy dataset"):
        validate_dataset_policy_request(
            cfg,
            str(frozen_fixture["main"]),
            "validation",
        )


def test_runtime_rejects_manifest_tampering(frozen_fixture):
    with frozen_fixture["manifest"].open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        validate_dataset_policy_request(
            _config(frozen_fixture["policy_path"], "main"),
            str(frozen_fixture["main"]),
            "validation",
        )


def test_runtime_rejects_policy_tampering(frozen_fixture):
    cfg = _config(frozen_fixture["policy_path"], "main")
    with frozen_fixture["policy_path"].open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="policy SHA-256"):
        validate_dataset_policy_request(
            cfg,
            str(frozen_fixture["main"]),
            "validation",
        )


def test_materialize_recreates_only_ignored_clean_artifact(frozen_fixture):
    policy_before = frozen_fixture["policy_path"].read_bytes()
    manifest_before = frozen_fixture["manifest"].read_bytes()
    frozen_fixture["clean"].unlink()

    report = materialize_clean_sensitivity(
        input_path=str(frozen_fixture["main"]),
        clean_output_path=str(frozen_fixture["clean"]),
        policy_path=str(frozen_fixture["policy_path"]),
    )

    assert report["clean_sensitivity"]["rows"] == 2
    assert frozen_fixture["policy_path"].read_bytes() == policy_before
    assert frozen_fixture["manifest"].read_bytes() == manifest_before


def test_freeze_fails_when_observed_damage_does_not_match_declared_policy(
    policy_tmp_dir,
):
    main_path = policy_tmp_dir / "main.jsonl"
    write_jsonl(str(main_path), [_row("validation_0", "Clean sentence.")])
    with pytest.raises(ValueError, match="policy precheck"):
        freeze_policy(
            input_path=str(main_path),
            clean_output_path=str(policy_tmp_dir / "clean.jsonl"),
            replacement_manifest_path=str(policy_tmp_dir / "manifest.jsonl"),
            policy_output_path=str(policy_tmp_dir / "policy.json"),
            expected_main_rows=1,
            expected_replacement_rows=1,
            expected_replacement_characters=1,
        )


def _write_errata(
    path,
    *,
    legacy_sha256: str,
    canonical_sha256: str,
    content_id: str = "toy fixture",
    representative_path: str = "some/toy/manifest.json",
    reference_count: int = 1,
):
    path.write_text(
        json.dumps(
            {
                "errata_version": "1.0",
                "equivalences": [
                    {
                        "content_id": content_id,
                        "representative_path": representative_path,
                        "legacy_crlf_sha256": legacy_sha256,
                        "lf_canonical_sha256": canonical_sha256,
                        "reference_count": reference_count,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_verify_pin_returns_pass_when_the_hash_already_matches(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")
    expected = sha256_file(str(target))

    assert verify_pin(str(target), expected) == "pass"


def test_verify_pin_returns_legacy_for_a_known_crlf_era_pin(tmp_path):
    target = tmp_path / "sub" / "manifest.json"
    target.parent.mkdir()
    lf_content = '{"a": 1}\n'
    target.write_text(lf_content, encoding="utf-8")
    canonical = sha256_file(str(target))
    legacy = hashlib.sha256(lf_content.replace("\n", "\r\n").encode("utf-8")).hexdigest()
    assert legacy != canonical

    errata_path = tmp_path / "errata.json"
    _write_errata(errata_path, legacy_sha256=legacy, canonical_sha256=canonical)

    result = verify_pin(str(target), legacy, errata_path=str(errata_path))

    assert result == "legacy"


def test_verify_pin_returns_legacy_regardless_of_where_the_file_actually_lives(tmp_path):
    """Matching is by hash pair alone: a legacy-era content blob copied to a
    path that has nothing to do with the errata's documentation-only
    ``representative_path`` must still be recognized, because this is
    exactly the shape of the dataset_preflight.json/partition_preflight.json
    family -- one identical blob copied into 100+ run directories."""

    target = tmp_path / "some" / "totally" / "different" / "directory" / "copy.json"
    target.parent.mkdir(parents=True)
    lf_content = '{"rows": 3935}\n'
    target.write_text(lf_content, encoding="utf-8")
    canonical = sha256_file(str(target))
    legacy = hashlib.sha256(lf_content.replace("\n", "\r\n").encode("utf-8")).hexdigest()

    errata_path = tmp_path / "errata.json"
    _write_errata(
        errata_path,
        legacy_sha256=legacy,
        canonical_sha256=canonical,
        representative_path="configs/validation_partitions/multinews_validation_dev_v1.json",
        reference_count=118,
    )

    assert verify_pin(str(target), legacy, errata_path=str(errata_path)) == "legacy"


def test_verify_pin_raises_when_not_covered_by_errata(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")
    wrong_expected = "f" * 64
    errata_path = tmp_path / "errata.json"
    _write_errata(errata_path, legacy_sha256="a" * 64, canonical_sha256="b" * 64)

    with pytest.raises(ValueError, match="not covered by errata"):
        verify_pin(str(target), wrong_expected, errata_path=str(errata_path))


def test_verify_pin_raises_when_only_legacy_matches_but_not_canonical(tmp_path):
    """``expected`` matches a recorded legacy hash, but the file's actual
    content matches neither ``expected`` nor that entry's canonical hash.
    Satisfying only the legacy side of the pair must still fail loud."""

    target = tmp_path / "manifest.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")
    legacy = "a" * 64
    errata_path = tmp_path / "errata.json"
    _write_errata(
        errata_path,
        legacy_sha256=legacy,
        canonical_sha256="c" * 64,  # does not match sha256_file(target)
    )

    with pytest.raises(ValueError, match="not covered by errata"):
        verify_pin(str(target), legacy, errata_path=str(errata_path))


def test_verify_pin_raises_when_only_canonical_matches_but_not_legacy(tmp_path):
    """The file's actual content happens to equal some errata entry's
    canonical hash, but the caller's ``expected`` is not that entry's
    legacy hash (it is not covered by the errata at all). A coincidental
    match on only the canonical side must not grant "legacy" status."""

    target = tmp_path / "manifest.json"
    lf_content = '{"a": 1}\n'
    target.write_text(lf_content, encoding="utf-8")
    canonical = sha256_file(str(target))
    legacy = hashlib.sha256(lf_content.replace("\n", "\r\n").encode("utf-8")).hexdigest()

    errata_path = tmp_path / "errata.json"
    _write_errata(errata_path, legacy_sha256=legacy, canonical_sha256=canonical)

    unrelated_expected = "f" * 64
    assert unrelated_expected != legacy

    with pytest.raises(ValueError, match="not covered by errata"):
        verify_pin(str(target), unrelated_expected, errata_path=str(errata_path))


def test_verify_pin_missing_errata_file_treats_everything_as_uncovered(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not covered by errata"):
        verify_pin(str(target), "d" * 64, errata_path=str(tmp_path / "does-not-exist.json"))


def test_verify_pin_raises_on_duplicate_legacy_hash_across_two_entries(tmp_path):
    """A hand-maintained errata file that grows over time must fail loud on
    a duplicate legacy_crlf_sha256 rather than silently keeping only the
    last entry -- a silent overwrite would make one equivalence vanish
    without any signal."""

    shared_legacy = "a" * 64
    errata_path = tmp_path / "errata.json"
    errata_path.write_text(
        json.dumps(
            {
                "errata_version": "1.0",
                "equivalences": [
                    {
                        "content_id": "first content",
                        "representative_path": "some/first/path.json",
                        "legacy_crlf_sha256": shared_legacy,
                        "lf_canonical_sha256": "b" * 64,
                        "reference_count": 1,
                    },
                    {
                        "content_id": "second content",
                        "representative_path": "some/second/path.json",
                        "legacy_crlf_sha256": shared_legacy,
                        "lf_canonical_sha256": "c" * 64,
                        "reference_count": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    target = tmp_path / "manifest.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate legacy_crlf_sha256"):
        verify_pin(str(target), shared_legacy, errata_path=str(errata_path))


def test_verify_pin_allows_the_same_canonical_hash_under_two_legacy_entries(tmp_path):
    """The inverse of the duplicate-legacy guard: the same current content
    having had more than one historical CRLF-era identity is expected, not
    an error -- there is no uniqueness requirement on lf_canonical_sha256."""

    target = tmp_path / "manifest.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")
    canonical = sha256_file(str(target))
    errata_path = tmp_path / "errata.json"
    errata_path.write_text(
        json.dumps(
            {
                "errata_version": "1.0",
                "equivalences": [
                    {
                        "content_id": "first historical identity",
                        "representative_path": "some/first/path.json",
                        "legacy_crlf_sha256": "a" * 64,
                        "lf_canonical_sha256": canonical,
                        "reference_count": 1,
                    },
                    {
                        "content_id": "second historical identity",
                        "representative_path": "some/second/path.json",
                        "legacy_crlf_sha256": "b" * 64,
                        "lf_canonical_sha256": canonical,
                        "reference_count": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert verify_pin(str(target), "a" * 64, errata_path=str(errata_path)) == "legacy"
    assert verify_pin(str(target), "b" * 64, errata_path=str(errata_path)) == "legacy"
