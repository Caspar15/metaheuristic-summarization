import hashlib
import json

import pytest

from scripts.audit.verify_provenance import ProvenanceMismatch, check_record, verify_root
from src.data.policy import sha256_file


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))


def test_verify_root_passes_when_every_declared_hash_matches(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
    _write(run_dir / "partition_preflight.json", '{"rows": 3}\n')
    evidence = {
        "dataset_preflight_sha256": sha256_file(str(run_dir / "dataset_preflight.json")),
        "partition_preflight_sha256": sha256_file(str(run_dir / "partition_preflight.json")),
    }
    _write(run_dir / "evidence.json", json.dumps(evidence))

    checked = verify_root(tmp_path)

    assert checked == 2


def test_verify_root_normalizes_crlf_exactly_like_the_writer_that_drifted(tmp_path):
    """The exact PR #16 scenario: file written with CRLF, hash recorded from
    that CRLF content, file later normalized to LF by git. The verifier must
    still pass, because both sides go through the same normalizing hash."""

    run_dir = tmp_path / "run"
    lf_content = '{"rows": 5}\n'
    crlf_bytes = lf_content.replace("\n", "\r\n").encode("utf-8")
    recorded_hash = hashlib.sha256(crlf_bytes.replace(b"\r\n", b"\n")).hexdigest()
    # The file on disk now (post git-normalization) is pure LF.
    _write(run_dir / "dataset_preflight.json", lf_content)
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": recorded_hash}),
    )

    checked = verify_root(tmp_path)

    assert checked == 1


def test_verify_root_raises_on_a_genuinely_drifted_hash(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": "0" * 64}),
    )

    with pytest.raises(ProvenanceMismatch) as excinfo:
        verify_root(tmp_path)

    message = str(excinfo.value)
    assert "dataset_preflight_sha256" in message
    assert str(run_dir / "dataset_preflight.json") in message
    assert "0" * 64 in message


def test_verify_root_reports_every_mismatch_not_just_the_first(tmp_path):
    for label in ("a", "b"):
        run_dir = tmp_path / label
        _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
        _write(
            run_dir / "evidence.json",
            json.dumps({"dataset_preflight_sha256": "0" * 64}),
        )

    with pytest.raises(ProvenanceMismatch) as excinfo:
        verify_root(tmp_path)

    message = str(excinfo.value)
    assert str(tmp_path / "a" / "dataset_preflight.json") in message
    assert str(tmp_path / "b" / "dataset_preflight.json") in message
    assert message.startswith("2 provenance hash(es) drifted")


def test_verify_root_fails_loud_when_the_named_sibling_is_missing(tmp_path):
    run_dir = tmp_path / "run"
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": "a" * 64}),
    )

    with pytest.raises(ProvenanceMismatch, match="declares .* but the file"):
        verify_root(tmp_path)


def test_verify_root_fails_loud_on_a_malformed_hash_field(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": "not-a-sha256"}),
    )

    with pytest.raises(ProvenanceMismatch, match="not a 64-character SHA-256"):
        verify_root(tmp_path)


def test_verify_root_fails_loud_on_invalid_json(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "evidence.json", "{not valid json")

    with pytest.raises(ProvenanceMismatch, match="not valid JSON"):
        verify_root(tmp_path)


def test_check_record_ignores_evidence_with_no_known_fields(tmp_path):
    run_dir = tmp_path / "run"
    record_path = run_dir / "evidence.json"
    _write(record_path, json.dumps({"status": "completed"}))

    assert check_record(record_path) == []


def test_check_record_ignores_non_object_json(tmp_path):
    record_path = tmp_path / "evidence.json"
    _write(record_path, json.dumps([1, 2, 3]))

    assert check_record(record_path) == []


def test_verify_root_rejects_a_root_that_is_not_a_directory(tmp_path):
    missing = tmp_path / "does-not-exist"

    with pytest.raises(ProvenanceMismatch, match="not a directory"):
        verify_root(missing)


def test_verify_root_checks_multiple_known_fields_in_one_record(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "config_used.json", '{"seed": 1}\n')
    _write(run_dir / "feasibility_report.json", '{"feasible": true}\n')
    evidence = {
        "config_used_json_sha256": sha256_file(str(run_dir / "config_used.json")),
        "feasibility_report_json_sha256": sha256_file(
            str(run_dir / "feasibility_report.json")
        ),
    }
    _write(run_dir / "evidence.json", json.dumps(evidence))

    assert verify_root(tmp_path) == 2
