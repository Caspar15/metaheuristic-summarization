import hashlib
import json

import pytest

from scripts.audit.verify_provenance import (
    ProvenanceMismatch,
    check_record,
    main,
    verify_root,
)
from src.data.policy import sha256_file


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))


def _write_errata(path, *, legacy_path: str, legacy_sha256: str, canonical_sha256: str):
    _write(
        path,
        json.dumps(
            {
                "errata_version": "1.0",
                "equivalences": [
                    {
                        "content_id": "toy fixture",
                        "representative_path": legacy_path,
                        "legacy_crlf_sha256": legacy_sha256,
                        "lf_canonical_sha256": canonical_sha256,
                        "reference_count": 1,
                    }
                ],
            }
        ),
    )


def test_verify_root_passes_when_every_declared_hash_matches(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
    _write(run_dir / "partition_preflight.json", '{"rows": 3}\n')
    evidence = {
        "dataset_preflight_sha256": sha256_file(str(run_dir / "dataset_preflight.json")),
        "partition_preflight_sha256": sha256_file(str(run_dir / "partition_preflight.json")),
    }
    _write(run_dir / "evidence.json", json.dumps(evidence))

    report = verify_root(tmp_path)

    assert report["counts"] == {"pass": 2, "legacy": 0, "fail": 0}
    assert report["fail_details"] == []


def test_verify_root_classifies_a_known_crlf_era_pin_as_legacy_not_pass(tmp_path):
    """Reproduces the PR #16 scenario end to end: a record recorded the
    CRLF-era hash of a file that has since been normalized to LF by git.
    That must be reported as "legacy", distinctly from "pass", and must not
    raise."""

    run_dir = tmp_path / "configs" / "validation_partitions"
    lf_content = '{"rows": 3935}\n'
    target = run_dir / "dataset_preflight.json"
    _write(target, lf_content)
    legacy_hash = hashlib.sha256(lf_content.replace("\n", "\r\n").encode("utf-8")).hexdigest()
    canonical_hash = sha256_file(str(target))
    assert legacy_hash != canonical_hash  # sanity: the two identities differ

    errata_path = tmp_path / "errata.json"
    _write_errata(
        errata_path,
        legacy_path="configs/validation_partitions/dataset_preflight.json",
        legacy_sha256=legacy_hash,
        canonical_sha256=canonical_hash,
    )
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": legacy_hash}),
    )

    report = verify_root(tmp_path, errata_path=str(errata_path))

    assert report["counts"] == {"pass": 0, "legacy": 1, "fail": 0}
    assert report["fail_details"] == []


def test_verify_root_buckets_a_genuine_mismatch_as_fail_without_raising(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": "0" * 64}),
    )

    report = verify_root(tmp_path)

    assert report["counts"] == {"pass": 0, "legacy": 0, "fail": 1}
    assert len(report["fail_details"]) == 1
    failure = report["fail_details"][0]
    assert failure["field"] == "dataset_preflight_sha256"
    assert failure["expected"] == "0" * 64
    assert "not covered by errata" in failure["detail"]


def test_verify_root_reports_every_result_not_just_the_first(tmp_path):
    for label in ("a", "b", "c"):
        run_dir = tmp_path / label
        _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
        _write(
            run_dir / "evidence.json",
            json.dumps({"dataset_preflight_sha256": "0" * 64}),
        )

    report = verify_root(tmp_path)

    assert report["counts"]["fail"] == 3
    assert {result["record"] for result in report["fail_details"]} == {
        str(tmp_path / label / "evidence.json") for label in ("a", "b", "c")
    }


def test_verify_root_still_fails_loud_when_the_named_sibling_is_missing(tmp_path):
    run_dir = tmp_path / "run"
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": "a" * 64}),
    )

    with pytest.raises(ProvenanceMismatch, match="declares .* but the file"):
        verify_root(tmp_path)


def test_verify_root_still_fails_loud_on_a_malformed_hash_field(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": "not-a-sha256"}),
    )

    with pytest.raises(ProvenanceMismatch, match="not a 64-character SHA-256"):
        verify_root(tmp_path)


def test_verify_root_still_fails_loud_on_invalid_json(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir / "evidence.json", "{not valid json")

    with pytest.raises(ProvenanceMismatch, match="not valid JSON"):
        verify_root(tmp_path)


def test_check_record_ignores_evidence_with_no_known_fields(tmp_path):
    record_path = tmp_path / "run" / "evidence.json"
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

    report = verify_root(tmp_path)

    assert report["counts"] == {"pass": 2, "legacy": 0, "fail": 0}


def test_main_exits_nonzero_only_when_fail_count_is_positive(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
    _write(
        run_dir / "evidence.json",
        json.dumps(
            {"dataset_preflight_sha256": sha256_file(str(run_dir / "dataset_preflight.json"))}
        ),
    )
    monkeypatch.setattr("sys.argv", ["verify_provenance", "--root", str(tmp_path)])

    main()  # must not raise / exit when there are zero failures

    output = capsys.readouterr().out
    assert "1 pass / 0 legacy / 0 fail" in output


def test_main_exits_nonzero_and_lists_failures_when_fail_count_is_positive(
    tmp_path, monkeypatch, capsys
):
    run_dir = tmp_path / "run"
    _write(run_dir / "dataset_preflight.json", '{"rows": 5}\n')
    _write(
        run_dir / "evidence.json",
        json.dumps({"dataset_preflight_sha256": "0" * 64}),
    )
    monkeypatch.setattr("sys.argv", ["verify_provenance", "--root", str(tmp_path)])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    output = capsys.readouterr().out
    assert "0 pass / 0 legacy / 1 fail" in output
    assert "FAILURE" in output
    assert str(run_dir / "evidence.json") in output
