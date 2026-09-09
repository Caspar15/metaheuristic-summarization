import hashlib
import json

import pytest

from scripts.audit.verify_submission_snapshot import (
    MANIFEST, REPO_ROOT, final_rows, verify_inventory,
)


def test_committed_snapshot_and_both_final_populations():
    manifest = json.loads((REPO_ROOT / MANIFEST).read_text(encoding="utf-8"))
    assert verify_inventory(REPO_ROOT, manifest) > 0
    rows = final_rows(REPO_ROOT)
    assert len(rows) == 18
    assert {row["rows"] for row in rows} == {973, 5621}


def test_snapshot_detects_tampering_and_missing_file(tmp_path):
    target = tmp_path / "evidence.json"
    target.write_bytes(b"original\n")
    manifest = {"files": {target.name: {"sha256": hashlib.sha256(target.read_bytes()).hexdigest()}}}
    assert verify_inventory(tmp_path, manifest) == 1
    target.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_inventory(tmp_path, manifest)
    target.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_inventory(tmp_path, manifest)


def test_only_source_mode_allows_checkout_line_endings(tmp_path):
    (tmp_path / "source.py").write_bytes(b"pass\r\n")
    pin = {"mode": "lf", "sha256": hashlib.sha256(b"pass\n").hexdigest()}
    manifest = {"files": {"source.py": pin}}
    assert verify_inventory(tmp_path, manifest) == 1
    pin["mode"] = "raw"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_inventory(tmp_path, manifest)


@pytest.mark.parametrize("path", ["../outside", "/outside", "C:/outside", "..\\outside"])
def test_snapshot_rejects_external_paths(tmp_path, path):
    with pytest.raises(ValueError, match="unsafe"):
        verify_inventory(tmp_path, {"files": {path: {"sha256": "0" * 64}}})


def test_empty_snapshot_cannot_pass(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        verify_inventory(tmp_path, {"files": {}})
