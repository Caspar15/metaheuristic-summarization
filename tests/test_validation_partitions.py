import hashlib
import json

import pytest

from src.data.partitions import (
    freeze_validation_partition_manifest,
    iter_partition_rows,
    partition_report_for_artifact,
    resolve_experiment_partition,
    selected_ids_sha256,
)


def _canonical_stub(row_id: str, split: str = "validation") -> dict:
    return {"id": row_id, "split": split}


def test_freeze_validation_partition_is_reference_blind_disjoint_and_complete(tmp_path):
    source = tmp_path / "validation.jsonl"
    rows = [
        {"id": f"v{i}", "split": "validation", "references": [f"secret-{i}"]}
        for i in range(10)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = freeze_validation_partition_manifest(
        source, dataset="Toy", seed=3407, dev_fraction=0.70
    )

    dev = manifest["partitions"]["dev"]["selected_ids"]
    dev_test = manifest["partitions"]["dev-test"]["selected_ids"]
    assert len(dev) == 7
    assert len(dev_test) == 3
    assert set(dev).isdisjoint(dev_test)
    assert set(dev) | set(dev_test) == {row["id"] for row in rows}
    assert manifest["partitions"]["dev"]["selected_ids_sha256"] == selected_ids_sha256(dev)
    assert "references" not in json.dumps(manifest)


def test_freeze_validation_partition_rejects_non_validation_rows(tmp_path):
    source = tmp_path / "wrong.jsonl"
    source.write_text(json.dumps(_canonical_stub("heldout", split="test")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only validation"):
        freeze_validation_partition_manifest(source, dataset="Toy", seed=1)


def test_resolve_and_stream_partition_requires_manifest_and_input_identity(tmp_path):
    source = tmp_path / "validation.jsonl"
    rows = [_canonical_stub(f"v{i}") for i in range(4)]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = freeze_validation_partition_manifest(
        source, dataset="Toy", seed=3, dev_fraction=0.50
    )
    manifest_path = tmp_path / "partition.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cfg = {
        "experiment": {"dataset": "Toy"},
        "experiment_partition": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "name": "dev",
        },
    }
    preflight = {
        "input_file_sha256": manifest["input_sha256"],
        "rows": 4,
    }
    resolved = resolve_experiment_partition(cfg, preflight)
    selected_rows = list(iter_partition_rows(rows, resolved))
    assert [row["id"] for row in selected_rows] == manifest["partitions"]["dev"]["selected_ids"]
    assert "selected_ids" not in partition_report_for_artifact(resolved)

    bad_preflight = dict(preflight, input_file_sha256="0" * 64)
    with pytest.raises(ValueError, match="input SHA-256"):
        resolve_experiment_partition(cfg, bad_preflight)


def test_iter_partition_rows_fails_if_manifest_id_is_missing():
    preflight = {
        "selected_ids": ["v0", "v-missing"],
        "rows": 2,
    }
    with pytest.raises(ValueError, match="missing 1 IDs"):
        list(iter_partition_rows([_canonical_stub("v0")], preflight))
