"""Reference-blind development partitions inside a governed validation split.

The canonical schema intentionally keeps the upstream split name (``validation``).
``dev`` and ``dev-test`` are experiment partitions, not replacement dataset
splits.  A governed runner must therefore validate the *complete* canonical
validation artifact against its frozen data policy before filtering row IDs
through one of these manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

from src.data.policy import sha256_file, verify_pin


PARTITION_SCHEMA_VERSION = "1.0"
PARTITION_STATUS = "frozen_before_optimization_scores"
ALLOWED_PARTITIONS = frozenset({"dev", "dev-test"})


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def selected_ids_sha256(selected_ids: Iterable[str]) -> str:
    """Hash an ordered ID list with an explicit LF representation."""

    return hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()


def _load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("validation partition manifest must be a JSON object")
    if manifest.get("manifest_schema_version") != PARTITION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported validation partition manifest schema "
            f"{manifest.get('manifest_schema_version')!r}"
        )
    if manifest.get("status") != PARTITION_STATUS:
        raise ValueError(
            "development runs require a partition frozen before optimization scores"
        )
    if manifest.get("base_split") != "validation":
        raise ValueError("development partition base_split must be 'validation'")
    return manifest


def resolve_experiment_partition(
    cfg: Mapping[str, Any],
    dataset_preflight: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    """Validate and resolve ``cfg.experiment_partition``.

    The return value contains ``selected_ids`` for the streaming runner.  Use
    :func:`partition_report_for_artifact` before serializing it as provenance.
    """

    partition_cfg = cfg.get("experiment_partition")
    if partition_cfg is None:
        return None
    if not isinstance(partition_cfg, Mapping):
        raise ValueError("experiment_partition must be an object")
    if dataset_preflight is None:
        raise ValueError(
            "experiment_partition requires a governed full-artifact data preflight"
        )

    manifest_path = partition_cfg.get("manifest_path")
    expected_manifest_sha256 = partition_cfg.get("manifest_sha256")
    partition_name = partition_cfg.get("name")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        raise ValueError("experiment_partition.manifest_path must be a non-empty path")
    if (
        not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
    ):
        raise ValueError("experiment_partition.manifest_sha256 must be a SHA-256 digest")
    if partition_name not in ALLOWED_PARTITIONS:
        raise ValueError(
            f"experiment_partition.name must be one of {sorted(ALLOWED_PARTITIONS)}"
        )
    if not os.path.isfile(manifest_path):
        raise ValueError(f"validation partition manifest is missing: {manifest_path}")
    pin_status = verify_pin(manifest_path, expected_manifest_sha256)
    if pin_status == "legacy":
        print(f"[legacy pin] {manifest_path} (CRLF-era pin, see errata)")
    actual_manifest_sha256 = sha256_file(manifest_path)

    manifest = _load_manifest(manifest_path)
    experiment = cfg.get("experiment")
    configured_dataset = (
        experiment.get("dataset") if isinstance(experiment, Mapping) else None
    )
    if _normalized_name(configured_dataset) != _normalized_name(manifest.get("dataset")):
        raise ValueError(
            f"experiment dataset {configured_dataset!r} does not match validation "
            f"partition dataset {manifest.get('dataset')!r}"
        )
    observed_input_sha256 = dataset_preflight.get("input_file_sha256")
    if manifest.get("input_sha256") != observed_input_sha256:
        raise ValueError(
            "validation partition was frozen for input SHA-256 "
            f"{manifest.get('input_sha256')}, not {observed_input_sha256}"
        )
    if manifest.get("input_rows") != dataset_preflight.get("rows"):
        raise ValueError(
            "validation partition input row count does not match governed preflight"
        )

    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("validation partition manifest must declare partitions")
    partition = partitions.get(partition_name)
    if not isinstance(partition, Mapping):
        raise ValueError(f"manifest does not declare partition {partition_name!r}")
    selected_ids = partition.get("selected_ids")
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or not all(isinstance(row_id, str) and row_id for row_id in selected_ids)
    ):
        raise ValueError(f"partition {partition_name!r} selected_ids must be non-empty strings")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"partition {partition_name!r} contains duplicate IDs")
    if partition.get("rows") != len(selected_ids):
        raise ValueError(f"partition {partition_name!r} row count is inconsistent")
    actual_ids_sha256 = selected_ids_sha256(selected_ids)
    if partition.get("selected_ids_sha256") != actual_ids_sha256:
        raise ValueError(f"partition {partition_name!r} selected_ids SHA-256 is inconsistent")

    return {
        "valid": True,
        "manifest_path": manifest_path,
        "manifest_file_sha256": actual_manifest_sha256,
        "manifest_status": manifest["status"],
        "dataset": manifest["dataset"],
        "base_split": manifest["base_split"],
        "partition": partition_name,
        "rows": len(selected_ids),
        "selected_ids_sha256": actual_ids_sha256,
        "selected_ids": list(selected_ids),
        "selection_algorithm": manifest.get("selection_algorithm"),
        "seed": manifest.get("seed"),
    }


def iter_partition_rows(
    rows: Iterable[Dict[str, Any]],
    partition_preflight: Mapping[str, Any] | None,
) -> Iterator[Dict[str, Any]]:
    """Yield a frozen partition in canonical source order and fail on drift."""

    if partition_preflight is None:
        yield from rows
        return

    selected_ids = partition_preflight.get("selected_ids")
    if not isinstance(selected_ids, list):
        raise ValueError("partition preflight is missing selected_ids")
    selected = set(selected_ids)
    observed: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"input row {line_number} has no valid id")
        if row_id in selected:
            if row_id in observed:
                raise ValueError(f"input contains duplicate selected id {row_id!r}")
            observed.add(row_id)
            yield row
    missing = selected - observed
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(
            f"validation partition is missing {len(missing)} IDs from input; first={preview}"
        )
    expected_rows = partition_preflight.get("rows")
    if len(observed) != expected_rows:
        raise ValueError(
            f"validation partition yielded {len(observed)} rows, expected {expected_rows}"
        )


def partition_report_for_artifact(
    partition_preflight: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    """Drop the large ID list while preserving auditable partition identity."""

    if partition_preflight is None:
        return None
    return {
        key: value
        for key, value in partition_preflight.items()
        if key != "selected_ids"
    }


def freeze_validation_partition_manifest(
    input_path: Path,
    *,
    dataset: str,
    seed: int,
    dev_fraction: float = 0.70,
) -> Dict[str, Any]:
    """Create a reference-blind 70/30 manifest from canonical validation IDs."""

    if not dataset.strip():
        raise ValueError("dataset must be non-empty")
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between zero and one")

    # Local import keeps the reusable runtime partition module lightweight.
    from src.utils.io import read_jsonl

    row_ids: list[str] = []
    seen: set[str] = set()
    for line_number, row in enumerate(read_jsonl(str(input_path)), start=1):
        if row.get("split") != "validation":
            raise ValueError(
                f"row {line_number} belongs to {row.get('split')!r}; only validation may be partitioned"
            )
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"input row {line_number} has no valid id")
        if row_id in seen:
            raise ValueError(f"duplicate input id: {row_id}")
        seen.add(row_id)
        row_ids.append(row_id)
    if len(row_ids) < 2:
        raise ValueError("at least two validation rows are required")

    ranked = sorted(
        row_ids,
        key=lambda row_id: (
            hashlib.sha256(f"{seed}\0{row_id}".encode("utf-8")).hexdigest(),
            row_id,
        ),
    )
    dev_rows = round(len(row_ids) * dev_fraction)
    dev_members = set(ranked[:dev_rows])
    ordered_dev = [row_id for row_id in row_ids if row_id in dev_members]
    ordered_dev_test = [row_id for row_id in row_ids if row_id not in dev_members]
    if set(ordered_dev) & set(ordered_dev_test):
        raise AssertionError("development partitions overlap")
    if len(ordered_dev) + len(ordered_dev_test) != len(row_ids):
        raise AssertionError("development partitions do not cover the input")

    return {
        "manifest_schema_version": PARTITION_SCHEMA_VERSION,
        "status": PARTITION_STATUS,
        "purpose": "reference_blind_primary_validation_dev_dev-test_partition",
        "dataset": dataset,
        "base_split": "validation",
        "input_path": input_path.as_posix(),
        "input_sha256": sha256_file(str(input_path)),
        "input_rows": len(row_ids),
        "selection_algorithm": (
            "lowest_sha256(seed\\0row_id) to nearest-integer dev_fraction; "
            "IDs stored in canonical input order"
        ),
        "seed": seed,
        "requested_dev_fraction": dev_fraction,
        "partitions": {
            "dev": {
                "role": "unlimited_configuration_search",
                "rows": len(ordered_dev),
                "fraction": len(ordered_dev) / len(row_ids),
                "selected_ids_sha256": selected_ids_sha256(ordered_dev),
                "selected_ids": ordered_dev,
            },
            "dev-test": {
                "role": "single_evaluation_per_candidate_configuration",
                "rows": len(ordered_dev_test),
                "fraction": len(ordered_dev_test) / len(row_ids),
                "selected_ids_sha256": selected_ids_sha256(ordered_dev_test),
                "selected_ids": ordered_dev_test,
            },
        },
        "prohibitions": [
            "No configuration may inspect dev-test more than once.",
            "No test split may be accessed before human freeze approval.",
        ],
    }
