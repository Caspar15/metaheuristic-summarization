"""Runtime enforcement for frozen canonical dataset policies."""

from __future__ import annotations

import json
import hashlib
import os
import re
from typing import Any, Dict, Mapping

from src.data.validate_dataset import validate_jsonl


_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str) -> str:
    """Hash a text provenance artifact after normalizing CRLF to LF.

    Every artifact this project points at ``sha256_file`` is JSON, JSONL,
    or YAML written by this project's own tooling. In any such file, a
    literal CR byte (0x0D) can only ever be a line terminator: a CR that is
    genuine string *content* must be JSON/YAML-escaped as the two-byte
    sequence ``\\r`` (0x5C 0x72), never emitted as a raw control byte.
    ``pathlib.Path.write_text()`` performs universal-newline translation by
    default, so the same call writes ``\n`` on Linux/macOS but ``\r\n`` on
    Windows (PR #16's CRLF pin audit: the value frozen into every
    downstream ``manifest_sha256``/``PREREGISTRATION_SHA256`` was computed
    from a Windows-written CRLF file, while a checkout of the same content
    on any other platform is LF-only). Normalizing here makes the digest
    depend only on content, not on which OS happened to write the file,
    without altering what the JSON/YAML actually says.

    A binary artifact (an archive, checkpoint, or embedding cache) may
    contain a genuine ``\r\n`` byte pair that is not a line ending, so it
    must go through :func:`sha256_binary_file` instead. A NUL byte cannot
    occur in a legitimate UTF-8 text artifact this project produces, so its
    presence means the caller pointed this function at something binary;
    that fails loud here rather than silently returning a normalized (and
    wrong) digest for it.
    """

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        carry = b""
        while True:
            chunk = stream.read(_CHUNK_SIZE)
            if not chunk:
                break
            chunk = carry + chunk
            if b"\x00" in chunk:
                raise ValueError(
                    f"{path}: contains a NUL byte, so it cannot be a "
                    "legitimate UTF-8 text artifact; use sha256_binary_file() "
                    "instead of sha256_file() for binary artifacts"
                )
            if chunk.endswith(b"\r"):
                carry = b"\r"
                chunk = chunk[:-1]
            else:
                carry = b""
            digest.update(chunk.replace(b"\r\n", b"\n"))
        if carry:
            digest.update(carry)
    return digest.hexdigest()


def sha256_binary_file(path: str) -> str:
    """Hash a binary artifact's raw bytes with no normalization.

    Use this for archives, model checkpoints, embedding caches, or any
    artifact where a ``\r\n`` byte pair might be genuine binary content
    rather than a text line ending. Unlike :func:`sha256_file`, this never
    rejects NUL bytes and never rewrites the digested bytes.
    """

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_policy(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        policy = json.load(stream)
    if not isinstance(policy, dict):
        raise ValueError("dataset policy must be a JSON object")
    if policy.get("policy_schema_version") != "1.0":
        raise ValueError(
            f"unsupported dataset policy schema {policy.get('policy_schema_version')!r}"
        )
    if policy.get("status") != "frozen_before_validation_results":
        raise ValueError(
            "formal runs require a policy frozen before validation results"
        )
    if not isinstance(policy.get("analyses"), dict):
        raise ValueError("dataset policy must declare analyses")
    return policy


def _require_file_identity(path: str, expected_sha256: str, label: str) -> str:
    if not os.path.isfile(path):
        raise ValueError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 is {actual}, expected {expected_sha256}"
        )
    return actual


def validate_dataset_policy_request(
    cfg: Mapping[str, Any],
    input_path: str,
    requested_split: str,
) -> Dict[str, Any] | None:
    """Validate the full input artifact before a governed run creates output."""

    experiment = cfg.get("experiment")
    if experiment is None:
        return None
    policy_cfg = cfg.get("data_policy")
    if not isinstance(policy_cfg, Mapping):
        raise ValueError("governed experiments require a data_policy object")
    policy_path = policy_cfg.get("policy_path")
    expected_policy_sha256 = policy_cfg.get("policy_sha256")
    analysis_name = policy_cfg.get("analysis")
    if not isinstance(policy_path, str) or not policy_path.strip():
        raise ValueError("data_policy.policy_path must be a non-empty path")
    if (
        not isinstance(expected_policy_sha256, str)
        or len(expected_policy_sha256) != 64
    ):
        raise ValueError("data_policy.policy_sha256 must be a SHA-256 digest")
    if not isinstance(analysis_name, str) or not analysis_name.strip():
        raise ValueError("data_policy.analysis must be a non-empty string")

    policy_file_sha256 = _require_file_identity(
        policy_path,
        expected_policy_sha256,
        "frozen dataset policy",
    )
    policy = load_frozen_policy(policy_path)
    dataset = policy.get("dataset", {})
    configured_dataset = (
        experiment.get("dataset") if isinstance(experiment, Mapping) else None
    )
    normalized_configured = re.sub(
        r"[^a-z0-9]+", "", str(configured_dataset or "").lower()
    )
    normalized_policy = re.sub(
        r"[^a-z0-9]+", "", str(dataset.get("name") or "").lower()
    )
    if not normalized_configured or normalized_configured != normalized_policy:
        raise ValueError(
            f"experiment dataset {configured_dataset!r} does not match frozen "
            f"policy dataset {dataset.get('name')!r}"
        )
    if dataset.get("split") != requested_split:
        raise ValueError(
            f"dataset policy is frozen for split {dataset.get('split')!r}, "
            f"not {requested_split!r}"
        )
    analyses = policy["analyses"]
    if analysis_name not in analyses:
        raise ValueError(
            f"analysis {analysis_name!r} is not declared by policy "
            f"{policy.get('policy_id')!r}"
        )
    analysis = analyses[analysis_name]
    if not isinstance(analysis, Mapping):
        raise ValueError(f"policy analysis {analysis_name!r} must be an object")

    report = validate_jsonl(
        input_path,
        expected_split=requested_split,
        expected_rows=int(analysis["expected_rows"]),
        expected_dataset_revision=str(dataset["dataset_revision"]),
        expected_dataset_name=str(dataset["name"]),
        expected_dataset_fingerprint=str(
            analysis["expected_dataset_fingerprint"]
        ),
        expected_replacement_rows=int(analysis["expected_replacement_rows"]),
        expected_replacement_characters=int(
            analysis["expected_replacement_characters"]
        ),
        allow_replacement_character=bool(
            analysis["allow_replacement_character"]
        ),
    )
    if not report["valid"]:
        raise ValueError(
            f"dataset violates frozen policy {policy.get('policy_id')!r}: "
            f"{report['errors']}"
        )
    input_sha256 = _require_file_identity(
        input_path,
        str(analysis["expected_file_sha256"]),
        "input dataset",
    )

    replacement_manifest = policy.get("replacement_character_manifest", {})
    replacement_manifest_path = replacement_manifest.get("path")
    replacement_manifest_sha = _require_file_identity(
        str(replacement_manifest_path),
        str(replacement_manifest.get("file_sha256")),
        "replacement-character manifest",
    )
    exclusion = policy.get("canonical_exclusions", {})
    exclusion_path = exclusion.get("manifest_path")
    exclusion_sha = None
    if exclusion_path is not None:
        exclusion_sha = _require_file_identity(
            str(exclusion_path),
            str(exclusion.get("manifest_file_sha256")),
            "canonical exclusion manifest",
        )

    return {
        "valid": True,
        "policy_id": policy["policy_id"],
        "policy_path": policy_path,
        "policy_file_sha256": policy_file_sha256,
        "analysis": analysis_name,
        "analysis_role": analysis["role"],
        "input_path": input_path,
        "input_file_sha256": input_sha256,
        "dataset_fingerprint": report["dataset_fingerprint"],
        "rows": report["rows"],
        "dataset_revision": dataset["dataset_revision"],
        "replacement_character_rows": report["health"][
            "unicode_replacement_rows"
        ],
        "replacement_characters": report["health"][
            "unicode_replacement_characters"
        ],
        "replacement_manifest_file_sha256": replacement_manifest_sha,
        "canonical_exclusion_manifest_file_sha256": exclusion_sha,
    }
