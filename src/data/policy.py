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

    Every current caller points ``sha256_file`` at a UTF-8 text provenance
    artifact (JSON, JSONL, YAML, plain text, or pinned Python source). In the
    structured artifacts written by this project's own tooling, a literal CR
    byte (0x0D) can only ever be a line terminator: a CR that is genuine
    string *content* must be escaped as the two-byte sequence ``\\r``
    (0x5C 0x72), never emitted as a raw control byte.
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


DEFAULT_PIN_ERRATA_PATH = "configs/pin_errata_lf_normalization.json"


def _load_pin_errata(errata_path: str) -> Dict[str, str]:
    """Load the LF-normalization errata, keyed by its legacy CRLF-era pin.

    Returns a mapping of ``legacy_crlf_sha256 -> lf_canonical_sha256``. Those
    two hashes are the entire matching contract (see :func:`verify_pin`);
    ``content_id``, ``representative_path``, and ``reference_count`` are
    validated for shape here (a malformed entry is still a hard error) but
    are documentation only and never participate in matching. A single
    legacy-era content blob is legitimately copied to well over a hundred
    different paths in this project (e.g. ``dataset_preflight.json`` is
    identical across every method's run directory under one partition), so
    the file's real path cannot be part of the safety condition without
    making the errata unable to describe that class of pin at all.

    A duplicate ``legacy_crlf_sha256`` across two entries fails loud instead
    of silently keeping only the last one: this file is hand-maintained and
    grows over time, and a silent overwrite would make one of the two
    equivalences vanish without any signal. There is no equivalent guard on
    ``lf_canonical_sha256`` -- the same current content can legitimately
    have had more than one legacy CRLF-era identity (e.g. if it was
    regenerated more than once before the write_text() fix), so repeated
    canonical values are expected, not a defect.

    Missing errata file is not an error: most callers never hit a legacy
    pin, and the errata file is deliberately excluded from the pin system
    it documents (see its own ``note`` field), so a repo checkout without
    it must still work for every pin that already matches directly.
    """

    try:
        with open(errata_path, "r", encoding="utf-8") as stream:
            errata = json.load(stream)
    except FileNotFoundError:
        return {}
    equivalences = errata.get("equivalences")
    if not isinstance(equivalences, list):
        raise ValueError(f"{errata_path}: missing or malformed 'equivalences' list")

    by_legacy_hash: Dict[str, str] = {}
    content_id_by_legacy_hash: Dict[str, str] = {}
    for entry in equivalences:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{errata_path}: equivalences entry is not an object: {entry!r}")
        legacy = entry.get("legacy_crlf_sha256")
        canonical = entry.get("lf_canonical_sha256")
        content_id = entry.get("content_id")
        representative_path = entry.get("representative_path")
        reference_count = entry.get("reference_count")
        if not (isinstance(legacy, str) and len(legacy) == 64):
            raise ValueError(
                f"{errata_path}: entry has an invalid 'legacy_crlf_sha256': {entry!r}"
            )
        if not (isinstance(canonical, str) and len(canonical) == 64):
            raise ValueError(
                f"{errata_path}: entry has an invalid 'lf_canonical_sha256': {entry!r}"
            )
        if not (isinstance(content_id, str) and content_id.strip()):
            raise ValueError(f"{errata_path}: entry has an invalid 'content_id': {entry!r}")
        if not (isinstance(representative_path, str) and representative_path.strip()):
            raise ValueError(
                f"{errata_path}: entry has an invalid 'representative_path': {entry!r}"
            )
        if not (isinstance(reference_count, int) and reference_count > 0):
            raise ValueError(
                f"{errata_path}: entry has an invalid 'reference_count': {entry!r}"
            )
        if legacy in by_legacy_hash:
            raise ValueError(
                f"{errata_path}: duplicate legacy_crlf_sha256 {legacy!r} -- "
                f"declared by both {content_id_by_legacy_hash[legacy]!r} and "
                f"{content_id!r}. Silently overwriting the first entry would "
                "make its equivalence disappear without warning; give each "
                "legacy pin its own entry or merge the two content_id "
                "descriptions by hand."
            )
        by_legacy_hash[legacy] = canonical
        content_id_by_legacy_hash[legacy] = content_id
    return by_legacy_hash


def verify_pin(
    path: str, expected: str, *, errata_path: str = DEFAULT_PIN_ERRATA_PATH
) -> str:
    """Verify ``path``'s SHA-256 against ``expected``, honoring the LF errata.

    Returns ``"pass"`` when :func:`sha256_file` already equals ``expected``
    directly -- the ordinary, current case.

    Returns ``"legacy"`` only when *both* hold: ``expected`` equals a
    recorded ``legacy_crlf_sha256`` in the errata, AND the file's actual
    (LF-normalized) hash equals that same entry's ``lf_canonical_sha256``.
    Matching is by hash pair alone, not by path -- a single legacy-era
    content blob can legitimately live at well over a hundred different
    paths (copies of the same run-input identity across method
    subdirectories), so requiring a specific path would make the errata
    unable to describe that case. The two-hash condition is still the full
    safety net: satisfying only one side is not enough to grant "legacy" --
    ``expected`` matching a legacy hash while the file's actual content
    matches neither that hash nor its recorded canonical value must fail
    loud, and a file whose content coincidentally equals some entry's
    canonical hash grants nothing unless the caller's ``expected`` is that
    same entry's legacy hash. Callers must record a ``"legacy"`` result
    distinctly from ``"pass"`` -- it means the pin predates the PR #16 CRLF
    fix, not that the file is currently pinned correctly.

    Anything else fails loud with both hashes in the message. This function
    never silently falls back to treating a mismatch as acceptable.
    """

    actual = sha256_file(path)
    if actual == expected:
        return "pass"

    errata = _load_pin_errata(errata_path)
    canonical = errata.get(expected)
    if canonical is not None and canonical == actual:
        return "legacy"

    raise ValueError(
        f"{path}: SHA-256 is {actual}, expected {expected} "
        "(not covered by errata)"
    )


def load_frozen_policy(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        policy = json.load(stream)
    if not isinstance(policy, dict):
        raise ValueError("dataset policy must be a JSON object")
    if policy.get("policy_schema_version") != "1.0":
        raise ValueError(
            f"unsupported dataset policy schema {policy.get('policy_schema_version')!r}"
        )
    allowed_statuses = {
        "frozen_before_validation_results",
        "frozen_before_test_results",
    }
    if policy.get("status") not in allowed_statuses:
        raise ValueError(
            "formal runs require a policy frozen before the governed split's results"
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
