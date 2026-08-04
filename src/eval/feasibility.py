"""Shared validation for the prediction-row feasibility schema."""

from __future__ import annotations

from typing import Any, Mapping, Tuple


LEGACY_SCHEMA_MESSAGE = (
    "prediction row {row_id!r} has no top-level 'feasible' field. This artifact "
    "predates the F-17 feasibility schema (see CODE_AUDIT_IEEE_Access.md); rerun "
    "it with the current pipeline, or pass --assume-legacy-feasible to explicitly "
    "assume every row in this artifact is feasible."
)


def classify_feasibility_row(
    row: Mapping[str, Any],
    *,
    assume_legacy_feasible: bool,
) -> Tuple[bool, bool]:
    """Return ``(feasible, legacy_assumption_used)`` for one prediction row.

    JSON booleans are required for the F-17 schema.  Truthy strings, ``null``,
    numbers, and silently missing fields must never be interpreted as a
    scientific feasibility judgement.
    """

    row_id = row.get("id")
    if "feasible" not in row:
        if not assume_legacy_feasible:
            raise ValueError(LEGACY_SCHEMA_MESSAGE.format(row_id=row_id))
        return True, True
    value = row["feasible"]
    if type(value) is not bool:
        raise ValueError(
            f"prediction row {row_id!r} has non-boolean 'feasible' value "
            f"{value!r}; expected JSON true or false"
        )
    return value, False
