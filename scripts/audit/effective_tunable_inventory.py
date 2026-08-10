"""Validate and summarize the versioned effective-parameter inventory.

This is deliberately a provenance check, not a source-code parser. Nested
``dict.get`` call chains cannot be reconstructed reliably from syntax alone,
so the inventory is hand-audited against the five named runtime modules and
kept reviewable as JSON. The script fails if paths are duplicated, required
runtime groups disappear, or a disposition is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_INVENTORY = Path("docs/research/evidence/d1_effective_tunable_inventory.json")
REQUIRED_GROUPS = {
    "governance", "tf_isf", "position", "feature_fusion", "representation",
    "length", "objectives", "candidate_switches", "candidate_budget",
    "route_gate", "graph_route", "semantic_route", "coverage_guards",
    "selector_inputs", "selector",
}


def validate_inventory(path: Path) -> dict:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("inventory.groups must be a non-empty list")
    names = [row.get("group") for row in groups]
    if len(names) != len(set(names)):
        raise ValueError("inventory group names must be unique")
    missing = sorted(REQUIRED_GROUPS - set(names))
    if missing:
        raise ValueError(f"inventory is missing required groups: {missing}")
    seen_paths: dict[str, str] = {}
    for row in groups:
        if not row.get("disposition"):
            raise ValueError(f"group {row.get('group')!r} has no disposition")
        paths = row.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"group {row.get('group')!r} has no paths")
        for dotted in paths:
            if dotted in seen_paths:
                raise ValueError(
                    f"duplicate path {dotted!r} in {seen_paths[dotted]!r} and "
                    f"{row['group']!r}"
                )
            seen_paths[dotted] = row["group"]
    return {
        "inventory_path": path.as_posix(),
        "inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "group_count": len(groups),
        "path_count": len(seen_paths),
        "contextual_inactivity_count": len(payload.get("contextual_inactivity", [])),
        "test_split_accessed": bool(payload.get("test_split_accessed")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()
    print(json.dumps(validate_inventory(args.inventory), indent=2))


if __name__ == "__main__":
    main()
