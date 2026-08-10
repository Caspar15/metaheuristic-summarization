import json
from pathlib import Path

import pytest

from scripts.audit.effective_tunable_inventory import validate_inventory


def test_checked_in_inventory_is_complete_and_test_blind():
    result = validate_inventory(
        Path("docs/research/evidence/d1_effective_tunable_inventory.json")
    )
    assert result["group_count"] >= 15
    assert result["path_count"] >= 60
    assert result["test_split_accessed"] is False


def test_duplicate_parameter_path_fails_loudly(tmp_path):
    path = tmp_path / "inventory.json"
    groups = []
    required = [
        "governance", "tf_isf", "position", "feature_fusion",
        "representation", "length", "objectives", "candidate_switches",
        "candidate_budget", "route_gate", "graph_route", "semantic_route",
        "coverage_guards", "selector_inputs", "selector",
    ]
    for index, group in enumerate(required):
        groups.append({"group": group, "paths": [f"x.p{index}"], "disposition": "fixed"})
    groups[-1]["paths"] = ["x.p0"]
    path.write_text(json.dumps({"groups": groups}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate path"):
        validate_inventory(path)
