import copy

import pytest

from scripts.audit.run_d3a_router_fusion_screen import (
    _load_preregistration,
    resolve_candidate,
)
from src.utils.io import load_yaml


def _base(dataset="Multi-News"):
    prereg = _load_preregistration()
    return load_yaml(prereg["datasets"][dataset]["base_config"])


def test_preregistration_is_dev_only_and_complete():
    prereg = _load_preregistration()
    assert prereg["dev_test_access"] == "none"
    assert prereg["test_split_prohibited"] is True
    assert len(prereg["candidates"]) == 14
    assert prereg["execution"]["workers_requested"] == 16


def test_route_weight_candidate_changes_only_declared_route():
    prereg = _load_preregistration()
    candidate = next(row for row in prereg["candidates"] if row["id"] == "R09_semantic_x2")
    resolved = resolve_candidate(_base(), candidate)
    assert resolved["candidates"]["route_weights"] == {"semantic": 2.0}
    assert resolved["candidate_budget"] == {
        "route_top_k": 40,
        "min_per_route": 20,
        "total": 80,
    }


def test_cost_bound_fails_loudly():
    candidate = {"id": "bad", "delta": {"candidate_budget.total": 161}}
    with pytest.raises(ValueError, match="cost bound"):
        resolve_candidate(_base(), candidate)


def test_selector_drift_fails_loudly():
    base = copy.deepcopy(_base())
    base["selector"]["similarity_source"] = "sbert"
    with pytest.raises(ValueError, match="TF-IDF"):
        resolve_candidate(base, {"id": "bad", "delta": {}})
