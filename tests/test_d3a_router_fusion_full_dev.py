import pytest

from scripts.audit.run_d3a_router_fusion_full_dev import _protocol, resolve_candidate
from src.utils.io import load_yaml


def _base(dataset="Multi-News"):
    protocol = _protocol()
    return load_yaml(protocol["datasets"][dataset]["base_config"])


def test_full_dev_protocol_has_no_protected_split_entrypoint():
    protocol = _protocol()
    assert protocol["partition"] == "dev"
    assert protocol["dev_test_access"] == "none"
    assert protocol["test_split_prohibited"] is True
    assert protocol["datasets"]["Multi-News"]["finalists"] == [
        "R00_anchor", "R01_total120", "R07_bigrams_position02", "R10_graph_x2"
    ]


def test_full_dev_resolves_task_profile_selector_unchanged():
    protocol = _protocol()
    candidate = protocol["candidates"]["R11_lexical_x05"]
    resolved = resolve_candidate(_base("GovReport"), candidate)
    assert resolved["optimizer"]["method"] == "mmr"
    assert resolved["optimizer"]["lambda_relevance"] == pytest.approx(0.7)
    assert resolved["candidates"]["route_weights"] == {"lexical": 0.5}


def test_full_dev_rejects_route_drift():
    base = _base()
    base["compute_budget"]["enabled_routes"] = ["lexical", "graph"]
    with pytest.raises(ValueError, match="route contract"):
        resolve_candidate(base, {"delta": {}})
