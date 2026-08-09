from scripts.audit.run_d3b_cross_profile_combination import _protocol, resolve_combination
from src.utils.io import load_yaml


def test_d3b_protocol_is_one_combination_per_dev_profile():
    protocol = _protocol()
    assert protocol["partition"] == "dev"
    assert protocol["dev_test_access"] == "none"
    assert protocol["test_split_prohibited"] is True
    assert protocol["measurement"]["formal_paired"].startswith("100,000")
    assert protocol["candidate"]["id"] == "C01_combined_salience_route_weight"


def test_d3b_multinews_combines_salience_and_graph_weight():
    protocol = _protocol()
    registration = protocol["datasets"]["Multi-News"]
    base = load_yaml(registration["base_config"])
    resolved = resolve_combination(base, registration)
    assert resolved["features"]["tf_isf"]["use_bigrams"] is True
    assert resolved["features"]["weights"]["position"] == 0.2
    assert resolved["candidates"]["route_weights"] == {"graph": 2.0}
    assert resolved["optimizer"]["method"] == "greedy"


def test_d3b_govreport_retains_mmr_policy_and_downweights_lexical():
    protocol = _protocol()
    registration = protocol["datasets"]["GovReport"]
    base = load_yaml(registration["base_config"])
    resolved = resolve_combination(base, registration)
    assert resolved["candidates"]["route_weights"] == {"lexical": 0.5}
    assert resolved["optimizer"]["method"] == "mmr"
    assert resolved["optimizer"]["lambda_relevance"] == 0.7
