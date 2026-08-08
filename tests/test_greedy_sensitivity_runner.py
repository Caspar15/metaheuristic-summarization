import copy
import json
from pathlib import Path

import pytest

from scripts.audit.run_greedy_sensitivity import (
    _load_protocol,
    _logical_hash,
    _set_dotted,
    resolve_variant,
)
from src.utils.io import load_yaml


def test_preregistered_variant_count_and_no_holdout_access():
    preregistration, variants = _load_protocol()
    assert len(variants) == 27
    assert preregistration["dev_test_access"] == "none_in_screening"
    assert preregistration["test_split_prohibited"] is True


def test_parent_delta_resolution_is_one_factor_from_graph_parent():
    preregistration, variants = _load_protocol()
    base = load_yaml("configs/studies/d1/multinews_base.yaml")
    graph = resolve_variant(base, variants, "G00_lexical_graph", "Multi-News")
    top_k = resolve_variant(base, variants, "G02_route_top_k_80", "Multi-News")
    expected = copy.deepcopy(graph)
    expected["candidate_budget"]["route_top_k"] = 80
    assert top_k == expected


def test_dataset_specific_structure_guard_resolves_differently():
    _, variants = _load_protocol()
    mn_base = load_yaml("configs/studies/d1/multinews_base.yaml")
    gov_base = load_yaml("configs/studies/d1/govreport_base.yaml")
    mn = resolve_variant(mn_base, variants, "G11_dataset_structure_guard", "Multi-News")
    gov = resolve_variant(gov_base, variants, "G11_dataset_structure_guard", "GovReport")
    assert mn["coverage_guard"]["document"] is False
    assert gov["coverage_guard"]["section"] is True


def test_dotted_delta_rejects_unknown_leaf():
    with pytest.raises(ValueError, match="leaf is absent"):
        _set_dotted({"a": {"b": 1}}, "a.typo", 2)


def test_logical_hash_is_partition_independent_and_delta_sensitive():
    _, variants = _load_protocol()
    base = load_yaml("configs/studies/d1/multinews_base.yaml")
    left = resolve_variant(base, variants, "L00_base", "Multi-News")
    right = resolve_variant(base, variants, "L01_importance_mean", "Multi-News")
    left_hash = _logical_hash(
        dataset="multinews", family="lexical_objective",
        variant_id="L00_base", resolved_without_partition=left,
    )
    right_hash = _logical_hash(
        dataset="multinews", family="lexical_objective",
        variant_id="L01_importance_mean", resolved_without_partition=right,
    )
    assert left_hash != right_hash

