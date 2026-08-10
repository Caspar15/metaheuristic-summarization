import numpy as np

from scripts.audit.run_length_contract_study import (
    _candidate_macro_rows,
    _dependency_versions,
    _logical_candidate_hash,
    _select_protocol,
    _selected_indices_digest,
)


def test_dependency_provenance_includes_plm_runtime():
    versions = _dependency_versions()
    for dependency in ("torch", "transformers", "tokenizers"):
        assert dependency in versions
        assert versions[dependency] != "not-installed"


def test_logical_candidate_hash_excludes_partition_by_construction():
    base = {"features": {"weights": {"importance": 1.0}}}
    first = _logical_candidate_hash(
        base,
        dataset="multinews",
        candidate_name="cap",
        length_contract={"min_words": 0, "max_words": 250},
    )
    second = _logical_candidate_hash(
        base,
        dataset="multinews",
        candidate_name="cap",
        length_contract={"min_words": 0, "max_words": 250},
    )
    changed = _logical_candidate_hash(
        base,
        dataset="multinews",
        candidate_name="cap",
        length_contract={"min_words": 0, "max_words": 260},
    )
    assert first == second
    assert first != changed


def test_candidate_macro_averages_all_methods_and_metrics_per_row():
    rows = {
        method: [
            {"rouge1": value, "rouge2": value, "rougeLsum": value},
            {"rouge1": value + 0.1, "rouge2": value + 0.1, "rougeLsum": value + 0.1},
        ]
        for method, value in (("lead", 0.1), ("random", 0.2), ("greedy", 0.3))
    }
    assert np.allclose(_candidate_macro_rows(rows), [0.2, 0.3])


def test_equal_protocols_apply_preregistered_tie_rule():
    result = _select_protocol(
        {
            "a": [0.2, 0.3, 0.4],
            "b": [0.2, 0.3, 0.4],
            "tie": [0.2, 0.3, 0.4],
            "d": [0.2, 0.3, 0.4],
        },
        tie_candidate="tie",
        n_resamples=100,
        seed=7,
    )
    assert result["raw_winner_significant_against_all"] is False
    assert result["selected_protocol"] == "tie"


def test_selected_indices_digest_ignores_unrelated_prediction_fields():
    first = _selected_indices_digest(
        [{"id": "x", "selected_indices": [1, 3], "summary": "A"}]
    )
    second = _selected_indices_digest(
        [{"id": "x", "selected_indices": [1, 3], "summary": "B"}]
    )
    changed = _selected_indices_digest(
        [{"id": "x", "selected_indices": [1, 4], "summary": "A"}]
    )
    assert first == second
    assert first != changed
