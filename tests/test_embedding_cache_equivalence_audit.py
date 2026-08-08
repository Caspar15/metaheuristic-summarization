"""Structural and exact-comparison tests for the F-51 dev-only audit."""

from scripts.audit import verify_embedding_cache_equivalence as audit


def _row(row_id="a"):
    return {
        "id": row_id,
        "selected_indices": [1, 3],
        "summary": "same summary",
        "feasible": True,
        "infeasible_code": None,
        "baseline_diagnostics": {
            "eligible_original_indices_sha256": "1" * 64,
            "centroid_relevance_sha256": "2" * 64,
            "similarity_sha256": "3" * 64,
        },
    }


def test_f51_protocol_is_hash_pinned_and_dev_only():
    protocol = audit._load_protocol()
    assert protocol["partition"] == "dev"
    assert protocol["dev_test_accessed"] is False
    assert protocol["test_split_prohibited"] is True
    assert protocol["cached_rerun"]["cold_populate_run"] != protocol[
        "cached_rerun"
    ]["warm_hit_run"]


def test_f51_exact_row_comparison_passes_identical_rows():
    result = audit._compare([_row()], [_row()], ["a"])
    assert result["all_row_contracts_exact"] is True
    assert set(result["mismatch_counts"].values()) == {0}


def test_f51_exact_row_comparison_reports_selection_mismatch():
    changed = _row()
    changed["selected_indices"] = [2]
    result = audit._compare([_row()], [changed], ["a"])
    assert result["all_row_contracts_exact"] is False
    assert result["mismatch_counts"]["selected_indices"] == 1
    assert result["mismatch_examples_first_20"]["selected_indices"] == ["a"]
