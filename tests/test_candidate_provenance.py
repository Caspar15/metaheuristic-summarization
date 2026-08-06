"""Tests for structured candidate provenance and route ranking."""

import pytest

from src.data.schemas import build_document_example, flatten_sentence_records
from src.pipeline.candidate_builder import build_candidate_pool, build_candidate_records


def two_document_records():
    example = build_document_example(
        example_id="cluster",
        split="validation",
        documents=[
            ["Document one lead.", "Document one tail."],
            ["Document two lead.", "Document two tail."],
        ],
        references=["Reference."],
        input_mode="multi_document",
        output_mode="multi_sentence",
    )
    return flatten_sentence_records(example)


def test_document_aware_position_route_preserves_both_document_leads():
    records = two_document_records()
    candidates = build_candidate_records(
        records,
        base_scores=[0.0] * len(records),
        k=2,
        sources=["position"],
    )
    assert [candidate["original_index"] for candidate in candidates] == [0, 2]
    assert len({candidate["document_id"] for candidate in candidates}) == 2
    assert all(
        candidate["route_scores"]["position_guard"]["rank"] in (1, 2)
        for candidate in candidates
    )
    assert all(
        candidate["route_scores"]["position_guard"]["route_type"]
        == "coverage_guard_signal"
        for candidate in candidates
    )


def test_union_records_route_agreement_and_raw_scores():
    records = two_document_records()
    candidates = build_candidate_records(
        records,
        base_scores=[0.9, 0.1, 0.8, 0.2],
        k=2,
        sources=["score", "position"],
    )
    assert [candidate["original_index"] for candidate in candidates] == [0, 2]
    assert all(candidate["route_agreement"] == 2 for candidate in candidates)
    assert candidates[0]["route_scores"]["lexical"]["raw"] == pytest.approx(0.9)


def test_union_candidate_keeps_scores_from_routes_that_did_not_select_it():
    records = two_document_records()
    candidates = build_candidate_records(
        records,
        base_scores=[0.1, 0.9, 0.2, 0.8],
        k=1,
        sources=["score", "position"],
    )
    assert [candidate["original_index"] for candidate in candidates] == [0, 1]
    assert all(
        set(candidate["route_scores"]) == {"lexical", "position_guard"}
        for candidate in candidates
    )
    assert candidates[0]["selected_by_routes"] == ["position_guard"]
    assert candidates[1]["selected_by_routes"] == ["lexical"]


def test_unknown_candidate_route_fails_loudly():
    records = two_document_records()
    with pytest.raises(ValueError, match="unknown candidate route"):
        build_candidate_records(
            records,
            base_scores=[0.0] * len(records),
            k=1,
            sources=["imaginary_route"],
        )


@pytest.mark.parametrize("field", ["quota", "total"])
def test_non_positive_candidate_budgets_fail(field):
    records = two_document_records()
    kwargs = {"k": 1, "total_budget": 2}
    if field == "quota":
        kwargs["k"] = 0
    else:
        kwargs["total_budget"] = 0
    with pytest.raises(ValueError, match="budget|quota"):
        build_candidate_records(
            records,
            base_scores=[0.0] * 4,
            sources=["lexical"],
            **kwargs,
        )


def test_total_budget_uses_fused_rank_not_original_order():
    records = two_document_records()
    candidates = build_candidate_records(
        records,
        base_scores=[0.1, 0.8, 0.2, 0.9],
        k=2,
        sources=["score"],
        total_budget=2,
    )
    assert [candidate["original_index"] for candidate in candidates] == [1, 3]
    assert sorted(candidate["fused_rank"] for candidate in candidates) == [1, 2]
    assert all(candidate["inclusion_reasons"] for candidate in candidates)


def test_total_cap_never_admits_sentences_outside_route_union():
    records = two_document_records()
    pool = build_candidate_pool(
        records,
        base_scores=[0.1, 0.2, 0.3, 0.9],
        k=1,
        sources=["lexical", "position"],
        total_budget=4,
    )
    proposal_indices = {
        proposal["original_index"]
        for proposals in pool["route_proposals"].values()
        for proposal in proposals
    }
    final_indices = {record["original_index"] for record in pool["records"]}
    assert final_indices <= proposal_indices
    assert pool["allocation"]["actual_size"] == len(proposal_indices)
    assert pool["allocation"]["underfilled_by"] == 4 - len(proposal_indices)


def test_route_reservations_protect_unique_proposals_before_rrf_fill():
    example = build_document_example(
        example_id="reservation",
        split="validation",
        documents=[[f"Sentence number {index}." for index in range(6)]],
        references=["Reference."],
        input_mode="single_document",
        output_mode="multi_sentence",
    )
    records = flatten_sentence_records(example)
    pool = build_candidate_pool(
        records,
        base_scores=[0.0, 0.1, 0.2, 0.3, 0.8, 0.9],
        k=3,
        sources=["lexical", "position"],
        min_per_route=2,
        total_budget=4,
    )
    final_indices = {record["original_index"] for record in pool["records"]}
    assert final_indices == {0, 1, 4, 5}
    assert pool["allocation"]["selected_proposals_by_route"] == {
        "lexical": 2,
        "position_guard": 2,
    }
    assert any(
        "reserve:lexical" in record["inclusion_reasons"]
        for record in pool["records"]
    )
    assert any(
        "reserve:position_guard" in record["inclusion_reasons"]
        for record in pool["records"]
    )


def test_impossible_route_reservations_fail_loudly():
    example = build_document_example(
        example_id="impossible",
        split="validation",
        documents=[["One.", "Two.", "Three.", "Four."]],
        references=["Reference."],
        input_mode="single_document",
        output_mode="multi_sentence",
    )
    records = flatten_sentence_records(example)
    with pytest.raises(ValueError, match="cannot fit"):
        build_candidate_pool(
            records,
            base_scores=[0.1, 0.2, 0.8, 0.9],
            k=2,
            sources=["lexical", "position"],
            min_per_route=2,
            total_budget=3,
        )


def test_document_guard_reserves_one_candidate_per_document():
    records = two_document_records()
    candidates = build_candidate_records(
        records,
        base_scores=[0.9, 0.8, 0.2, 0.1],
        k=1,
        sources=["score"],
        total_budget=2,
        coverage_guard={"enabled": True, "document": True},
    )
    assert {candidate["document_id"] for candidate in candidates} == {
        record["document_id"] for record in records
    }
    assert sum(
        "guard:document" in candidate["inclusion_reasons"]
        for candidate in candidates
    ) == 2


def test_semantic_route_scores_full_input_and_records_model_metadata(monkeypatch):
    records = two_document_records()
    calls = []

    def fake_encoder_route_scores(sentences, **kwargs):
        calls.append((list(sentences), kwargs))
        return [0.1, 0.9, 0.8, 0.2], {
            "model_name": kwargs["model_name"],
            "model_revision": "fake-commit",
            "max_model_tokens": kwargs["max_model_tokens"],
            "effective_max_model_tokens": kwargs["max_model_tokens"],
            "batch_size": kwargs["batch_size"],
            "estimated_cost": {
                "encoded_sentences": len(sentences),
                "input_tokens_before_truncation": 20,
            },
            "truncated_sentences": 1,
            "truncation_rate": 0.25,
        }

    monkeypatch.setattr(
        "src.pipeline.candidate_builder.encoder_route_scores",
        fake_encoder_route_scores,
    )
    candidates = build_candidate_records(
        records,
        base_scores=[0.0] * 4,
        k=1,
        sources=["plm"],
        route_config={
            "semantic": {
                "model_name": "sentence-transformers/fake",
                "batch_size": 2,
                "max_model_tokens": 64,
            }
        },
    )
    assert len(calls) == 1
    assert len(calls[0][0]) == 4
    assert [candidate["original_index"] for candidate in candidates] == [1]
    score = candidates[0]["route_scores"]["semantic"]
    assert score["model_revision"] == "fake-commit"
    assert score["estimated_cost"]["encoded_sentences"] == 4
    assert score["metadata"]["truncation_rate"] == pytest.approx(0.25)
    assert 0.0 <= candidates[0]["fusion_normalized"] <= 1.0


def test_semantic_route_requires_explicit_checkpoint():
    records = two_document_records()
    with pytest.raises(RuntimeError, match="explicit sentence-similarity checkpoint"):
        build_candidate_records(
            records,
            base_scores=[0.0] * 4,
            k=1,
            sources=["semantic"],
        )


def test_semantic_route_accepts_verified_precomputed_scores(monkeypatch):
    records = two_document_records()

    def must_not_encode(*args, **kwargs):
        raise AssertionError("semantic route encoded a second time")

    monkeypatch.setattr(
        "src.pipeline.candidate_builder.encoder_route_scores", must_not_encode
    )
    revision = "a" * 40
    candidates = build_candidate_records(
        records,
        base_scores=[0.0] * 4,
        k=2,
        sources=["semantic"],
        route_config={
            "semantic": {
                "model_name": "sentence-transformers/fake",
                "revision": revision,
            }
        },
        precomputed_route_data={
            "semantic": {
                "values": [0.1, 0.9, 0.8, 0.2],
                "metadata": {
                    "model_name": "sentence-transformers/fake",
                    "model_revision": revision,
                    "pooling": "attention_mask_mean",
                    "normalize_embeddings": True,
                    "estimated_cost": {"encoded_sentences": 4},
                },
            }
        },
    )
    assert [candidate["original_index"] for candidate in candidates] == [1, 2]
    assert all(
        candidate["route_scores"]["semantic"]["metadata"][
            "normalize_embeddings"
        ]
        is True
        for candidate in candidates
    )


def test_precomputed_semantic_revision_mismatch_fails_loudly():
    records = two_document_records()
    with pytest.raises(RuntimeError, match="revision does not match"):
        build_candidate_records(
            records,
            base_scores=[0.0] * 4,
            k=1,
            sources=["semantic"],
            route_config={
                "semantic": {
                    "model_name": "sentence-transformers/fake",
                    "revision": "a" * 40,
                }
            },
            precomputed_route_data={
                "semantic": {
                    "values": [0.1, 0.9, 0.8, 0.2],
                    "metadata": {
                        "model_name": "sentence-transformers/fake",
                        "model_revision": "b" * 40,
                    },
                }
            },
        )


def test_graph_route_defaults_to_bounded_sparse_knn():
    records = two_document_records()
    candidates = build_candidate_records(
        records,
        base_scores=[0.0] * 4,
        k=2,
        sources=["graph"],
        route_config={
            "graph": {"n_neighbors": 1, "min_similarity": 0.0}
        },
    )
    assert len(candidates) == 2
    graph_score = candidates[0]["route_scores"]["graph"]
    assert graph_score["model_revision"] == "sparse_tfidf_knn_textrank:v1"
    assert graph_score["metadata"]["bounded_sparse"] is True
    assert graph_score["estimated_cost"]["stored_edges"] <= 8
