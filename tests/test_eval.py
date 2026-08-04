"""Golden tests for the audited evaluation semantics."""

import pytest

from src.eval.rouge import rouge_scores, score_single
from src.eval.protocol import (
    MULTISENTENCE_LSUM,
    SCITLDR_OFFICIAL,
    ProtocolUnavailableError,
    evaluate_corpus,
)
from src.pipeline.evaluate import align_evaluation_rows, plan_feasibility_scoring


def test_multireference_uses_one_reference_selected_by_rouge1():
    # Reference 1 has the higher R1; reference 2 has the higher R2.  The local
    # multi-reference primitive chooses one reference by R1 and reports every
    # metric from that same reference, rather than taking optimistic maxima.
    # It is not labeled as the official SciTLDR protocol.
    scores = score_single(
        "a b c d",
        ["a c d", "a b c x y z"],
        metrics=("rouge1", "rouge2", "rougeL"),
    )

    assert scores["rouge1"] == pytest.approx(6 / 7)
    assert scores["rouge2"] == pytest.approx(0.4)
    assert scores["rougeL"] == pytest.approx(6 / 7)


def test_prediction_reference_length_mismatch_fails():
    with pytest.raises(ValueError, match="length mismatch"):
        rouge_scores(["prediction"], [])


def test_empty_corpus_fails():
    with pytest.raises(ValueError, match="empty corpus"):
        rouge_scores([], [])


def test_explicit_multisentence_protocol_runs():
    scores = evaluate_corpus(
        ["a short prediction"],
        [["a short reference"]],
        protocol=MULTISENTENCE_LSUM,
    )
    assert "rougeLsum" in scores


def test_scitldr_cannot_be_mislabeled_official():
    with pytest.raises(ProtocolUnavailableError, match="conformance-tested"):
        evaluate_corpus(
            ["prediction"],
            [["reference one", "reference two"]],
            protocol=SCITLDR_OFFICIAL,
        )


def test_unknown_protocol_fails():
    with pytest.raises(ValueError, match="unknown evaluation protocol"):
        evaluate_corpus(["prediction"], [["reference"]], protocol="implicit_default")


def test_predictions_and_gold_are_joined_by_id_not_row_order():
    predictions, references = align_evaluation_rows(
        [{"id": "b", "summary": "prediction b"}, {"id": "a", "summary": "prediction a"}],
        [{"id": "a", "highlights": "reference a"}, {"id": "b", "highlights": "reference b"}],
    )
    assert predictions == ["prediction b", "prediction a"]
    assert references == [["reference b"], ["reference a"]]


def test_partial_gold_prediction_alignment_fails():
    with pytest.raises(ValueError, match="ID mismatch"):
        align_evaluation_rows(
            [{"id": "a", "summary": "prediction a"}],
            [{"id": "a", "highlights": "reference a"}, {"id": "b", "highlights": "reference b"}],
        )


def test_excluding_an_id_from_scoring_is_not_a_missing_prediction():
    """F-17: a row present but excluded from scoring (infeasible, under
    feasible_only) must not trip the ID-mismatch guard the way a row that is
    genuinely absent from the artifact would."""
    predictions, references = align_evaluation_rows(
        [
            {"id": "a", "summary": "prediction a"},
            {"id": "b", "summary": "prediction b"},
        ],
        [
            {"id": "a", "highlights": "reference a"},
            {"id": "b", "highlights": "reference b"},
        ],
        include_ids={"a"},
    )
    assert predictions == ["prediction a"]
    assert references == [["reference a"]]


def test_plan_feasibility_scoring_feasible_only_excludes_infeasible_ids():
    rows = [
        {"id": "a", "feasible": True},
        {"id": "b", "feasible": False},
    ]
    include_ids, stats = plan_feasibility_scoring(
        rows, feasible_only=True, assume_legacy_feasible=False
    )
    assert include_ids == {"a"}
    assert stats == {
        "total_rows": 2,
        "feasible_rows": 1,
        "infeasible_rows": 1,
        "legacy_schema_assumed_feasible_rows": 0,
        "scoring_mode": "feasible_only",
    }


def test_plan_feasibility_scoring_all_rows_mode_scores_everyone():
    rows = [{"id": "a", "feasible": True}, {"id": "b", "feasible": False}]
    include_ids, stats = plan_feasibility_scoring(
        rows, feasible_only=False, assume_legacy_feasible=False
    )
    assert include_ids is None
    assert stats["scoring_mode"] == "all_rows"
    assert stats["feasible_rows"] == 1
    assert stats["infeasible_rows"] == 1


def test_plan_feasibility_scoring_rejects_legacy_schema_by_default():
    rows = [{"id": "a"}]
    with pytest.raises(ValueError, match="assume-legacy-feasible"):
        plan_feasibility_scoring(rows, feasible_only=True, assume_legacy_feasible=False)


def test_plan_feasibility_scoring_assume_legacy_feasible_records_the_assumption():
    rows = [{"id": "a"}, {"id": "b"}]
    include_ids, stats = plan_feasibility_scoring(
        rows, feasible_only=True, assume_legacy_feasible=True
    )
    assert include_ids == {"a", "b"}
    assert stats["legacy_schema_assumed_feasible_rows"] == 2
    assert stats["feasible_rows"] == 2
