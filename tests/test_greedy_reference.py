from types import SimpleNamespace

import pytest

from src.data.schemas import build_document_example
from src.eval.oracle import (
    greedy_reference_report,
    greedy_reference_run,
    greedy_reference_summary,
)


def _canonical_example():
    return build_document_example(
        example_id="validation-example",
        split="validation",
        documents=[["alpha gamma beta delta", "alpha beta x y"]],
        references=["alpha beta gamma delta"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="Toy",
    )


def test_canonical_documents_are_flattened_instead_of_silently_scoring_zero():
    run = greedy_reference_run(
        [_canonical_example()], target_metric="rouge1", max_sentences=1
    )
    assert run["selections"][0]["selected_indices"] == [0]
    assert run["scores"]["rouge1"] > 0.0


def test_metric_targets_are_independent_searches():
    report = greedy_reference_report([_canonical_example()], max_sentences=1)
    runs = report["optimization_targets"]
    assert runs["rouge1"]["selections"][0]["selected_indices"] == [0]
    assert runs["rouge2"]["selections"][0]["selected_indices"] == [1]
    assert set(runs) == {"rouge1", "rouge2", "rougeLsum"}
    assert report["exact_upper_bound"] is False


@pytest.mark.parametrize(
    "bad_row, message",
    [
        ({"id": "missing-source", "references": ["reference"]}, "no source sentences"),
        ({"id": "missing-reference", "sentences": ["source"]}, "no non-empty references"),
        ({"id": "bad-source", "sentences": "not-a-list", "reference": "r"}, "schema"),
    ],
)
def test_schema_mismatch_fails_loud(bad_row, message):
    with pytest.raises(ValueError, match=message):
        greedy_reference_run([bad_row])


def test_invalid_metric_and_budget_fail_loud():
    with pytest.raises(ValueError, match="unsupported metrics"):
        greedy_reference_summary(["source"], "reference", metric="rougeL")
    with pytest.raises(ValueError, match="max_words must be positive"):
        greedy_reference_summary(["source"], "reference", max_words=0)


def test_search_scores_the_same_source_order_that_it_returns():
    class FakeScorer:
        def score(self, _reference, prediction):
            values = {
                "early": 0.4,
                "late": 0.6,
                "early late": 0.8,
                "late early": 0.5,
            }
            return {"rouge1": SimpleNamespace(fmeasure=values[prediction])}

    assert greedy_reference_summary(
        ["early", "late"],
        "reference",
        metric="rouge1",
        scorer=FakeScorer(),
    ) == [0, 1]
