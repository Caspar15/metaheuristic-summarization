"""Hand-computed golden tests for the Lead baseline.

Unlike the pipeline snapshot test, every expected value here is derived by
hand arithmetic before running anything, specifically to pin the two
easy-to-get-wrong behaviours: the requested-to-effective minimum-words
relaxation, and the exclusion of individually-oversized sentences from
consideration. A test that only checks "the first few sentences were picked"
would not catch either.
"""

import pytest

from src.baselines.lead import summarize_one_lead
from src.data.schemas import build_document_example
from src.objectives.evaluator import (
    ObjectiveWeights,
    SelectionConstraints,
    SelectionObjective,
)


def _words(tag: str, count: int) -> str:
    return " ".join([tag] * count)


def _doc_two_documents_ten_words_each():
    return build_document_example(
        example_id="lead_case1",
        split="validation",
        documents=[
            [_words("a", 10), _words("b", 10)],
            [_words("c", 10), _words("d", 10)],
        ],
        references=["a reference"],
        input_mode="multi_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def _doc_single_document_three_short_sentences():
    return build_document_example(
        example_id="lead_case2",
        split="validation",
        documents=[[_words("w", 10), _words("w", 10), _words("w", 10)]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def _doc_with_oversized_leading_sentence():
    return build_document_example(
        example_id="lead_case3",
        split="validation",
        documents=[[_words("x", 30), _words("y", 10), _words("z", 10)]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def _doc_three_documents_two_sentences_each():
    eight_words = _words("s", 8)
    return build_document_example(
        example_id="lead_case4",
        split="validation",
        documents=[
            [eight_words, eight_words],
            [eight_words, eight_words],
            [eight_words, eight_words],
        ],
        references=["a reference"],
        input_mode="multi_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def test_case1_document_order_normal_length_matched_case():
    """4 sentences of 10 words, budget 25/min 15: 2 fit, a 3rd would not.

    min_words is never applied to Lead (see lead.py's
    LEAD_MIN_WORDS_NOT_APPLIED_REASON), so the requested floor of 15 has no
    effect on selection here even though it would have been satisfiable.
    """

    doc = _doc_two_documents_ten_words_each()
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 25,
            "min_words": 15,
            "require_nonempty": True,
        }
    }
    result = summarize_one_lead(doc, cfg, ordering="document_order")

    # Hand computation: sentences a(10) b(10) c(10) d(10). Greedy prefix adds
    # a (10<=25), then b (20<=25), then c would make 30>25 -> stop.
    assert result["selected_indices"] == [0, 1]
    assert result["summary_sentences"] == [_words("a", 10), _words("b", 10)]

    budget = result["output_budget"]
    # Best attainable subset sum under max_length=25 from four 10-word
    # sentences is 20 (two of them; three would be 30 > 25). This is still
    # computed and recorded even though min_words is not applied.
    assert budget["source_capacity_words"] == 20
    assert budget["requested_min_words"] == 15
    assert budget["effective_min_words"] == 0
    assert budget["min_words"] == 0
    assert budget["min_words_relaxed"] is False
    assert budget["relaxation_reason"] is None
    assert budget["length_gate"] is True
    assert budget["min_words_applied"] is False
    assert budget["min_words_not_applied_reason"]
    assert budget["selected_words"] == 20

    # length_gate is True here, so requested and applied upper bounds match:
    # max_words=25 was both requested and enforced, and max_sentences was
    # never set at all (genuinely absent, not withheld).
    assert budget["max_words"] == 25
    assert budget["requested_max_words"] == 25
    assert budget["max_sentences"] is None
    assert budget["requested_max_sentences"] is None

    evaluation = result["selection_evaluation"]
    assert evaluation["selected_words"] == 20
    assert evaluation["feasible"] is True


def test_case2_document_order_does_not_apply_min_words_but_still_records_it():
    """3x10-word sentences cannot reach a 200-word floor.

    Before this baseline stopped applying min_words to Lead, this would have
    relaxed the floor down to the 30-word source capacity. Now the floor is
    not applied at all (min_words_applied is False), but requested_min_words
    (200) and source_capacity_words (30) are still recorded in the artifact
    -- turning the floor off must not make the requested value disappear.
    """

    doc = _doc_single_document_three_short_sentences()
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 250,
            "min_words": 200,
            "require_nonempty": True,
        }
    }
    result = summarize_one_lead(doc, cfg, ordering="document_order")

    # All 3 sentences fit comfortably under 250 words, so the prefix walk
    # consumes every eligible sentence: 10 + 10 + 10 = 30 words total.
    assert result["selected_indices"] == [0, 1, 2]

    budget = result["output_budget"]
    assert budget["requested_min_words"] == 200
    assert budget["source_capacity_words"] == 30
    assert budget["effective_min_words"] == 0
    assert budget["min_words"] == 0
    assert budget["min_words_relaxed"] is False
    assert budget["relaxation_reason"] is None
    assert budget["min_words_applied"] is False
    assert budget["min_words_not_applied_reason"]
    assert budget["selected_words"] == 30

    evaluation = result["selection_evaluation"]
    assert evaluation["selected_words"] == 30
    assert evaluation["feasible"] is True


def test_prefix_landing_point_unreachable_succeeds_when_min_words_not_applied():
    """190+75-word sentences under max_words=250, requested min_words=200.

    No subset of {190, 75} lands in [200, 250]: the only attainable sums
    under the 250-word cap are 0, 75, and 190. A strict reading-order prefix
    starting at the 190-word sentence can only ever reach 190 (adding the
    75-word sentence would push the total to 265 > 250, so the prefix stops
    at one sentence). Because Lead does not apply min_words, this document
    still produces a feasible, successful summary at 190 words, and the
    artifact records the un-relaxed requested floor (200) plus why it was
    not applied instead of silently dropping it.
    """

    doc = build_document_example(
        example_id="lead_case5",
        split="validation",
        documents=[[_words("x", 190), _words("y", 75)]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 250,
            "min_words": 200,
            "require_nonempty": True,
        }
    }
    result = summarize_one_lead(doc, cfg, ordering="document_order")

    assert result["selected_indices"] == [0]

    budget = result["output_budget"]
    assert budget["min_words_applied"] is False
    assert budget["requested_min_words"] == 200
    assert budget["selected_words"] == 190
    assert budget["min_words_not_applied_reason"]

    evaluation = result["selection_evaluation"]
    assert evaluation["selected_words"] == 190
    assert evaluation["feasible"] is True


def test_prefix_landing_point_would_raise_if_min_words_were_applied():
    """Pin down *why* min_words had to be removed for Lead: the identical
    190-word selection from the case above is infeasible under a
    SelectionObjective built with min_words=200 applied directly (bypassing
    Lead entirely), because 190 < 200. This is the failure mode Lead would
    hit if it applied its requested min_words like a system run does.
    """

    sentences = [_words("x", 190), _words("y", 75)]
    evaluator = SelectionObjective(
        sentences,
        [0.0, 0.0],
        None,
        weights=ObjectiveWeights(salience=0.0, facility_coverage=0.0, redundancy=0.0),
        constraints=SelectionConstraints(
            length_unit="words",
            max_length=250,
            min_words=200,
            require_nonempty=True,
        ),
    )

    with pytest.raises(ValueError, match="min_words"):
        evaluator.assert_feasible([0])


def test_case3_oversized_leading_sentence_is_excluded_not_truncated():
    """A 30-word sentence under a 25-word budget can never be selected, even
    though it is first in document order."""

    doc = _doc_with_oversized_leading_sentence()
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 25,
            "min_words": 15,
            "require_nonempty": True,
        }
    }
    result = summarize_one_lead(doc, cfg, ordering="document_order")

    # Sentence 0 (30 words) is excluded before selection ever runs; the walk
    # starts from sentence 1 (10 words) and sentence 2 (10 words): 20 <= 25.
    assert result["selected_indices"] == [1, 2]
    assert result["selection_evaluation"]["selected_words"] == 20
    assert result["selection_evaluation"]["feasible"] is True

    ineligible = result["candidate_pool"]["selection_ineligible_sentences"]
    assert len(ineligible) == 1
    assert ineligible[0]["original_index"] == 0
    assert ineligible[0]["word_count"] == 30
    assert ineligible[0]["reason"] == "exceeds_active_output_budget"


def test_case4_round_robin_differs_from_document_order_under_a_sentence_cap():
    """3 documents x 2 sentences, capped at 3 total sentences: document_order
    exhausts document 0 before touching document 1 or 2; round_robin takes
    exactly one sentence from each document instead."""

    doc = _doc_three_documents_two_sentences_each()
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 1000,  # never binds; only max_sentences does
            "max_sentences": 3,
            "min_words": 0,
            "require_nonempty": True,
        }
    }
    document_order = summarize_one_lead(doc, cfg, ordering="document_order")
    round_robin = summarize_one_lead(doc, cfg, ordering="round_robin")

    # document order: sentences are indexed 0,1 (doc0) 2,3 (doc1) 4,5 (doc2).
    # Prefix walk: add 0 (count1), add 1 (count2), add 2 (count3) -> stop,
    # since a 4th sentence would exceed max_sentences=3.
    assert document_order["selected_indices"] == [0, 1, 2]

    # round-robin: round 1 takes doc0's first (0), doc1's first (2), doc2's
    # first (4) -> already at the cap of 3, so round 2 never starts.
    assert round_robin["selected_indices"] == [0, 2, 4]

    assert document_order["selected_indices"] != round_robin["selected_indices"]
    for result in (document_order, round_robin):
        assert result["selection_evaluation"]["feasible"] is True


def test_fabbri_first_k_is_diagnostic_and_bypasses_the_length_gate():
    """fabbri_first_k reproduces Fabbri et al. 2019's own First-k definition
    (k sentences per source document) and is explicitly NOT length-matched:
    a 5-word budget would normally make every 8-word sentence ineligible,
    but this mode does not apply the word-budget gate at all."""

    doc = _doc_three_documents_two_sentences_each()
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 5,
            "min_words": 5,
            "require_nonempty": True,
        }
    }
    result = summarize_one_lead(doc, cfg, ordering="fabbri_first_k", first_k=1)

    # First-1 sentence of each of the 3 documents, in ascending document order.
    assert result["selected_indices"] == [0, 2, 4]
    assert result["output_budget"]["length_gate"] is False
    assert result["output_budget"]["max_words"] is None
    assert result["selection_evaluation"]["feasible"] is True
    assert result["objective_spec"]["method"] == "lead_fabbri_first_k"
    assert result["objective_spec"]["status"] == "baseline"

    # The length gate is off, but the config's requested min_words (5) must
    # still surface in the artifact -- not silently zeroed just because it
    # is not enforced. min_words itself stays 0 since nothing is enforced.
    assert result["output_budget"]["requested_min_words"] == 5
    assert result["output_budget"]["min_words"] == 0
    assert result["output_budget"]["min_words_applied"] is False

    # Same principle for the upper bound: max_words=5 was requested but is
    # not applied (max_words itself is None), and there was no requested
    # max_sentences at all (genuinely absent, not withheld).
    assert result["output_budget"]["requested_max_words"] == 5
    assert result["output_budget"]["requested_max_sentences"] is None


def test_fabbri_first_k_records_requested_upper_bounds_without_applying_them():
    """cfg asks for max_words=250 and min_words=5; fabbri_first_k enforces
    neither (length_gate=False), but the artifact must still show what was
    requested -- distinct from the applied fields, which are None -- plus
    the explicit length_gate=False marker, rather than silently zeroing or
    nulling out the config's request."""

    doc = _doc_three_documents_two_sentences_each()
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 250,
            "min_words": 5,
            "require_nonempty": True,
        }
    }
    result = summarize_one_lead(doc, cfg, ordering="fabbri_first_k", first_k=1)

    budget = result["output_budget"]
    assert budget["length_gate"] is False

    # Applied values: nothing is enforced under this diagnostic ordering.
    assert budget["max_words"] is None
    assert budget["max_tokens"] is None
    assert budget["max_sentences"] is None
    assert budget["min_words"] == 0

    # Requested values: preserved from length_control regardless of whether
    # this ordering applies them.
    assert budget["requested_max_words"] == 250
    assert budget["requested_max_tokens"] is None
    assert budget["requested_max_sentences"] is None
    assert budget["requested_min_words"] == 5

    # Explicit "not applied" markers, not an implicit None.
    assert budget["min_words_applied"] is False
    assert budget["min_words_not_applied_reason"]


def test_objective_spec_and_candidate_pool_are_honestly_labelled():
    doc = _doc_two_documents_ten_words_each()
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 25,
            "min_words": 15,
            "require_nonempty": True,
        }
    }
    result = summarize_one_lead(doc, cfg, ordering="document_order")

    assert result["objective_spec"]["status"] == "baseline"
    assert result["objective_spec"]["method"] == "lead_document_order"
    assert result["objective_spec"]["status"] not in {
        "task_profiled_v1",
        "legacy_unprofiled",
    }

    pool = result["candidate_pool"]
    assert pool["enabled"] is False
    assert pool["actual_size"] == 0
    assert result["candidate_records"] == []
    assert result["optimizer_diagnostics"] is None
