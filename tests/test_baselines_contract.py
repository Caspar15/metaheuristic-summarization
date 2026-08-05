"""Hand-computed golden tests for src.baselines.contract.select_by_score.

select_by_score is the shared budget-fill primitive for any "rank once,
walk once" baseline (TextRank, LexRank, and future rank-then-fill
methods) -- see its own docstring for why this is deliberately not
src.models.extractive.greedy.greedy_select. These tests pin the two
easy-to-get-wrong behaviours: skip-tolerant fill (a highest-scoring
sentence that does not fit must not stop the walk), and stable tie-break
on relative position when scores are equal.
"""

import numpy as np
import pytest

from src.baselines.contract import select_by_score
from src.objectives.evaluator import (
    ObjectiveWeights,
    SelectionConstraints,
    SelectionObjective,
)


def _evaluator(sentences, max_length=None, max_sentences=None):
    return SelectionObjective(
        sentences,
        np.zeros(len(sentences)),
        None,
        weights=ObjectiveWeights(salience=0.0, facility_coverage=0.0, redundancy=0.0),
        constraints=SelectionConstraints(
            length_unit="words",
            max_length=max_length,
            min_words=0,
            max_sentences=max_sentences,
            require_nonempty=True,
        ),
    )


def _words(tag: str, count: int) -> str:
    return " ".join([tag] * count)


def test_skip_tolerant_fill_does_not_stop_at_a_highest_scoring_miss():
    """4 sentences, 10 words each, budget 25. Scores make the walk order
    b(0.9), a(0.7), d(0.5), c(0.1). Hand trace: add b (10<=25), add a
    (20<=25), try d -> 30>25, skip, try c -> 30>25, skip. Final selection
    is {a, b} in original order.
    """

    records = [{"text": _words(tag, 10)} for tag in ("a", "b", "c", "d")]
    sentences = [r["text"] for r in records]
    evaluator = _evaluator(sentences, max_length=25)
    scores = [0.7, 0.9, 0.1, 0.5]  # a, b, c, d

    selected = select_by_score(records, evaluator, scores)

    assert sorted(selected) == [0, 1]


def test_skip_tolerant_fill_recovers_budget_a_stop_rule_would_waste():
    """a=12 words (score 0.9), b=10 words (score 0.5), c=6 words (score
    0.4). Budget 18. Walk order by score is a, b, c: add a (12<=18); b
    would make 22>18 -- a STOP rule (Lead's reading-order prefix) would
    halt the whole walk here at {a}=12 words. This is a SKIP rule:
    continue past b to c, 12+6=18<=18, add it. Final {a, c} = 18 words,
    strictly more than a stop rule would have reached from the same
    score order.
    """

    records = [
        {"text": _words("a", 12)},
        {"text": _words("b", 10)},
        {"text": _words("c", 6)},
    ]
    sentences = [r["text"] for r in records]
    evaluator = _evaluator(sentences, max_length=18)
    scores = [0.9, 0.5, 0.4]

    selected = select_by_score(records, evaluator, scores)

    assert sorted(selected) == [0, 2]


def test_tie_break_is_stable_on_relative_position():
    """Three equal scores: order must be preserved (0, 1, 2), not
    reshuffled by an unstable sort."""

    records = [{"text": _words(tag, 5)} for tag in ("a", "b", "c")]
    sentences = [r["text"] for r in records]
    evaluator = _evaluator(sentences, max_length=100)
    scores = [0.5, 0.5, 0.5]

    selected = select_by_score(records, evaluator, scores)

    assert selected == [0, 1, 2]


def test_mismatched_scores_length_fails_loud():
    records = [{"text": _words("a", 5)}, {"text": _words("b", 5)}]
    sentences = [r["text"] for r in records]
    evaluator = _evaluator(sentences, max_length=100)

    with pytest.raises(ValueError, match="scores length"):
        select_by_score(records, evaluator, [0.1])


def test_max_sentences_upper_bound_still_respected():
    records = [{"text": _words(tag, 5)} for tag in ("a", "b", "c")]
    sentences = [r["text"] for r in records]
    evaluator = _evaluator(sentences, max_length=100, max_sentences=1)
    scores = [0.1, 0.9, 0.5]

    selected = select_by_score(records, evaluator, scores)

    assert selected == [1]
