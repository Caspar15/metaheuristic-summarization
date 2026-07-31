"""Golden tests for the optimizer-independent objective contract."""

import numpy as np
import pytest

from src.models.extractive.greedy import greedy_select
from src.models.extractive.grasp import grasp_select
from src.objectives.evaluator import (
    ObjectiveWeights,
    SelectionConstraints,
    SelectionObjective,
    maximum_feasible_words,
    resolve_effective_min_words,
    resolve_selection_eligibility,
)


def test_maximum_feasible_words_respects_indivisible_sentences():
    sentences = ["x " * 130, "y " * 130]
    assert maximum_feasible_words(
        sentences, max_length=250, length_unit="words"
    ) == 130


def test_maximum_feasible_words_respects_sentence_cap():
    sentences = ["x " * 100, "y " * 90, "z " * 80]
    assert maximum_feasible_words(
        sentences,
        max_length=250,
        length_unit="words",
        max_sentences=2,
    ) == 190


def test_resolve_effective_min_words_no_relaxation_when_capacity_suffices():
    result = resolve_effective_min_words(
        ["x " * 10, "y " * 10, "z " * 10, "w " * 10],
        requested_min_words=15,
        max_length=25,
        length_unit="words",
    )
    assert result.requested_min_words == 15
    assert result.source_capacity_words == 20
    assert result.effective_min_words == 15
    assert result.min_words_relaxed is False
    assert result.relaxation_reason is None


def test_resolve_effective_min_words_relaxes_to_source_capacity():
    result = resolve_effective_min_words(
        ["x " * 10, "y " * 10, "z " * 10],
        requested_min_words=200,
        max_length=250,
        length_unit="words",
    )
    assert result.source_capacity_words == 30
    assert result.effective_min_words == 30
    assert result.min_words_relaxed is True
    assert result.relaxation_reason == "source_intrinsic_capacity"


def test_resolve_selection_eligibility_excludes_oversized_sentences():
    sentences = ["x " * 30, "y " * 10, "z " * 10]
    records = [{"sentence_id": f"s{i}"} for i in range(3)]
    eligibility = resolve_selection_eligibility(
        sentences,
        records,
        max_length=25,
        length_unit="words",
        require_nonempty=True,
    )
    assert eligibility.eligible_indices == [1, 2]
    assert eligibility.ineligible_sentences == [
        {
            "sentence_id": "s0",
            "original_index": 0,
            "word_count": 30,
            "reason": "exceeds_active_output_budget",
        }
    ]


def test_resolve_selection_eligibility_raises_when_nothing_fits():
    sentences = ["x " * 30]
    records = [{"sentence_id": "s0"}]
    with pytest.raises(ValueError, match="no sentence eligible"):
        resolve_selection_eligibility(
            sentences,
            records,
            max_length=25,
            length_unit="words",
            require_nonempty=True,
            document_id="doc1",
        )


def _golden_evaluator(**constraint_overrides):
    constraints = {
        "length_unit": "words",
        "max_length": 4,
        "min_words": 3,
        "max_sentences": 2,
        "require_nonempty": True,
    }
    constraints.update(constraint_overrides)
    return SelectionObjective(
        ["alpha beta", "gamma", "delta epsilon zeta"],
        [0.9, 0.6, 0.3],
        np.array(
            [
                [1.0, 0.2, 0.4],
                [0.2, 1.0, 0.5],
                [0.4, 0.5, 1.0],
            ]
        ),
        importance_aggregation="mean",
        coverage_method="max",
        weights=ObjectiveWeights(1.0, 0.8, 0.7),
        constraints=SelectionConstraints(**constraints),
    )


def test_hand_computed_objective_and_constraints():
    result = _golden_evaluator().evaluate([0, 1])

    assert result.salience == pytest.approx(0.75)
    assert result.facility_coverage == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert result.redundancy == pytest.approx(0.2)
    assert result.scalar_utility == pytest.approx(
        0.75 + 0.8 * ((1.0 + 1.0 + 0.5) / 3) - 0.7 * 0.2
    )
    assert result.selected_words == 3
    assert result.selected_sentences == 2
    assert result.coverage_universe_size == 3
    assert result.feasible
    assert all(value <= 0 for value in result.violations.values())


def test_facility_coverage_can_use_full_source_universe():
    evaluator = SelectionObjective(
        ["candidate zero", "candidate one"],
        [0.9, 0.8],
        np.array([[1.0, 0.2], [0.2, 1.0]]),
        coverage_matrix=np.array(
            [
                [1.0, 0.2],
                [0.2, 1.0],
                [0.7, 0.1],
                [0.6, 0.3],
            ]
        ),
        importance_aggregation="mean",
        weights=ObjectiveWeights(1.0, 0.8, 0.7),
        constraints=SelectionConstraints(max_length=10),
    )
    result = evaluator.evaluate([0])
    assert result.facility_coverage == pytest.approx((1.0 + 0.2 + 0.7 + 0.6) / 4)
    assert result.coverage_universe_size == 4


def test_empty_and_too_short_subsets_are_explicitly_infeasible():
    evaluator = _golden_evaluator()
    empty = evaluator.evaluate([])
    short = evaluator.evaluate([1])

    assert not empty.feasible
    assert empty.violations["nonempty"] > 0
    assert empty.violations["min_words"] > 0
    assert not short.feasible
    assert short.violations["min_words"] > 0


def test_greedy_satisfies_lower_and_upper_bounds():
    evaluator = _golden_evaluator()
    selected = greedy_select(
        evaluator.sentences,
        evaluator.importance.tolist(),
        evaluator.similarity_matrix,
        4,
        evaluator=evaluator,
    )
    assert evaluator.evaluate(selected).feasible


def test_greedy_fails_loudly_when_no_feasible_subset_exists():
    evaluator = _golden_evaluator(max_length=2, min_words=3)
    with pytest.raises(ValueError, match="infeasible summary"):
        greedy_select(
            evaluator.sentences,
            evaluator.importance.tolist(),
            evaluator.similarity_matrix,
            2,
            evaluator=evaluator,
        )


def test_grasp_is_seed_deterministic_under_shared_objective():
    evaluator = _golden_evaluator()
    kwargs = dict(
        sentences=evaluator.sentences,
        base_scores=evaluator.importance.tolist(),
        sim_mat=evaluator.similarity_matrix,
        max_tokens=4,
        evaluator=evaluator,
        seed=2024,
        iters=8,
    )
    first = grasp_select(**kwargs)
    second = grasp_select(**kwargs)
    assert first == second
    assert evaluator.evaluate(first).feasible


def test_nsga_problem_uses_same_golden_values_and_constraints():
    pytest.importorskip("pymoo")
    from src.models.extractive.nsga2 import SummarizationProblem

    evaluator = _golden_evaluator()
    out = {}
    SummarizationProblem(evaluator)._evaluate(np.array([1, 1, 0]), out)

    expected = evaluator.evaluate([0, 1])
    assert out["F"] == pytest.approx(
        [-expected.salience, -expected.facility_coverage, expected.redundancy]
    )
    assert out["G"] == pytest.approx(evaluator.inequality_constraints([0, 1]))


def test_nsga2_same_seed_returns_same_indices():
    pytest.importorskip("pymoo")
    from src.models.extractive.nsga2 import nsga2_select

    evaluator = _golden_evaluator()
    kwargs = dict(
        sentences=evaluator.sentences,
        importance=evaluator.importance.tolist(),
        sim_mat=evaluator.similarity_matrix,
        max_tokens=4,
        evaluator=evaluator,
        pop_size=12,
        n_gen=8,
        seed=2024,
    )
    first_diagnostics = {}
    second_diagnostics = {}
    first = nsga2_select(**kwargs, diagnostics=first_diagnostics)
    second = nsga2_select(**kwargs, diagnostics=second_diagnostics)
    assert first == second
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["pareto_size"] > 0
    assert first_diagnostics["search"] == {
        "population_size": 12,
        "generations": 8,
        "seed": 2024,
        "sampling": "BinaryRandomSampling",
        "crossover": "TwoPointCrossover",
        "mutation": "BitflipMutation",
        "eliminate_duplicates": True,
    }
    assert first_diagnostics["selection_weights"] == {
        "salience": 1.0,
        "facility_coverage": 0.8,
        "redundancy": 0.7,
    }
    selected_row = first_diagnostics["pareto_front"][
        first_diagnostics["selected_pareto_row"]
    ]
    assert selected_row["selected_indices"] == first
    assert evaluator.evaluate(first).feasible
