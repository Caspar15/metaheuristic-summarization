"""Shared objective and feasibility contract for extractive selectors.

All formal selectors must evaluate a subset through this module.  Search
procedures may differ, but the scientific meaning and scale of each objective
must not silently change with the optimizer.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from src.objectives.factory import aggregate_importance
from src.utils.tokenizer import count_tokens


def maximum_feasible_words(
    sentences: Sequence[str],
    *,
    max_length: Optional[int],
    length_unit: str,
    max_sentences: Optional[int] = None,
) -> int:
    """Return the exact largest attainable word count under upper bounds.

    Extractive sentences are indivisible, so ``min(requested, total_words)``
    is not a sufficient feasibility check.  For word/token budgets this uses
    a bounded subset-sum bitset; for a sentence-only budget, taking the longest
    allowed sentences is exact.  ``tokens`` intentionally follows the current
    pipeline contract where output tokens are whitespace-delimited words.
    """

    unit = str(length_unit).lower()
    if unit not in {"words", "tokens", "sentences"}:
        raise ValueError(f"unknown length unit: {length_unit!r}")
    if max_length is not None and int(max_length) < 1:
        raise ValueError("max_length must be positive when declared")
    if max_sentences is not None and int(max_sentences) < 1:
        raise ValueError("max_sentences must be positive when declared")

    lengths = [count_tokens(sentence) for sentence in sentences]
    if not lengths:
        return 0

    sentence_cap = None if max_sentences is None else int(max_sentences)
    if unit == "sentences":
        unit_cap = None if max_length is None else int(max_length)
        sentence_cap = (
            unit_cap
            if sentence_cap is None
            else sentence_cap if unit_cap is None else min(sentence_cap, unit_cap)
        )
        if sentence_cap is None:
            return sum(lengths)
        return sum(sorted(lengths, reverse=True)[:sentence_cap])

    if max_length is None:
        if sentence_cap is None:
            return sum(lengths)
        return sum(sorted(lengths, reverse=True)[:sentence_cap])

    word_cap = min(int(max_length), sum(lengths))
    mask = (1 << (word_cap + 1)) - 1
    eligible_lengths = [length for length in lengths if length <= word_cap]
    if sentence_cap is None:
        reachable = 1
        for length in eligible_lengths:
            reachable = (reachable | (reachable << length)) & mask
        return reachable.bit_length() - 1

    sentence_cap = min(sentence_cap, len(eligible_lengths))
    reachable_by_count = [0] * (sentence_cap + 1)
    reachable_by_count[0] = 1
    for length in eligible_lengths:
        for count in range(sentence_cap, 0, -1):
            reachable_by_count[count] |= (
                reachable_by_count[count - 1] << length
            ) & mask
    reachable = 0
    for values in reachable_by_count:
        reachable |= values
    return reachable.bit_length() - 1


@dataclass(frozen=True)
class EffectiveMinWords:
    """The result of clamping a requested lower bound to what a source can give.

    ``requested_min_words`` is the config's declared floor. ``effective_min_words``
    is the value actually enforced as ``SelectionConstraints.min_words``; it is
    only ever relaxed downward, and only because the source itself cannot
    reach the requested floor under the same upper bounds used for selection
    -- never because a candidate pool or search procedure fell short.
    """

    requested_min_words: int
    source_capacity_words: int
    effective_min_words: int
    min_words_relaxed: bool
    relaxation_reason: Optional[str]


def resolve_effective_min_words(
    sentences: Sequence[str],
    *,
    requested_min_words: int,
    max_length: Optional[int],
    length_unit: str,
    max_sentences: Optional[int] = None,
) -> EffectiveMinWords:
    """Clamp a requested minimum to the exact source-attainable capacity.

    This is the one place a per-document lower bound is relaxed. Any caller
    that needs a length-matched minimum (a system run or a baseline) must
    call this rather than re-deriving the relaxation itself, so the same
    document always yields the same effective minimum regardless of which
    method is selecting sentences.
    """

    source_capacity_words = maximum_feasible_words(
        sentences,
        max_length=max_length,
        length_unit=length_unit,
        max_sentences=max_sentences,
    )
    effective_min_words = min(int(requested_min_words), source_capacity_words)
    min_words_relaxed = effective_min_words < requested_min_words
    return EffectiveMinWords(
        requested_min_words=int(requested_min_words),
        source_capacity_words=source_capacity_words,
        effective_min_words=effective_min_words,
        min_words_relaxed=min_words_relaxed,
        relaxation_reason=(
            "source_intrinsic_capacity" if min_words_relaxed else None
        ),
    )


@dataclass(frozen=True)
class SelectionEligibility:
    """Which sentences may ever appear in a feasible extractive summary.

    ``ineligible_sentences`` retains an auditable record of every sentence
    excluded for being individually longer than the active budget, rather
    than letting it silently disappear or consume route/candidate quota.
    """

    eligible_indices: list[int]
    ineligible_sentences: list[dict]


def resolve_selection_eligibility(
    sentences: Sequence[str],
    sentence_records: Sequence[Mapping[str, Any]],
    *,
    max_length: Optional[int],
    length_unit: str,
    require_nonempty: bool,
    document_id: Any = None,
) -> SelectionEligibility:
    """Exclude sentences that can never occur in a feasible extractive summary.

    A sentence longer than the active word/token budget can never be selected
    on its own. This is the one place that exclusion is decided; a system run
    and a baseline must both call this rather than reimplementing the filter,
    so the same document yields the same eligible set regardless of method.
    """

    unit = str(length_unit).lower()
    if unit in {"words", "tokens"}:
        eligible_indices = [
            index
            for index, sentence in enumerate(sentences)
            if count_tokens(sentence) <= max_length
        ]
    else:
        eligible_indices = list(range(len(sentences)))
    eligible_set = set(eligible_indices)
    ineligible_sentences = [
        {
            "sentence_id": sentence_records[index]["sentence_id"],
            "original_index": index,
            "word_count": count_tokens(sentences[index]),
            "reason": "exceeds_active_output_budget",
        }
        for index in range(len(sentences))
        if index not in eligible_set
    ]
    if sentences and require_nonempty and not eligible_indices:
        raise ValueError(
            f"source document {document_id!r} has no sentence eligible under "
            f"the active {unit} budget {max_length}"
        )
    return SelectionEligibility(
        eligible_indices=eligible_indices,
        ineligible_sentences=ineligible_sentences,
    )


@dataclass(frozen=True)
class ObjectiveWeights:
    salience: float = 1.0
    facility_coverage: float = 0.8
    redundancy: float = 0.7

    def __post_init__(self) -> None:
        values = (self.salience, self.facility_coverage, self.redundancy)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("objective weights must be finite")
        if any(value < 0 for value in values):
            raise ValueError("objective weights must be non-negative")


@dataclass(frozen=True)
class SelectionConstraints:
    length_unit: str = "tokens"
    max_length: Optional[int] = None
    min_words: int = 0
    max_sentences: Optional[int] = None
    require_nonempty: bool = True

    def __post_init__(self) -> None:
        unit = self.length_unit.lower()
        if unit not in {"words", "tokens", "sentences"}:
            raise ValueError(f"unknown length unit: {self.length_unit!r}")
        if self.max_length is not None and self.max_length < 1:
            raise ValueError("max_length must be positive when declared")
        if self.min_words < 0:
            raise ValueError("min_words cannot be negative")
        if self.max_sentences is not None and self.max_sentences < 1:
            raise ValueError("max_sentences must be positive when declared")


@dataclass(frozen=True)
class SelectionEvaluation:
    selected_indices: list[int]
    salience: float
    facility_coverage: float
    redundancy: float
    scalar_utility: float
    selected_words: int
    selected_sentences: int
    coverage_universe_size: int
    feasible: bool
    violations: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class SelectionObjective:
    """Evaluate subsets with one objective definition and one constraint set."""

    def __init__(
        self,
        sentences: Sequence[str],
        importance: Sequence[float],
        similarity_matrix: Optional[np.ndarray],
        *,
        importance_aggregation: str = "mean",
        coverage_method: str = "max",
        weights: ObjectiveWeights = ObjectiveWeights(),
        constraints: SelectionConstraints = SelectionConstraints(),
        coverage_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.sentences = list(sentences)
        self.importance = np.asarray(importance, dtype=float)
        self.similarity_matrix = (
            None
            if similarity_matrix is None
            else np.asarray(similarity_matrix, dtype=float)
        )
        self.coverage_matrix = (
            self.similarity_matrix
            if coverage_matrix is None
            else np.asarray(coverage_matrix, dtype=float)
        )
        self.importance_aggregation = importance_aggregation
        self.coverage_method = coverage_method
        self.weights = weights
        self.constraints = constraints
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        n = len(self.sentences)
        if self.importance.shape != (n,):
            raise ValueError(
                f"importance must have shape ({n},), got {self.importance.shape}"
            )
        if not np.all(np.isfinite(self.importance)):
            raise ValueError("importance contains non-finite values")
        if self.similarity_matrix is not None:
            if self.similarity_matrix.shape != (n, n):
                raise ValueError(
                    "similarity matrix must be square and aligned with sentences"
                )
            if not np.all(np.isfinite(self.similarity_matrix)):
                raise ValueError("similarity matrix contains non-finite values")
        elif self.weights.redundancy != 0:
            raise ValueError(
                "selector requires a candidate similarity matrix when "
                "redundancy has non-zero weight"
            )
        if self.coverage_matrix is not None:
            if self.coverage_matrix.ndim != 2 or self.coverage_matrix.shape[1] != n:
                raise ValueError(
                    "coverage matrix must have shape (source_units, candidates)"
                )
            if not np.all(np.isfinite(self.coverage_matrix)):
                raise ValueError("coverage matrix contains non-finite values")
        elif self.weights.facility_coverage != 0:
            raise ValueError(
                "selector requires a source-to-candidate coverage matrix when "
                "facility coverage has non-zero weight"
            )
        if self.coverage_method not in {"max", "set", "diversity"}:
            raise ValueError(f"unknown coverage method: {self.coverage_method!r}")
        # Validate aggregation even for an empty selection.
        if self.importance_aggregation not in {
            "sum",
            "mean",
            "length_normalized",
            "single_item",
        }:
            raise ValueError(
                f"unknown importance aggregation: {self.importance_aggregation!r}"
            )

    def _indices(self, indices: Iterable[int]) -> np.ndarray:
        values = [int(index) for index in indices]
        if len(values) != len(set(values)):
            raise ValueError("selected indices must be unique")
        if any(index < 0 or index >= len(self.sentences) for index in values):
            raise IndexError("selected index is outside the candidate set")
        return np.asarray(sorted(values), dtype=int)

    def _salience(self, indices: np.ndarray) -> float:
        method = self.importance_aggregation
        if method == "single_item":
            if indices.size == 0:
                return 0.0
            if indices.size != 1:
                raise ValueError("single_item salience requires exactly one sentence")
            return float(self.importance[indices[0]])
        return aggregate_importance(
            self.importance, indices, self.sentences, method
        )

    def _redundancy(self, indices: np.ndarray) -> float:
        if indices.size < 2 or self.similarity_matrix is None:
            return 0.0
        selected = self.similarity_matrix[np.ix_(indices, indices)]
        upper = selected[np.triu_indices(indices.size, k=1)]
        return float(np.mean(upper))

    def _coverage(self, indices: np.ndarray, redundancy: float) -> float:
        if indices.size == 0 or self.coverage_matrix is None:
            return 0.0
        # ``set`` was historically implemented as iterative maxima and is
        # mathematically identical to ``max``.  Keep the label for compatible
        # ablations while using one auditable implementation.
        sub = self.coverage_matrix[:, indices]
        coverage = float(np.mean(np.max(sub, axis=1)))
        if self.coverage_method == "diversity":
            return coverage - 0.3 * redundancy
        return coverage

    def _violations(
        self, selected_words: int, selected_sentences: int
    ) -> dict[str, float]:
        c = self.constraints
        max_length_value = (
            selected_sentences
            if c.length_unit.lower() == "sentences"
            else selected_words
        )
        return {
            "nonempty": float(1 - selected_sentences) if c.require_nonempty else 0.0,
            "min_words": float(c.min_words - selected_words),
            "max_length": (
                float(max_length_value - c.max_length)
                if c.max_length is not None
                else 0.0
            ),
            "max_sentences": (
                float(selected_sentences - c.max_sentences)
                if c.max_sentences is not None
                else 0.0
            ),
        }

    def evaluate(self, indices: Iterable[int]) -> SelectionEvaluation:
        selected = self._indices(indices)
        salience = self._salience(selected)
        redundancy = self._redundancy(selected)
        coverage = self._coverage(selected, redundancy)
        selected_words = sum(count_tokens(self.sentences[i]) for i in selected)
        violations = self._violations(selected_words, int(selected.size))
        feasible = all(value <= 0 for value in violations.values())
        utility = (
            self.weights.salience * salience
            + self.weights.facility_coverage * coverage
            - self.weights.redundancy * redundancy
        )
        return SelectionEvaluation(
            selected_indices=selected.tolist(),
            salience=salience,
            facility_coverage=coverage,
            redundancy=redundancy,
            scalar_utility=float(utility),
            selected_words=selected_words,
            selected_sentences=int(selected.size),
            coverage_universe_size=(
                0 if self.coverage_matrix is None else self.coverage_matrix.shape[0]
            ),
            feasible=feasible,
            violations=violations,
        )

    def can_add(self, indices: Iterable[int], candidate: int) -> bool:
        selected = list(indices)
        if candidate in selected:
            return False
        if (
            self.constraints.max_sentences is not None
            and len(selected) + 1 > self.constraints.max_sentences
        ):
            return False
        evaluation = self.evaluate(selected + [candidate])
        # Lower bounds are allowed to be violated during construction.  Hard
        # upper bounds never are.
        return (
            evaluation.violations["max_length"] <= 0
            and evaluation.violations["max_sentences"] <= 0
        )

    def assert_feasible(self, indices: Iterable[int]) -> SelectionEvaluation:
        evaluation = self.evaluate(indices)
        if not evaluation.feasible:
            positive = {
                key: value
                for key, value in evaluation.violations.items()
                if value > 0
            }
            raise ValueError(
                f"selector returned an infeasible summary: {positive}"
            )
        return evaluation

    def inequality_constraints(self, indices: Iterable[int]) -> list[float]:
        """Return the shared <= 0 form expected by constrained optimizers."""

        violations = self.evaluate(indices).violations
        return [
            violations["nonempty"],
            violations["min_words"],
            violations["max_length"],
            violations["max_sentences"],
        ]


def objective_from_spec(
    sentences: Sequence[str],
    importance: Sequence[float],
    similarity_matrix: Optional[np.ndarray],
    objective_spec: Optional[dict],
    *,
    max_length: Optional[int],
    length_unit: str,
    max_sentences: Optional[int],
    min_words: int = 0,
    require_nonempty: bool = True,
    coverage_matrix: Optional[np.ndarray] = None,
) -> SelectionObjective:
    """Build the one evaluator used by dispatch, optimizers, and artifacts."""

    spec = objective_spec or {}
    weight_spec = spec.get("weights", {}) or {}
    weights = ObjectiveWeights(
        salience=float(weight_spec.get("salience", 1.0)),
        facility_coverage=float(weight_spec.get("facility_coverage", 0.8)),
        redundancy=float(weight_spec.get("redundancy", 0.7)),
    )
    return SelectionObjective(
        sentences,
        importance,
        similarity_matrix,
        importance_aggregation=str(spec.get("importance_aggregation", "sum")),
        coverage_method=str(spec.get("coverage_method", "max")),
        weights=weights,
        constraints=SelectionConstraints(
            length_unit=length_unit,
            max_length=max_length,
            min_words=int(min_words),
            max_sentences=max_sentences,
            require_nonempty=bool(require_nonempty),
        ),
        coverage_matrix=coverage_matrix,
    )
