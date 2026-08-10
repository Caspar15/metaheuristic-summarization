"""Metric-specific greedy extractive references.

The ICT Express draft called the mean of SciTLDR's per-source-sentence
``rouge_scores`` field an oracle. That statistic is not an oracle and can be
lower than an ordinary system. This module instead performs greedy
reference-aware sentence selection.

Naming is deliberately strict: greedy search is neither exhaustive nor a
mathematical upper bound. It is an approximate, reference-aware diagnostic
and must be reported as a *greedy reference*. ROUGE-1-, ROUGE-2-, and
ROUGE-Lsum-optimized references are different searches and are therefore run
and reported separately.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Sequence, Union

from src.data.schemas import (
    SchemaValidationError,
    extract_references,
    flatten_sentence_texts,
)
from src.eval.rouge import _as_lsum, _new_scorer, DEFAULT_METRICS, rouge_scores


def _validate_metrics(metrics: Sequence[str], *, label: str) -> tuple[str, ...]:
    resolved = tuple(metrics)
    if not resolved:
        raise ValueError(f"{label} must contain at least one metric")
    unknown = set(resolved) - set(DEFAULT_METRICS)
    if unknown:
        raise ValueError(
            f"{label} contains unsupported metrics {sorted(unknown)}; "
            f"choose from {list(DEFAULT_METRICS)}"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} contains duplicate metrics")
    return resolved


def _fmeasure(scorer, pred: str, refs: Sequence[str], metric: str) -> float:
    prediction = _as_lsum(pred)
    return max(
        scorer.score(_as_lsum(reference), prediction)[metric].fmeasure
        for reference in refs
    )


def _validate_inputs(
    sentences: Sequence[str], refs: Union[str, Sequence[str]]
) -> tuple[list[str], list[str]]:
    sentence_list = list(sentences)
    if not sentence_list:
        raise ValueError("greedy reference requires at least one source sentence")
    if not all(isinstance(sentence, str) and sentence.strip() for sentence in sentence_list):
        raise ValueError("source sentences must be non-empty strings")
    reference_list = [refs] if isinstance(refs, str) else list(refs)
    if not reference_list or not all(
        isinstance(reference, str) and reference.strip()
        for reference in reference_list
    ):
        raise ValueError("greedy reference requires at least one non-empty reference")
    return sentence_list, reference_list


def greedy_reference_summary(
    sentences: Sequence[str],
    refs: Union[str, Sequence[str]],
    *,
    max_words: int | None = None,
    max_sentences: int | None = None,
    metric: str = "rouge1",
    scorer=None,
) -> List[int]:
    """Return indices selected by metric-specific greedy reference search."""

    _validate_metrics((metric,), label="metric")
    sentence_list, reference_list = _validate_inputs(sentences, refs)
    if max_words is not None and max_words < 1:
        raise ValueError("max_words must be positive when provided")
    if max_sentences is not None and max_sentences < 1:
        raise ValueError("max_sentences must be positive when provided")
    score_fn = scorer or _new_scorer((metric,), True)

    selected: List[int] = []
    selected_set: set[int] = set()
    current_summary = ""
    selected_words = 0
    current_score = 0.0
    while True:
        if max_sentences is not None and len(selected) >= max_sentences:
            break
        best_index: int | None = None
        best_gain = 1e-9
        for index, sentence in enumerate(sentence_list):
            if index in selected_set:
                continue
            word_count = len(sentence.split())
            if max_words is not None and selected_words + word_count > max_words:
                continue
            # The delivered extract is ordered by source position. Search
            # must score that same object; scoring greedy insertion order and
            # sorting only at return time changes ROUGE-2/Lsum semantics.
            candidate_indices = sorted([*selected, index])
            candidate_summary = " ".join(
                sentence_list[candidate_index]
                for candidate_index in candidate_indices
            )
            candidate_score = _fmeasure(
                score_fn, candidate_summary, reference_list, metric
            )
            gain = candidate_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        selected_set.add(best_index)
        current_summary = " ".join(
            sentence_list[index] for index in sorted(selected)
        )
        selected_words += len(sentence_list[best_index].split())
        current_score = _fmeasure(
            score_fn, current_summary, reference_list, metric
        )
    return sorted(selected)


def _document_inputs(
    document: Mapping[str, Any], row_number: int
) -> tuple[list[str], list[str]]:
    """Extract canonical or explicit legacy inputs, rejecting schema drift."""

    if not isinstance(document, Mapping):
        raise ValueError(f"row {row_number} must be a JSON object")
    try:
        sentences = flatten_sentence_texts(document)
        references = extract_references(document)
    except (SchemaValidationError, TypeError, ValueError) as error:
        raise ValueError(
            f"row {row_number} ({document.get('id')!r}) violates the "
            f"source/reference schema: {error}"
        ) from error
    if not sentences:
        schema = "canonical" if "documents" in document else "legacy"
        raise ValueError(
            f"row {row_number} ({document.get('id')!r}) has no source sentences "
            f"under the {schema} schema"
        )
    if not references:
        raise ValueError(
            f"row {row_number} ({document.get('id')!r}) has no non-empty references"
        )
    return sentences, references


def greedy_reference_run(
    docs: Sequence[Mapping[str, Any]],
    *,
    max_words: int | None = None,
    max_sentences: int | None = None,
    target_metric: str = "rouge1",
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Dict[str, Any]:
    """Run one optimization target and return scores plus auditable selections."""

    _validate_metrics((target_metric,), label="target_metric")
    report_metrics = _validate_metrics(metrics, label="metrics")
    if not docs:
        raise ValueError("greedy reference cannot evaluate an empty corpus")
    selection_scorer = _new_scorer((target_metric,), True)
    predictions: list[str] = []
    references: list[list[str]] = []
    selections: list[dict[str, Any]] = []
    for row_number, document in enumerate(docs, start=1):
        sentences, row_references = _document_inputs(document, row_number)
        selected_indices = greedy_reference_summary(
            sentences,
            row_references,
            max_words=max_words,
            max_sentences=max_sentences,
            metric=target_metric,
            scorer=selection_scorer,
        )
        summary = " ".join(sentences[index] for index in selected_indices)
        predictions.append(summary)
        references.append(row_references)
        selections.append(
            {
                "id": document.get("id"),
                "selected_indices": selected_indices,
                "selected_words": len(summary.split()),
                "selected_sentences": len(selected_indices),
            }
        )

    scores = rouge_scores(
        predictions,
        references,
        metrics=report_metrics,
        reference_metric=target_metric,
    )
    return {
        "optimization_target": target_metric,
        "scores": scores,
        "mean_selected_words": sum(row["selected_words"] for row in selections)
        / len(selections),
        "mean_selected_sentences": sum(
            row["selected_sentences"] for row in selections
        )
        / len(selections),
        "selections": selections,
    }


def greedy_reference_report(
    docs: Sequence[Mapping[str, Any]],
    *,
    max_words: int | None = None,
    max_sentences: int | None = None,
    target_metrics: Sequence[str] = DEFAULT_METRICS,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Dict[str, Any]:
    """Run independent greedy searches for every requested ROUGE target."""

    targets = _validate_metrics(target_metrics, label="target_metrics")
    report_metrics = _validate_metrics(metrics, label="metrics")
    runs = {
        target: greedy_reference_run(
            docs,
            max_words=max_words,
            max_sentences=max_sentences,
            target_metric=target,
            metrics=report_metrics,
        )
        for target in targets
    }
    return {
        "method": "metric_specific_greedy_reference",
        "exact_upper_bound": False,
        "interpretation": (
            "Reference-aware greedy diagnostic; a lower bound on exhaustive "
            "search quality, not a mathematical oracle ceiling."
        ),
        "n_docs": len(docs),
        "max_words": max_words,
        "max_sentences": max_sentences,
        "optimization_targets": runs,
    }


# Backward-compatible APIs for historical scripts. New code and prose must
# use the greedy_reference_* names above.
def greedy_oracle_summary(
    sentences: List[str],
    refs: Union[str, Sequence[str]],
    max_tokens: int | None = None,
    max_sentences: int | None = None,
    metric: str = "rouge1",
    scorer=None,
) -> List[int]:
    return greedy_reference_summary(
        sentences,
        refs,
        max_words=max_tokens,
        max_sentences=max_sentences,
        metric=metric,
        scorer=scorer,
    )


def oracle_scores(
    docs: List[Dict],
    max_tokens: int | None = None,
    max_sentences: int | None = None,
    target_metric: str = "rouge1",
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Dict[str, float]:
    return greedy_reference_run(
        docs,
        max_words=max_tokens,
        max_sentences=max_sentences,
        target_metric=target_metric,
        metrics=metrics,
    )["scores"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute metric-specific greedy extractive references"
    )
    parser.add_argument("--input", required=True, help="canonical or explicit legacy JSONL")
    parser.add_argument(
        "--max_words",
        "--max_tokens",
        dest="max_words",
        type=int,
        default=None,
        help="whitespace-delimited word budget; --max_tokens is a legacy alias",
    )
    parser.add_argument("--max_sentences", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--target_metric",
        action="append",
        choices=DEFAULT_METRICS,
        help=(
            "optimization target; repeat for multiple targets. If omitted, "
            "rouge1, rouge2, and rougeLsum are each run independently"
        ),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    docs: list[dict[str, Any]] = []
    with open(args.input, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at input line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"input line {line_number} must be a JSON object")
            docs.append(row)
            if args.limit is not None and len(docs) >= args.limit:
                break

    report = greedy_reference_report(
        docs,
        max_words=args.max_words,
        max_sentences=args.max_sentences,
        target_metrics=tuple(args.target_metric or DEFAULT_METRICS),
    )
    print(
        f"Metric-specific greedy reference over {len(docs)} docs "
        f"(max_words={args.max_words}, max_sentences={args.max_sentences}; "
        "not an exact upper bound):"
    )
    for target, run in report["optimization_targets"].items():
        values = ", ".join(
            f"{metric}={score:.4f}" for metric, score in run["scores"].items()
        )
        print(f"  optimize {target}: {values}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
