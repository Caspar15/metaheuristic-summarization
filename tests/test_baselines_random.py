"""Tests for the Random baseline.

Covers what the Lead golden tests could not: determinism and reproducibility
properties that only matter once a baseline's ``select_fn`` actually uses
randomness. Also includes a real-data sample (not just synthetic toy docs)
-- this project has hit "tests cover the API but not the real data
distribution" enough times that a real-distribution case is required here
by design, not left to a follow-up.
"""

import json
import os

import pytest

from src.baselines.contract import derive_row_seed, summarize_one_baseline
from src.baselines.random_baseline import _select_random, summarize_one_random
from src.data.schemas import build_document_example

REAL_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "multi_news_validation_canonical.jsonl",
)


def _words(tag: str, count: int) -> str:
    return " ".join([tag] * count)


def _toy_doc_ten_sentences():
    return build_document_example(
        example_id="random_toy1",
        split="validation",
        documents=[[_words(chr(97 + i), 15) for i in range(10)]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def _toy_cfg(max_words=60, min_words=0):
    return {
        "length_control": {
            "unit": "words",
            "max_words": max_words,
            "min_words": min_words,
            "require_nonempty": True,
        }
    }


def _load_real_rows(n):
    rows = []
    with open(REAL_DATA_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def _gate2_cfg():
    return {
        "length_control": {
            "unit": "words",
            "max_words": 250,
            "min_words": 200,
            "require_nonempty": True,
        }
    }


def test_same_seed_reproduces_same_indices():
    doc = _toy_doc_ten_sentences()
    cfg = _toy_cfg()
    r1 = summarize_one_random(doc, cfg, seed=123)
    r2 = summarize_one_random(doc, cfg, seed=123)
    assert r1["selected_indices"] == r2["selected_indices"]
    assert r1["row_seed"] == r2["row_seed"]


def test_different_seed_gives_different_indices():
    doc = _toy_doc_ten_sentences()
    cfg = _toy_cfg()
    indices_by_seed = {
        seed: summarize_one_random(doc, cfg, seed=seed)["selected_indices"]
        for seed in (1, 2, 10, 20, 100, 200)
    }
    # Every pair of distinct seeds tried here happens to produce a distinct
    # selection (confirmed empirically); assert pairwise inequality directly
    # rather than just "not all identical" so a future regression that makes
    # only *some* pairs collide would still be caught.
    seeds = list(indices_by_seed)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            assert indices_by_seed[seeds[i]] != indices_by_seed[seeds[j]], (
                f"seed {seeds[i]} and {seeds[j]} produced identical indices"
            )


def test_seed_and_row_seed_are_recorded_in_the_artifact():
    doc = _toy_doc_ten_sentences()
    cfg = _toy_cfg()
    result = summarize_one_random(doc, cfg, seed=999)
    assert result["seed"] == 999
    assert result["row_seed"] == derive_row_seed(999, doc["id"])
    assert isinstance(result["row_seed"], int)


def test_requires_seed_fails_loud_without_a_seed():
    doc = _toy_doc_ten_sentences()
    cfg = _toy_cfg()
    with pytest.raises(ValueError, match="seed"):
        summarize_one_baseline(
            doc,
            cfg,
            method="random",
            select_fn=_select_random,
            length_gate=True,
            apply_min_words=True,
            requires_seed=True,
            seed=None,
        )


def test_per_row_independence_single_row_vs_full_file():
    """Running just row N in isolation must match running the whole file --
    a property a global random.seed() call cannot provide, since its state
    is cumulative across every prior row processed. Checked forward,
    reversed, and alone, on real data (not a synthetic single-row case)."""

    rows = _load_real_rows(5)
    cfg = _gate2_cfg()
    target = rows[2]

    alone = summarize_one_random(target, cfg, seed=7)["selected_indices"]

    forward = None
    for doc in rows:
        result = summarize_one_random(doc, cfg, seed=7)
        if doc["id"] == target["id"]:
            forward = result["selected_indices"]

    reverse = None
    for doc in reversed(rows):
        result = summarize_one_random(doc, cfg, seed=7)
        if doc["id"] == target["id"]:
            reverse = result["selected_indices"]

    assert alone == forward == reverse


def test_output_stays_in_original_document_order_despite_random_draw_order():
    """random_baseline.py's docstring claims the shuffled *draw* order never
    leaks into the output because summarize_one_baseline sorts the final
    selected indices back into original position -- pin that down directly,
    since it is exactly the property the docstring relies on instead of
    reimplementing an ordering step in this module."""

    doc = _toy_doc_ten_sentences()
    cfg = _toy_cfg()
    result = summarize_one_random(doc, cfg, seed=55)
    assert result["selected_indices"] == sorted(result["selected_indices"])


def test_real_data_sample_is_feasible_with_min_words_applied():
    """20 real Multi-News validation rows (not 5,621, not synthetic) under
    the project's Gate 2 protocol (max_words=250, requested min_words=200).
    apply_min_words=True here relies on the skip-tolerant sampling in
    _select_random (see random_baseline.py's docstring for the measured
    justification); every row must come out feasible."""

    rows = _load_real_rows(20)
    cfg = _gate2_cfg()
    for doc in rows:
        result = summarize_one_random(doc, cfg, seed=42)
        assert result["selection_evaluation"]["feasible"] is True, doc["id"]
        assert result["output_budget"]["min_words_applied"] is True
        assert result["output_budget"]["selected_words"] >= (
            result["output_budget"]["effective_min_words"]
        )
