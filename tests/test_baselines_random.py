"""Tests for the Random baseline.

Covers what the Lead golden tests could not: determinism and reproducibility
properties that only matter once a baseline's ``select_fn`` actually uses
randomness. Also includes a real-data sample (not just synthetic toy docs)
-- this project has hit "tests cover the API but not the real data
distribution" enough times that a real-distribution case is required here
by design, not left to a follow-up.

The real-data sample is a checked-in fixture, not a direct read of
``data/processed/`` (which is ``.gitignore``d and absent in CI): see
``tests/fixtures/README.md`` for where it came from, why it is a curated
16-row sample rather than a blind "first N rows" slice (the first 20 rows
turned out to all be the easy case), why its sentence/reference text is
placeholder text rather than real Multi-News content (de-lexicalized for
redistribution-risk reasons -- this fixture must not be used by any test
that depends on actual word content), and the recorded SHA-256 this module
checks on every load.
"""

import hashlib
import json
import os

import pytest

from src.baselines.contract import derive_row_seed, summarize_one_baseline
from src.baselines.random_baseline import _select_random, summarize_one_random
from src.data.schemas import build_document_example

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "multi_news_validation_diagnostic_sample.jsonl"
)
# Recorded in tests/fixtures/README.md. If this fixture is intentionally
# changed, update both the file and this constant together -- that is the
# whole point of the check below.
FIXTURE_SHA256 = "dca3bae4f93a1bb51fe0b66ed5c23367988fcc51a0a425a397a87d24c38bd848"


def _assert_fixture_integrity() -> None:
    with open(FIXTURE_PATH, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    assert actual == FIXTURE_SHA256, (
        f"tests/fixtures/multi_news_validation_diagnostic_sample.jsonl has changed "
        f"(sha256 {actual}) but tests/test_baselines_random.py's FIXTURE_SHA256 and "
        f"tests/fixtures/README.md were not updated to match -- see that README "
        f"before editing this fixture"
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


def _load_fixture_rows():
    _assert_fixture_integrity()
    rows = []
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        for line in f:
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
    reversed, and alone, on real data (not a synthetic single-row case).

    The target is validation_538, one of the fixture's min_words_relaxed
    rows (source_capacity_words=51): its selection sits right at the edge
    of what is reachable at all, which is exactly where an order-dependent
    bug would be most likely to show up as a different result rather than
    just a different-looking-but-still-feasible one.
    """

    rows = _load_fixture_rows()
    cfg = _gate2_cfg()
    target = next(doc for doc in rows if doc["id"] == "validation_538")

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


def test_real_data_sample_is_feasible_with_min_words_not_applied():
    """The 16-row curated real-data fixture (not 5,621, not synthetic, and
    not just the easy first-20-rows case -- see tests/fixtures/README.md)
    under the project's Gate 2 protocol (max_words=250, requested
    min_words=200). Random uses apply_min_words=False, same as Lead -- see
    random_baseline.py's docstring, "MIN_WORDS DOES NOT APPLY HERE EITHER",
    for why: min_words is out of scope for any baseline that does not
    search over subsets to satisfy it, this module's skip-tolerant selector
    included. Every row must still come out feasible under the upper bound
    alone (including the min_words_relaxed rows, the single-source-document
    rows, and validation_4576 -- see the dedicated regression test below for
    that last one), and requested_min_words must stay recorded even though
    it is not enforced."""

    rows = _load_fixture_rows()
    assert len(rows) == 16
    for doc in rows:
        result = summarize_one_random(doc, cfg=_gate2_cfg(), seed=42)
        assert result["selection_evaluation"]["feasible"] is True, doc["id"]
        assert result["output_budget"]["min_words_applied"] is False
        assert result["output_budget"]["requested_min_words"] == 200
        assert result["output_budget"]["effective_min_words"] == 0


def test_validation_4576_succeeds_with_apply_min_words_false():
    """Pins the concrete row that motivated flipping Random's
    apply_min_words to False (PR #11 review, "Blocking 1"). validation_4576
    has source_capacity_words=244 and min_words_relaxed=False -- 200 words
    is genuinely reachable by *some* subset of its 7 eligible sentences --
    but a full-split rerun (scripts/audit/random_baseline_min_words.py)
    found the skip-tolerant selector this module implements fails to reach
    it under every one of seeds 0, 1, 42, and 9999: the 244-word optimum
    requires keeping both of the two long sentences (72 and 83 words) while
    dropping the two shortest, a combination one random walk rarely lands
    on. With apply_min_words=False this row must succeed regardless, and
    requested_min_words must still be recorded as 200 -- the request is
    never dropped, only not enforced."""

    rows = _load_fixture_rows()
    target = next(doc for doc in rows if doc["id"] == "validation_4576")
    for seed in (0, 42):
        result = summarize_one_random(target, cfg=_gate2_cfg(), seed=seed)
        assert result["selection_evaluation"]["feasible"] is True, seed
        assert result["output_budget"]["requested_min_words"] == 200
        assert result["output_budget"]["min_words_applied"] is False


def test_fixture_sha256_matches_recorded_value():
    """Dedicated, explicitly-named check so a silent fixture edit shows up
    as its own failing test, not just as a side effect inside another
    test's setup."""

    _assert_fixture_integrity()
