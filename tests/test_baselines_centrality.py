"""Golden and correctness tests for the TextRank/LexRank baselines.

Requires the classic-NLTK-avoiding shared tokenizer's opposite number: sumy's
own Tokenizer("english") needs the NLTK punkt_tab/english resource cached
somewhere on nltk.data.path (see docs/research/COMPUTE_ENVIRONMENT.md for
exactly what and why). On a machine that has already run
`python -c "import nltk; nltk.download('punkt_tab')"` once, no extra
environment setup is needed here -- nltk's default search path picks up
~/nltk_data automatically.
"""

import json
import sys

import pytest

from src.baselines import cli as baseline_cli
from src.baselines.centrality import (
    _score_document_by_original_index,
    summarize_one_lexrank,
    summarize_one_textrank,
)
from src.data.schemas import build_document_example
from src.utils.io import write_jsonl_atomic


def _finance_and_bakery_doc():
    """4 realistic sentences, 3 topically related + 1 unrelated -- avoids
    the degenerate all-stopword-content toy sentences used elsewhere in
    this test suite (e.g. test_baselines_lead.py's `_words("a", 10)`
    pattern), which collapse LexRank's similarity graph to all-zero (every
    token is an English stopword) and would make this a test of the
    non-finite-score guard rather than of real scoring behaviour."""

    return build_document_example(
        example_id="golden1",
        split="validation",
        documents=[[
            "The stock market rallied sharply after the central bank announcement.",
            "Investors welcomed the interest rate decision from the central bank.",
            "A local bakery introduced a new sourdough bread recipe this week.",
            "The central bank also signaled further rate cuts are likely next year.",
        ]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def test_golden_scores_textrank_and_lexrank_are_deterministic():
    """Pinned regression values for sumy==0.12.0. This doubles as the
    regression gate for src.baselines.centrality._LexRankWithRatings,
    which depends on LexRankSummarizer's private method names staying
    stable across a sumy version bump -- if this test starts failing after
    a dependency update, check that first, not this module's own logic."""

    doc = _finance_and_bakery_doc()

    textrank_scores = _score_document_by_original_index("textrank", doc)
    assert textrank_scores[0] == pytest.approx(0.2409818920903655, rel=1e-6)
    assert textrank_scores[1] == pytest.approx(0.25790366344735627, rel=1e-6)
    assert textrank_scores[2] == pytest.approx(0.24999993563417033, rel=1e-6)
    assert textrank_scores[3] == pytest.approx(0.2511143100062544, rel=1e-6)

    lexrank_scores = _score_document_by_original_index("lexrank", doc)
    # Real, reproducible sumy 0.12.0 behaviour for this small a document
    # under LexRank's default threshold=0.1 -- not a placeholder or a sign
    # something is broken; a 4-sentence graph at this threshold commonly
    # converges to a uniform stationary distribution.
    for index in range(4):
        assert lexrank_scores[index] == pytest.approx(0.5, rel=1e-9)


def test_unknown_method_fails_loud():
    with pytest.raises(ValueError, match="unknown centrality method"):
        _score_document_by_original_index("mmr", _finance_and_bakery_doc())


def test_all_stopword_document_resolves_by_taking_everything_when_it_all_fits():
    """This is one way to reproduce the underlying scorer degeneracy
    synthetically, not the actual mechanism: scanning the real Multi-News
    validation split found the true trigger is a length-2 document whose
    two sentences share no vocabulary at all, which makes LexRank's
    idf(term) = log(2/(1+n_j)) exactly 0 for every term (n_j=1 for anything
    appearing in only one of the two sentences) -- collapsing the whole
    similarity matrix to all-zero regardless of whether either sentence is
    individually rich in content. An all-stopword document happens to also
    be a length-2, zero-shared-vocabulary case, which is why it triggers
    the same degeneracy, but "entirely stopwords" is not the general
    condition (see src.baselines.centrality's inline comment for the
    corrected mechanism and docs/research/COMPUTE_ENVIRONMENT.md for the
    two real occurrences found in Multi-News validation, both resolved the
    same way this test expects: budget large enough that ranking never
    mattered)."""

    degenerate = build_document_example(
        example_id="degenerate1",
        split="validation",
        documents=[["the the the the", "a a a a"]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 20,
            "min_words": 0,
            "require_nonempty": True,
        }
    }
    # The RuntimeWarning is the expected, load-bearing signature of the
    # idf-collapse this test deliberately triggers -- numpy warns rather
    # than raises when power_method divides a zero vector by its own zero
    # L2 norm (see centrality.py's "SCORER DEGENERACY" comment). Asserting
    # on it, not just letting it print, is the point: scorer_degenerate's
    # guard exists specifically to catch this NaN *after* numpy has already
    # produced it, so this warning is the intermediate state the guard is
    # built around -- a fourth, unexpected NaN must not be able to hide
    # among three already-known occurrences of this exact message.
    with pytest.warns(RuntimeWarning, match="invalid value"):
        result = summarize_one_lexrank(degenerate, cfg)
    assert result["feasible"] is True
    assert sorted(result["selected_indices"]) == [0, 1]
    assert result["scorer_degenerate"] is True
    assert result["scorer_degenerate_reason"]


def test_length_two_disjoint_vocabulary_document_resolves_the_same_way_with_rich_content():
    """The verified mechanism, not the stopword coincidence above: two
    sentences that are each individually rich in real, non-stopword
    content, sharing zero vocabulary between them, still collapse idf to 0
    for every term (n_j=1 for every term in a length-2 document with no
    overlap). This is the actual shape of both real occurrences found in
    Multi-News validation (validation_1082, validation_2303 -- see
    docs/research/COMPUTE_ENVIRONMENT.md): neither was an all-stopword
    sentence, both were ordinary (if boilerplate) prose that simply shared
    no words with its sibling sentence, and both fit entirely under the
    250-word budget (59 and 77 words respectively) -- so the scorer
    degeneracy never had a chance to change the output."""

    doc = build_document_example(
        example_id="disjoint_vocab1",
        split="validation",
        documents=[[
            "The seed for this crawl was a list of every host previously "
            "identified as part of a large scale web archiving project.",
            "Please subscribe to our newsletter for weekly updates about "
            "upcoming concerts in your area.",
        ]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 40,
            "min_words": 0,
            "require_nonempty": True,
        }
    }
    with pytest.warns(RuntimeWarning, match="invalid value"):
        result = summarize_one_lexrank(doc, cfg)
    assert result["feasible"] is True
    assert sorted(result["selected_indices"]) == [0, 1]
    assert result["scorer_degenerate"] is True


def test_length_two_disjoint_vocabulary_document_fails_loud_when_it_does_not_all_fit():
    """Not yet observed on Multi-News validation (both real occurrences fit
    entirely under budget), but the code path must still refuse to guess:
    if the budget is tight enough that not every eligible sentence fits,
    which one(s) to drop genuinely depends on a ranking LexRank cannot
    produce for this document. This is an open policy question (see
    docs/research/COMPUTE_ENVIRONMENT.md's "LexRank 相似度矩陣的已知退化"),
    not something to resolve by picking an arbitrary sentence silently.

    max_words=25 is chosen deliberately: both sentences (22 and 14 words)
    are individually under 25, so both are ELIGIBLE (neither is excluded
    as individually oversized) -- but their sum, 36, is not, so the two of
    them together do not both fit. A tighter budget that excluded the
    22-word sentence from eligibility entirely would trivially "fit"
    whatever's left, which is exactly the resolved case the previous test
    already covers, not this one.
    """

    doc = build_document_example(
        example_id="disjoint_vocab_tight1",
        split="validation",
        documents=[[
            "The seed for this crawl was a list of every host previously "
            "identified as part of a large scale web archiving project.",
            "Please subscribe to our newsletter for weekly updates about "
            "upcoming concerts in your area.",
        ]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 25,
            "min_words": 0,
            "require_nonempty": True,
        }
    }
    with pytest.warns(RuntimeWarning, match="invalid value"):
        with pytest.raises(ValueError, match="needs a policy decision"):
            summarize_one_lexrank(doc, cfg)


def test_scorer_degenerate_is_false_for_ordinary_documents():
    result = summarize_one_lexrank(
        _finance_and_bakery_doc(),
        {
            "length_control": {
                "unit": "words",
                "max_words": 40,
                "min_words": 0,
                "require_nonempty": True,
            }
        },
    )
    assert result["scorer_degenerate"] is False
    assert result["scorer_degenerate_reason"] is None


def test_single_sentence_document_is_not_a_collapse_and_scores_fine():
    """Algebraic edge case, not the same pathology: for N=1, every term's
    idf = log(1/(1+1)) = log(0.5), NEGATIVE rather than 0. Since only the
    square of idf ever appears in LexRank's cosine_similarity, the sign
    does not matter -- self-similarity is exactly 1.0 regardless, so a
    single-sentence document scores fine and never triggers scorer
    degeneracy. Multi-News validation canonical has zero such documents
    (minimum sentence count is 2, confirmed by a full-split scan), so this
    is a documented edge case for future datasets (e.g. GovReport), not one
    Multi-News exercises."""

    doc = build_document_example(
        example_id="single1",
        split="validation",
        documents=[[
            "The central bank raised interest rates on Tuesday amid rising "
            "inflation concerns."
        ]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 40,
            "min_words": 0,
            "require_nonempty": True,
        }
    }
    result = summarize_one_lexrank(doc, cfg)
    assert result["feasible"] is True
    assert result["scorer_degenerate"] is False
    assert result["selected_indices"] == [0]


def _doc_with_an_abbreviation_sentence():
    """One canonical sentence contains an internal period after an
    abbreviation ("Dr."). If sumy were allowed to run its own sentence
    tokenizer over this text (instead of consuming our pre-split canonical
    sentences directly, per centrality.py's module docstring), a naive
    splitter could cut this into two sentences at that period -- silently
    changing the sentence set this baseline operates on relative to Lead/
    Random/the system pipeline, which all consume the same canonical
    sentences unchanged."""

    return build_document_example(
        example_id="no_resplit1",
        split="validation",
        documents=[[
            "Dr. Smith announced the merger on Tuesday afternoon.",
            "Markets reacted calmly to the unexpected announcement.",
            "The merger is expected to close by the end of the year.",
        ]],
        references=["a reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


@pytest.mark.parametrize("summarize_fn", [summarize_one_textrank, summarize_one_lexrank])
def test_selected_text_equals_canonical_input_verbatim_no_resplit(summarize_fn):
    doc = _doc_with_an_abbreviation_sentence()
    canonical_sentences = [
        sentence["text"]
        for document in doc["documents"]
        for section in document["sections"]
        for sentence in section["sentences"]
    ]
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 100,
            "min_words": 0,
            "require_nonempty": True,
        }
    }

    result = summarize_fn(doc, cfg)

    # Every selected sentence's text must be byte-identical to one of the
    # three canonical input sentences -- not a substring, not a merge of
    # two, not a further split of one (which is what a period-triggered
    # re-split of "Dr. Smith..." would produce).
    for sentence_text in result["summary_sentences"]:
        assert sentence_text in canonical_sentences
    # The abbreviation sentence, if selected at all, appears exactly once
    # and exactly as written -- not split into "Dr." and "Smith announced...".
    assert result["summary_sentences"].count(
        "Dr. Smith announced the merger on Tuesday afternoon."
    ) <= 1
    assert not any(s == "Dr." for s in result["summary_sentences"])


@pytest.mark.parametrize("baseline", ["textrank", "lexrank"])
def test_cli_provenance_has_no_seed_or_ordering_keys(tmp_path, monkeypatch, baseline):
    doc = _finance_and_bakery_doc()
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [doc])

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 40\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", baseline,
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "run1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    baseline_cli.main()

    provenance = json.loads((run_dir / "run1" / "baseline_run.json").read_text(encoding="utf-8"))
    assert provenance["baseline"] == baseline
    assert "seed" not in provenance
    assert "ordering" not in provenance
    assert "first_k" not in provenance

    feasibility = json.loads(
        (run_dir / "run1" / "feasibility_report.json").read_text(encoding="utf-8")
    )
    assert feasibility["total_count"] == 1
    assert feasibility["feasible_count"] == 1


@pytest.mark.parametrize("baseline", ["textrank", "lexrank"])
def test_cli_rejects_ordering_and_seed_for_centrality_baselines(tmp_path, monkeypatch, baseline):
    doc = _finance_and_bakery_doc()
    input_path = tmp_path / "toy.jsonl"
    write_jsonl_atomic(str(input_path), [doc])
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "length_control:\n"
        "  unit: words\n"
        "  max_words: 40\n"
        "  min_words: 0\n"
        "  require_nonempty: true\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs"

    argv = [
        "baselines-cli",
        "--baseline", baseline,
        "--config", str(config_path),
        "--split", "validation",
        "--input", str(input_path),
        "--run_dir", str(run_dir),
        "--stamp", "rejected",
        "--seed", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="seed"):
        baseline_cli.main()
    assert not (run_dir / "rejected").exists()


def test_min_words_is_not_applied_and_reason_is_recorded():
    doc = _finance_and_bakery_doc()
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": 15,
            "min_words": 12,
            "require_nonempty": True,
        }
    }
    result = summarize_one_textrank(doc, cfg)
    budget = result["output_budget"]
    assert budget["min_words_applied"] is False
    assert budget["requested_min_words"] == 12
    assert budget["effective_min_words"] == 0
    assert budget["min_words_not_applied_reason"]
    assert "TextRank/LexRank" in budget["min_words_not_applied_reason"] or "searching" in budget["min_words_not_applied_reason"]
