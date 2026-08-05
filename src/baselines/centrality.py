"""TextRank / LexRank baselines for extractive summarization.

WHY THIS EXISTS
---------------
Phase 2's third and fourth baselines (docs/research/ACTION_PLAN.md, "Baseline
與 reality check"). The legacy paper's Table 6 Lead/TextRank/LexRank numbers
were adopted from a different paper's pipeline (docs/research/
CODE_AUDIT_IEEE_Access.md, around the "baseline 數字是從別的論文抄來的"
finding) and cannot be reproduced from this repo -- exactly the selective-
reporting concern reviewer R4 raised. That is the only reason this module
wraps a pinned third-party implementation (``sumy==0.12.0``, Apache-2.0)
instead of writing a new PageRank/centrality implementation: a self-written
centrality baseline invites the same "did you implement it weaker than the
literature" question this project cannot afford to reopen. It is not because
this repo lacks a PageRank implementation to reuse --
``src.features.graph.compute_textrank_scores`` already has one, and it is
unrelated to F-11 (that finding is about ``src.features.semantic``'s
degenerate row-mean centrality/novelty pair, a different module entirely) --
the reviewer-credibility argument stands on its own regardless.

MUST NOT LET sumy RE-SPLIT SENTENCES
-------------------------------------
The canonical dataset's sentence boundaries are frozen, checksum-protected
data (see F-12 and ``src.data.sentence_split``): a baseline must consume
those exact sentences, not sumy's own tokenizer's opinion of where they are.
sumy's top-level entry points (``PlaintextParser.from_string``) call their
own sentence tokenizer, which would silently run this baseline on a
different sentence set than the system pipeline and Lead/Random -- making
every comparison invalid. This module never calls that path. Instead it
constructs ``sumy.models.dom.Sentence`` objects directly from canonical
sentence strings (``Sentence.__init__`` takes a raw string and never
splits it) and wraps them in a single ``Paragraph``/``ObjectDocumentModel``.
A word-level tokenizer is still required (TF-IDF/word-overlap scoring needs
words), so the real ``sumy.nlp.tokenizers.Tokenizer("english")`` is used
rather than a hand-rolled one -- word tokenization affects TF-IDF and
LexRank's scores directly, and substituting a custom tokenizer would reopen
exactly the "weaker self-implementation" question this module exists to
avoid (see docs/research/COMPUTE_ENVIRONMENT.md for the NLTK data this
requires and why it is offline-safe once cached).

SCORING IS OVER THE FULL DOCUMENT, SELECTION IS RESTRICTED TO ELIGIBLE
SENTENCES AFTERWARD
-----------------------------------------------------------------------
TextRank/LexRank rank over the whole document's sentence graph in the
literature (removing individually-oversized sentences from the graph before
ranking would change every other sentence's centrality relative to keeping
them in, which is not what either algorithm's citable definition does).
So scoring happens over every sentence ``flatten_sentence_records`` returns,
before ``summarize_one_baseline``'s own separate eligibility filter ever
runs. The two calls to ``flatten_sentence_records`` (one here to build the
graph, one inside ``summarize_one_baseline`` for eligibility) are redundant
but harmless: the function is pure and deterministic, so both calls agree
on ``original_index``/``sentence_id`` for every sentence, which is what lets
this module's ``select_fn`` closure look a score up correctly for whichever
subset of records ``summarize_one_baseline`` decides is eligible.

MIN_WORDS IS NOT APPLIED -- SAME GENERAL PRINCIPLE AS LEAD AND RANDOM, NOT
A NEW CARVE-OUT
---------------------------------------------------------------------------
``random_baseline.py``'s module docstring states the general rule this
module also falls under: ``min_words`` is only in scope for a *searching*
selector, one that can trade one candidate sentence for another to satisfy
the floor. TextRank/LexRank rank every sentence once, then walk the ranked
list once (see ``src.baselines.contract.select_by_score``) -- exactly the
same shape as Lead's reading-order prefix and Random's shuffled walk, not a
search. An earlier draft of this module considered ``apply_min_words=True``
on the theory that a *scored* selector is different in kind from an
*unordered* one (Lead/Random): that theory does not survive contact with
what "searching" actually means here -- ranking sentences by centrality
before walking them once still walks exactly once, it does not backtrack to
trade a short sentence for a long one the way Greedy/GRASP/NSGA-II can. The
system's ``min_words`` guard exists specifically against a *searching*
``mean``-aggregation objective degenerating to a handful of high-scoring
sentences (see ``configs/phase1_mvp_multinews.yaml``'s comment on the
provisional validation band, and F-18(a)'s measured 6.05-sentence
collapse) -- TextRank/LexRank have no such search to degenerate, so that
guard does not apply to them either. Length comparability against Lead is
achieved entirely by ``max_words`` plus the skip-tolerant fill in
``select_by_score``, not by a floor. Measured on the full Multi-News
validation split (2026-08-05): TextRank's mean is 247.35 words/doc, 13.75
over Lead's 233.6 -- this is NOT a TextRank-specific effect. Every
skip-tolerant method measured (Random 246.83, the system's
greedy+length_normalized 244.0, TextRank 247.35) lands within about 3 words
of each other and close to the 250 ceiling; Lead is the outlier because it
is the only *stop*-tolerant one (see ``lead.py``'s own docstring for why
that is a deliberate, citable-definition choice, not an oversight). The gap
is a fill-rule artifact common to every skip-tolerant baseline, not a
property of any one method, and must be resolved analytically (a
length-matched bracket, the general form of the technique F-18(b) used for
greedy+length_normalized -- see ``docs/research/CODE_AUDIT_IEEE_Access.md``
for the generalized version of that finding) rather than by adding a
constraint layer to paper over it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import nltk
import numpy as np
from sumy.models.dom import ObjectDocumentModel, Paragraph, Sentence
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

from src.baselines.contract import select_by_score, summarize_one_baseline
from src.data.schemas import flatten_sentence_records

# DELIBERATE MODULE-LEVEL MUTATION OF GLOBAL nltk.data.path -- not an
# accidental import side effect. This has to happen here, at import time of
# this module specifically, rather than in a pytest conftest.py fixture:
# the offline compute cluster runs `python -m src.baselines.cli` directly
# and never goes through conftest.py at all, so a fixture-based approach
# would leave the actual production entry point unregistered while only
# tests worked. Registered at import time, not via an environment variable:
# NLTK_DATA would need to be set identically in CI, on this machine, and on
# the offline compute cluster, and is exactly the kind of per-environment
# setup step that gets forgotten on one of the three. Doing it here means
# every caller that imports this module -- pytest, `python -m
# src.baselines.cli`, a future audit script -- gets the same vendored data
# with no setup step at all. See vendor/nltk_punkt_tab/README.md for what
# this is and why it is vendored rather than downloaded (a download step in
# CI would make CI depend on NLTK's servers being reachable, which does
# nothing for the offline cluster this is actually for).
#
# .resolve() is not cosmetic: this nltk version's resource resolution
# (nltk/tokenize/punkt.py's find(), via a _assert_no_encoded_bypass-style
# check) requires an absolute path with no ".."/traversal-like components --
# confirmed necessary by a real CI failure on a clean GitHub Actions runner,
# not assumed. Inserted at the front of nltk.data.path (not appended) so
# this vendored copy always wins over any other punkt_tab a given machine
# might happen to already have cached, keeping resolution identical across
# environments rather than depending on search-path ordering. The cost of
# that front-insertion priority: a stale vendored copy left behind after an
# nltk version bump would also win silently, not raise -- see
# tests/test_nltk_punkt_tab_pin.py, which is the gate against exactly that.
_VENDORED_NLTK_DATA_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "vendor" / "nltk_punkt_tab"
)
if _VENDORED_NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _VENDORED_NLTK_DATA_DIR)

METHODS = ("textrank", "lexrank")

CENTRALITY_MIN_WORDS_NOT_APPLIED_REASON = (
    "min_words is not enforced for TextRank/LexRank; see the 'MIN_WORDS IS "
    "NOT APPLIED' section of src/baselines/centrality.py's module "
    "docstring. Same general principle as Lead and Random (see "
    "src/baselines/random_baseline.py's docstring): min_words is only in "
    "scope for a *searching* selector that can trade one candidate for "
    "another to satisfy the floor. TextRank/LexRank rank every sentence "
    "once via sumy, then walk the ranked list exactly once "
    "(select_by_score) -- the same non-searching shape as Lead's "
    "reading-order prefix and Random's shuffled walk, not a search. The "
    "system's min_words guard exists against a *searching* mean-aggregation "
    "objective degenerating to a handful of sentences (F-18(a)); "
    "TextRank/LexRank have no such search to degenerate."
)


class _LexRankWithRatings(LexRankSummarizer):
    """Exposes sumy's internal LexRank rating dict without truncating it.

    ``LexRankSummarizer.__call__`` computes a full ``{Sentence: score}``
    dict internally and then discards it after selecting the top
    ``sentences_count`` -- unlike ``TextRankSummarizer``, it has no public
    ``rate_sentences``. This subclass calls the exact same private methods
    ``__call__`` does, in the same order, with no algorithmic changes of
    any kind -- it only returns the intermediate dict instead of truncating
    it, because this module needs real-valued scores for a word-budget
    fill, not a pre-decided sentence count. This depends on
    ``LexRankSummarizer``'s private method names (``_to_words_set``,
    ``_compute_tf``, ``_compute_idf``, ``_create_matrix``, ``power_method``)
    staying stable across a sumy version bump -- the exact version pin
    (``sumy==0.12.0``) plus this module's golden-score regression test are
    the safety net for that.
    """

    def rate_sentences(self, document: ObjectDocumentModel) -> Dict[Any, float]:
        sentences_words = [self._to_words_set(s) for s in document.sentences]
        if not sentences_words:
            return {}
        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)
        matrix = self._create_matrix(
            sentences_words, self.threshold, tf_metrics, idf_metrics
        )
        scores = self.power_method(matrix, self.epsilon)
        return dict(zip(document.sentences, scores))


@lru_cache(maxsize=1)
def _get_tokenizer() -> Tokenizer:
    """Cached: constructing Tokenizer("english") loads NLTK's punkt_tab
    resource from disk (see docs/research/COMPUTE_ENVIRONMENT.md); doing
    that once per run, not once per document, matches this project's
    model-caching convention (see src/models/extractive/encoder_rank.py)."""

    return Tokenizer("english")


@lru_cache(maxsize=1)
def _get_stop_words() -> frozenset:
    return get_stop_words("english")


@lru_cache(maxsize=None)
def _get_summarizer(method: str):
    if method == "textrank":
        summarizer = TextRankSummarizer(Stemmer("english"))
    elif method == "lexrank":
        summarizer = _LexRankWithRatings(Stemmer("english"))
    else:
        raise ValueError(f"unknown centrality method {method!r}; choose one of {METHODS}")
    summarizer.stop_words = _get_stop_words()
    return summarizer


def _score_document_by_original_index(
    method: str, doc: Mapping[str, Any]
) -> Optional[Dict[int, float]]:
    """Rank every sentence in the full document, keyed by original_index.

    See this module's docstring, "SCORING IS OVER THE FULL DOCUMENT" -- this
    intentionally re-derives the full sentence list rather than reusing
    whatever subset ``summarize_one_baseline`` later decides is eligible.

    Returns ``None`` -- not an exception -- when the scorer produced a
    non-finite score for any sentence: this is a data characteristic of the
    document (see "SCORER DEGENERACY" below and
    ``docs/research/COMPUTE_ENVIRONMENT.md``'s "LexRank 相似度矩陣的已知退化"
    section), not a config/schema/programming error, so it does not belong
    in the fail-loud-and-abort category CLAUDE.md reserves for those. What
    to do about it depends on whether every eligible sentence fits under
    the active budget regardless of ranking -- information this function
    does not have (eligibility is decided later, inside
    ``summarize_one_baseline``) -- so that decision is deferred to
    ``summarize_one_centrality``'s ``select_fn``, which does have it.
    """

    records = flatten_sentence_records(doc)
    if not records:
        return {}

    tokenizer = _get_tokenizer()
    sentences = [Sentence(record["text"], tokenizer) for record in records]
    paragraph = Paragraph(sentences)
    document = ObjectDocumentModel([paragraph])

    summarizer = _get_summarizer(method)
    ratings = summarizer.rate_sentences(document)
    scores = [ratings[sentence] for sentence in document.sentences]

    # SCORER DEGENERACY. Root cause, confirmed against the real Multi-News
    # validation split (2/5,621 documents, both length-2 documents with no
    # shared vocabulary between their two sentences -- see
    # docs/research/COMPUTE_ENVIRONMENT.md): LexRank's
    # idf(term) = log(N / (1 + n_j)) is exactly 0 whenever a term's
    # document frequency n_j equals N-1. For an N=2 document this holds
    # for *every* term that appears in only one of the two sentences --
    # i.e. any pair of sentences sharing no vocabulary at all, independent
    # of whether either sentence is individually rich in content. When
    # every term in the whole document has idf=0, every pairwise
    # cosine_similarity (including each sentence's self-similarity on the
    # diagonal) evaluates to 0, the entire matrix collapses to all-zero,
    # and power_method's per-iteration renormalization
    # (`next_p /= numpy.linalg.norm(next_p)`) divides zero by zero. numpy
    # only warns ("RuntimeWarning: invalid value encountered in divide"),
    # it does not raise, so a NaN score would otherwise flow silently into
    # select_by_score's sort (where NaN comparisons are undefined, not a
    # crash -- a silent, order-dependent corruption). An earlier version of
    # this guard blamed "a sentence entirely of stopwords" and aborted the
    # whole batch on it -- that was one way to reproduce the condition
    # synthetically, not the actual mechanism found in either real
    # occurrence, and aborting reproduced F-17's original failure shape
    # (one document's data characteristic destroying an entire batch's
    # worth of already-completed work) in a code path F-17 never covered.
    if not all(np.isfinite(score) for score in scores):
        return None

    return {
        record["original_index"]: score for record, score in zip(records, scores)
    }


def summarize_one_centrality(
    doc: Mapping[str, Any], cfg: Mapping[str, Any], *, method: str
) -> Dict[str, Any]:
    """TextRank/LexRank-baseline analogue of ``select_sentences.summarize_one``.

    ``min_words`` is never enforced here -- see this module's
    ``CENTRALITY_MIN_WORDS_NOT_APPLIED_REASON`` and the "MIN_WORDS IS NOT
    APPLIED" section of the module docstring.

    ``scorer_degenerate``/``scorer_degenerate_reason`` are always present in
    the returned row (``False``/``None`` in the ordinary case), the same
    always-record-whether-it-happened convention as
    ``min_words_relaxed``/``relaxation_reason``. This is deliberately a
    separate pair of fields, not a repurposed ``infeasible_code``: a
    degenerate scorer on a document where everything still fits under
    budget is not an infeasibility (the length/sentence-count constraints
    are fully satisfied) -- it is an orthogonal fact about this method's
    behavior on this document, and folding it into ``infeasible_code``
    would break the invariant every other reader of this artifact relies on
    (``infeasible_code is None`` whenever ``feasible`` is ``True``, see
    ``src.baselines.contract.summarize_one_baseline``).
    """

    if method not in METHODS:
        raise ValueError(f"unknown centrality method {method!r}; choose one of {METHODS}")

    scores_by_original_index = _score_document_by_original_index(method, doc)
    degenerate = {"triggered": False}

    def select_fn(eligible_records, evaluator):
        if scores_by_original_index is not None:
            scores = [
                scores_by_original_index[record["original_index"]]
                for record in eligible_records
            ]
            return select_by_score(eligible_records, evaluator, scores)

        # Degenerate scorer (see _score_document_by_original_index): no
        # real ranking signal exists for this document. A uniform
        # placeholder score reduces select_by_score to "walk
        # eligible_records in original document order, skip-tolerant".
        # If that walk still selects every eligible sentence, the ranking
        # never mattered -- any scorer, however it ranked these sentences,
        # would have produced the identical output (everything) -- so it
        # is safe to resolve automatically, as long as it is visibly
        # marked (see summarize_one_centrality's docstring). If it does
        # NOT select everything, which sentences get dropped genuinely
        # depends on a ranking this method cannot produce here; do not
        # guess -- raise, and treat it as an open policy question (see
        # docs/research/COMPUTE_ENVIRONMENT.md's "LexRank 相似度矩陣的
        # 已知退化" section; not yet observed on Multi-News validation).
        placeholder_scores = [0.0] * len(eligible_records)
        selected = select_by_score(eligible_records, evaluator, placeholder_scores)
        if len(selected) < len(eligible_records):
            raise ValueError(
                f"{method} could not rank sentences for document "
                f"{doc.get('id')!r} (similarity matrix collapsed to "
                "all-zero) and not every eligible sentence fits under the "
                "active budget, so which sentences to drop genuinely "
                "depends on a ranking this method cannot produce here; "
                "this needs a policy decision, not a guess -- see "
                "docs/research/COMPUTE_ENVIRONMENT.md"
            )
        degenerate["triggered"] = True
        return selected

    result = summarize_one_baseline(
        doc,
        cfg,
        method=method,
        select_fn=select_fn,
        length_gate=True,
        apply_min_words=False,
        min_words_not_applied_reason=CENTRALITY_MIN_WORDS_NOT_APPLIED_REASON,
    )
    result["scorer_degenerate"] = degenerate["triggered"]
    result["scorer_degenerate_reason"] = (
        (
            f"{method}'s similarity matrix collapsed to all-zero for this "
            "document (see docs/research/COMPUTE_ENVIRONMENT.md, 'LexRank "
            "相似度矩陣的已知退化'); every eligible sentence fit under the "
            "active budget regardless of ranking, so selection is "
            "well-defined (all eligible sentences, original document "
            "order) despite the undefined score."
        )
        if degenerate["triggered"]
        else None
    )
    return result


def summarize_one_textrank(doc: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return summarize_one_centrality(doc, cfg, method="textrank")


def summarize_one_lexrank(doc: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return summarize_one_centrality(doc, cfg, method="lexrank")
