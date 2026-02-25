from typing import List, Optional, Set
from collections import Counter
import math
import re

# ---- built-in English stop-words (no NLTK dependency) ----
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "d", "did", "didn", "do", "does", "doesn", "doing", "don",
    "down", "during", "each", "few", "for", "from", "further", "get",
    "got", "had", "hadn", "has", "hasn", "have", "haven", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "isn", "it", "its", "itself", "just", "ll",
    "m", "ma", "may", "me", "might", "more", "most", "must", "mustn", "my",
    "myself", "need", "no", "nor", "not", "now", "o", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "re", "s", "said", "same", "shan", "she", "should",
    "shouldn", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "ve",
    "very", "was", "wasn", "we", "were", "weren", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "won",
    "would", "wouldn", "y", "you", "your", "yours", "yourself",
    "yourselves", "also", "like", "new", "one", "two",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _tokenize(
    text: str,
    use_stopwords: bool = False,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """Lower-case, strip punctuation, optionally filter stop-words."""
    text = _PUNCT_RE.sub(" ", text.lower())
    tokens = text.split()
    if use_stopwords:
        stops = stopwords if stopwords is not None else ENGLISH_STOPWORDS
        tokens = [t for t in tokens if t not in stops and len(t) > 1]
    return tokens


# ---------- original API (kept for backward compat) ----------

def sentence_tf_isf_scores(sentences: List[str]) -> List[float]:
    """Compute TF-ISF per sentence (sum over tokens):
    ISF(t) = log( N / (1 + n_t) ), where n_t is #sentences containing token t.
    Tokenization: simple whitespace lowercased split.
    """
    toks_per_sent = [s.lower().split() for s in sentences]
    N = max(1, len(toks_per_sent))
    # sentence frequency per token
    sf: Counter = Counter()
    for toks in toks_per_sent:
        for t in set(toks):
            sf[t] += 1
    scores: List[float] = []
    for toks in toks_per_sent:
        tf = Counter(toks)
        val = 0.0
        for t, c in tf.items():
            isf = math.log(N / (1.0 + sf[t]))
            val += c * isf
        scores.append(val)
    # length-normalize to avoid bias
    if scores:
        m = max(scores) or 1.0
        scores = [s / m for s in scores]
    return scores


# ---------- improved version ----------

def sentence_tf_isf_scores_v2(
    sentences: List[str],
    *,
    use_stopwords: bool = True,
    use_sublinear_tf: bool = True,
    use_bigrams: bool = False,
) -> List[float]:
    """Improved TF-ISF with stop-word filtering, punctuation removal,
    sublinear TF weighting, and optional bigram support.

    Parameters
    ----------
    use_stopwords : bool
        Remove common English stop-words before scoring.
    use_sublinear_tf : bool
        Use ``1 + log(tf)`` instead of raw tf.
    use_bigrams : bool
        Also count bigrams as scoring features.
    """
    toks_per_sent = [_tokenize(s, use_stopwords=use_stopwords) for s in sentences]

    # optional: append bigrams
    if use_bigrams:
        for i, toks in enumerate(toks_per_sent):
            bigrams = [f"{toks[j]}_{toks[j+1]}" for j in range(len(toks) - 1)]
            toks_per_sent[i] = toks + bigrams

    N = max(1, len(toks_per_sent))

    sf: Counter = Counter()
    for toks in toks_per_sent:
        for t in set(toks):
            sf[t] += 1

    scores: List[float] = []
    for toks in toks_per_sent:
        if not toks:
            scores.append(0.0)
            continue
        tf = Counter(toks)
        val = 0.0
        for t, c in tf.items():
            tf_weight = (1.0 + math.log(c)) if (use_sublinear_tf and c > 0) else float(c)
            isf = math.log(N / (1.0 + sf[t]))
            val += tf_weight * isf
        # length-normalize by sqrt(len) to reduce long-sentence bias
        val /= math.sqrt(len(toks))
        scores.append(val)

    # min-max normalize
    if scores:
        lo = min(scores)
        hi = max(scores)
        if hi - lo > 1e-9:
            scores = [(s - lo) / (hi - lo) for s in scores]
        else:
            scores = [0.5 for _ in scores]
    return scores
