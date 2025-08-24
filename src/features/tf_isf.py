from typing import List
from collections import Counter
import math


def sentence_tf_isf_scores(sentences: List[str]) -> List[float]:
    """Compute TF-ISF per sentence (sum over tokens):
    ISF(t) = log( N / (1 + n_t) ), where n_t is #sentences containing token t.
    Tokenization: simple whitespace lowercased split.
    """
    toks_per_sent = [s.lower().split() for s in sentences]
    N = max(1, len(toks_per_sent))
    # sentence frequency per token
    sf = Counter()
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

