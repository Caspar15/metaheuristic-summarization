import argparse
import os
from typing import Dict, List, Tuple

import numpy as np

from src.utils.io import read_jsonl, ensure_dir
from src.features.tf_isf import sentence_tf_isf_scores
from src.features.length import length_scores
from src.features.position import position_scores
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix


def _count_tokens(s: str) -> int:
    return len(s.split())


def _greedy_oracle_labels(sentences: List[str], reference: str, max_tokens: int) -> List[int]:
    """以 ROUGE-1 F 提升為準則的貪婪 oracle 標記。"""
    try:
        from rouge_score import rouge_scorer
    except Exception as e:
        raise RuntimeError("需要安裝 rouge-score 以產生 oracle 標記") from e

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    selected: List[int] = []
    cur_summary = ""
    cur_tokens = 0
    best_f = 0.0
    n = len(sentences)
    for _ in range(n):
        best_i = None
        best_gain = 0.0
        for i in range(n):
            if i in selected:
                continue
            t = _count_tokens(sentences[i])
            if cur_tokens + t > max_tokens:
                continue
            cand = (cur_summary + " " + sentences[i]).strip()
            f = scorer.score(reference, cand)["rouge1"].fmeasure
            gain = f - best_f
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        cur_summary = (cur_summary + " " + sentences[best_i]).strip()
        cur_tokens += _count_tokens(sentences[best_i])
        best_f = scorer.score(reference, cur_summary)["rouge1"].fmeasure
    labels = [1 if i in selected else 0 for i in range(n)]
    return labels


def _build_features(sentences: List[str]) -> np.ndarray:
    imp = sentence_tf_isf_scores(sentences)
    ln = length_scores(sentences)
    pos = position_scores(sentences)
    vec = SentenceVectors(method="tfidf")
    X = vec.fit_transform(sentences)
    sim = cosine_similarity_matrix(X)
    cent = (sim.mean(axis=1)).tolist()
    feats = np.vstack([imp, ln, pos, cent]).T
    return feats.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="train jsonl path")
    ap.add_argument("--out_model", required=True, help="output model path (joblib)")
    ap.add_argument("--max_tokens", type=int, default=100)
    args = ap.parse_args()

    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    cnt = 0
    for row in read_jsonl(args.input):
        sents: List[str] = row.get("sentences", [])
        ref: str = row.get("highlights", "")
        if not sents or not ref:
            continue
        labels = _greedy_oracle_labels(sents, ref, args.max_tokens)
        feats = _build_features(sents)
        X_all.append(feats)
        y_all.append(np.array(labels, dtype=int))
        cnt += 1

    if not X_all:
        raise RuntimeError("No training data prepared. Check inputs.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    # train a simple Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    clf.fit(X, y)

    ensure_dir(os.path.dirname(args.out_model) or ".")
    joblib.dump(clf, args.out_model)
    print(f"Trained supervised model on {cnt} docs, saved to {args.out_model}")


if __name__ == "__main__":
    main()

