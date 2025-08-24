from typing import List, Dict
import numpy as np

from src.features.tf_isf import sentence_tf_isf_scores
from src.features.length import length_scores
from src.features.position import position_scores
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix


class SupervisedScorer:
    """使用已訓練的分類器（LogReg/SVM）對句子打分。

    模型需接受特徵向量輸入（importance/length/position/centrality），
    並輸出屬於"選句"類別的機率（或分數）。
    """

    def __init__(self, model_path: str):
        import joblib

        self.model = joblib.load(model_path)

    def _build_features(self, sentences: List[str], cfg: Dict) -> np.ndarray:
        imp = sentence_tf_isf_scores(sentences)
        ln = length_scores(sentences)
        pos = position_scores(sentences)
        # centrality via TF-IDF similarity
        rep_cfg = cfg.get("representations", {})
        method = rep_cfg.get("method", "tfidf")
        vec = SentenceVectors(method=method)
        X = vec.fit_transform(sentences)
        sim = cosine_similarity_matrix(X)
        cent = (sim.mean(axis=1)).tolist()
        # stack
        feats = np.vstack([imp, ln, pos, cent]).T
        return feats.astype(float)

    def predict_scores(self, sentences: List[str], cfg: Dict) -> List[float]:
        X = self._build_features(sentences, cfg)
        # if classifier supports predict_proba, use proba of class 1, else decision_function
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 2:
                scores = proba[:, 1]
            else:
                # one-vs-rest; choose max non-zero class
                scores = proba.max(axis=1)
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            if df.ndim == 1:
                scores = df
            else:
                scores = df.max(axis=1)
        else:
            scores = self.model.predict(X)
        # normalize to [0,1]
        scores = np.asarray(scores)
        if scores.size:
            mn, mx = scores.min(), scores.max()
            denom = (mx - mn) if mx > mn else 1.0
            scores = (scores - mn) / denom
        return scores.tolist()
