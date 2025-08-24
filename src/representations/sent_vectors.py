from typing import List, Optional


class SentenceVectors:
    def __init__(self, method: str = "tfidf"):
        self.method = method
        self._vectorizer = None
        self._model = None

    def fit_transform(self, sentences: List[str]):
        if self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
            X = self._vectorizer.fit_transform(sentences)
            return X
        elif self.method == "sbert":
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np

                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                embs = self._model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
                return embs
            except Exception as e:
                raise RuntimeError(
                    "SBERT 未安裝或初始化失敗，請安裝 sentence-transformers/torch 後重試"
                ) from e
        else:
            raise ValueError(f"Unknown representation method: {self.method}")

    def transform(self, sentences: List[str]):
        if self.method == "tfidf":
            if self._vectorizer is None:
                return self.fit_transform(sentences)
            return self._vectorizer.transform(sentences)
        elif self.method == "sbert":
            if self._model is None:
                return self.fit_transform(sentences)
            embs = self._model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            return embs
        else:
            raise ValueError(f"Unknown representation method: {self.method}")

