from typing import Union
import numpy as np


def cosine_similarity_matrix(X: Union[np.ndarray, "scipy.sparse.spmatrix"]) -> np.ndarray:
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(X)
    except Exception:
        # minimal fallback
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        return Xn @ Xn.T

