from typing import List, Optional
"""
Graph-based centrality features for extractive summarization.
Implements PageRank (TextRank) algorithm to identify central sentences based on similarity.
"""
import numpy as np

def compute_textrank_scores(
    similarity_matrix: np.ndarray,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    threshold: float = 0.0,
) -> List[float]:
    """
    Computes PageRank (TextRank) scores from a similarity matrix.
    
    Args:
        similarity_matrix: N x N cosine similarity matrix (numpy array).
                           Values should be non-negative.
        alpha: Damping factor (default 0.85 as per PageRank paper).
        max_iter: Maximum number of power iterations.
        tol: Convergence tolerance.
        
    Returns:
        List of scores summing to 1.0 (or close to it).
    """
    N = similarity_matrix.shape[0]
    if N == 0:
        return []
    if N == 1:
        return [1.0]

    # Thresholding: Zero out weak edges to reduce noise (Sparse Graph)
    if threshold > 0:
        similarity_matrix[similarity_matrix < threshold] = 0.0

    # Normalize rows to sum to 1 to create a stochastic matrix M
    # Avoid division by zero
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    # If a row sums to 0 (isolated node), we distribute probability evenly
    row_sums[row_sums == 0] = 1.0 
    
    M = similarity_matrix / row_sums
    
    # Power iteration
    # p_{t+1} = alpha * M^T * p_t + (1 - alpha) / N
    # Initialize uniform probability
    p = np.ones(N) / N
    teleport = np.ones(N) / N
    
    for _ in range(max_iter):
        new_p = alpha * np.dot(M.T, p) + (1 - alpha) * teleport
        # Check convergence
        if np.linalg.norm(new_p - p, 1) < tol:
            p = new_p
            break
        p = new_p
        
    return p.tolist()
