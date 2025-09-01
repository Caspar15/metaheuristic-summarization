from typing import List, Optional


class CrossEncoderReranker:
    """Placeholder for a Hugging Face cross-encoder reranker.

    Intended usage:
      - Initialize with a model name from Hugging Face Hub.
      - Score candidate summaries against the full document text.
      - Return a scalar score per candidate for per-document ranking.

    Actual model loading/inference will be implemented after model selection.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        # TODO: lazy-load transformer model/tokenizer upon first use

    def score(self, document: str, summary: str) -> float:
        """Return a scalar score for (document, summary).

        Placeholder implementation; replace with actual inference.
        """
        raise NotImplementedError("CrossEncoderReranker.score is not implemented yet.")

    def score_batch(self, document: str, summaries: List[str]) -> List[float]:
        """Return scores for a batch of candidate summaries for one document.

        Placeholder implementation; replace with batched inference for speed.
        """
        raise NotImplementedError("CrossEncoderReranker.score_batch is not implemented yet.")

