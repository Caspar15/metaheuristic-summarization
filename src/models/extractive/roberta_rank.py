import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from typing import List

class RobertaRanker:
    """
    RoBERTa-based sentence ranker for extractive summarization
    """
    def __init__(self, model_name: str = "roberta-base"):
        """
        Initialize RoBERTa ranker
        
        Args:
            model_name: RoBERTa model name (roberta-base, roberta-large, etc.)
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load tokenizer and model
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=1  # For regression/ranking
            )
            self.model.to(self.device)
            self.model.eval()
            
            print(f"RoBERTa ranker loaded: {model_name} on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load RoBERTa model {model_name}: {e}")
    
    def rank_sentences(self, sentences: List[str], candidates: List[int], 
                      context: str = "", top_k: int = 20) -> List[int]:
        """
        Rank sentences using RoBERTa
        
        Args:
            sentences: List of all sentences
            candidates: List of candidate sentence indices
            context: Document context (optional)
            top_k: Number of top sentences to return
            
        Returns:
            List of top-k sentence indices ranked by relevance
        """
        if not candidates:
            return []
        
        try:
            candidate_sentences = [sentences[i] for i in candidates]
            scores = self._score_sentences(candidate_sentences, context)
            
            # Combine candidates with scores and sort
            ranked_pairs = list(zip(candidates, scores))
            ranked_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k indices
            top_k = min(top_k, len(ranked_pairs))
            return [pair[0] for pair in ranked_pairs[:top_k]]
            
        except Exception as e:
            print(f"RoBERTa ranking failed: {e}")
            return candidates[:top_k]  # fallback to original order
    
    def _score_sentences(self, sentences: List[str], context: str = "") -> List[float]:
        """
        Score sentences using RoBERTa
        
        Args:
            sentences: List of sentences to score
            context: Optional context for scoring
            
        Returns:
            List of relevance scores
        """
        scores = []
        
        with torch.no_grad():
            for sentence in sentences:
                try:
                    # Prepare input text
                    if context:
                        input_text = f"{context} [SEP] {sentence}"
                    else:
                        input_text = sentence
                    
                    # Tokenize and encode
                    inputs = self.tokenizer(
                        input_text,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get model prediction
                    outputs = self.model(**inputs)
                    
                    # Extract score (logits for regression)
                    score = outputs.logits.squeeze().cpu().item()
                    scores.append(score)
                    
                except Exception as e:
                    print(f"Error scoring sentence: {e}")
                    scores.append(0.0)  # fallback score
        
        return scores

# Test function
def test_roberta_ranker():
    """Test the RoBERTa ranker"""
    sentences = [
        "The company announced record quarterly earnings.",
        "Stock markets showed mixed performance today.",
        "The CEO outlined the company's strategic vision.",
        "Revenue growth exceeded Wall Street expectations.",
        "New product launches are scheduled for next quarter."
    ]
    
    try:
        ranker = RobertaRanker("roberta-base")
        candidates = list(range(len(sentences)))
        context = "Financial news about technology companies"
        
        ranked = ranker.rank_sentences(sentences, candidates, context, top_k=3)
        
        print("Ranked sentences:")
        for i, idx in enumerate(ranked):
            print(f"{i+1}. {sentences[idx]}")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_roberta_ranker()
