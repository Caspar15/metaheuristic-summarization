"""
Advanced Stage 2 Sentence Selection Optimizer
Implements multiple optimization techniques to improve ROUGE scores
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from scipy.optimize import differential_evolution
import pickle
import re
from collections import Counter
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade

class AdvancedSentenceSelector:
    """
    Enhanced Stage 2 sentence selection with multiple optimization strategies
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize models
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Load spacy model for linguistic features
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Feature weights (will be optimized)
        self.feature_weights = {
            'position': 0.3,
            'length': 0.15,
            'tf_idf': 0.25,
            'semantic_similarity': 0.2,
            'diversity': 0.1,
            'readability': 0.05,
            'named_entities': 0.1,
            'discourse_markers': 0.05
        }
        
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'max_summary_length': 100,  # words
            'min_sentence_length': 5,   # words
            'max_sentence_length': 50,  # words
            'diversity_threshold': 0.8,
            'position_boost_factor': 2.0,
            'semantic_threshold': 0.3,
            'use_metaheuristic': True,
            'optimization_method': 'differential_evolution'  # or 'genetic_algorithm', 'grasp'
        }
    
    def extract_features(self, sentences: List[str], document: str) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive features for sentence selection
        """
        features = {}
        n_sentences = len(sentences)
        
        # 1. Position-based features
        features['position'] = self._position_features(n_sentences)
        
        # 2. Length-based features
        features['length'] = self._length_features(sentences)
        
        # 3. TF-IDF features
        features['tf_idf'] = self._tfidf_features(sentences, document)
        
        # 4. Semantic similarity features
        features['semantic_similarity'] = self._semantic_features(sentences, document)
        
        # 5. Readability features
        features['readability'] = self._readability_features(sentences)
        
        # 6. Named entity features
        features['named_entities'] = self._named_entity_features(sentences)
        
        # 7. Discourse marker features
        features['discourse_markers'] = self._discourse_marker_features(sentences)
        
        return features
    
    def _position_features(self, n_sentences: int) -> np.ndarray:
        """Position-based scoring with higher weight for beginning and end"""
        positions = np.arange(n_sentences)
        
        # U-shaped distribution favoring beginning and end
        beginning_boost = np.exp(-positions / (n_sentences * 0.3))
        ending_boost = np.exp(-(n_sentences - 1 - positions) / (n_sentences * 0.3))
        
        position_scores = beginning_boost + ending_boost * 0.5
        return position_scores / np.max(position_scores)
    
    def _length_features(self, sentences: List[str]) -> np.ndarray:
        """Length-based features with optimal length preference"""
        lengths = np.array([len(sentence.split()) for sentence in sentences])
        
        # Prefer sentences of moderate length (10-25 words)
        optimal_length = 17
        length_scores = 1.0 / (1.0 + np.abs(lengths - optimal_length) / optimal_length)
        
        # Penalize very short or very long sentences
        length_scores[lengths < self.config['min_sentence_length']] *= 0.1
        length_scores[lengths > self.config['max_sentence_length']] *= 0.3
        
        return length_scores
    
    def _tfidf_features(self, sentences: List[str], document: str) -> np.ndarray:
        """Enhanced TF-IDF features"""
        all_text = sentences + [document]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_text)
        
        sentence_vectors = tfidf_matrix[:-1]
        document_vector = tfidf_matrix[-1]
        
        # Similarity to document
        similarities = cosine_similarity(sentence_vectors, document_vector).flatten()
        
        # TF-IDF sum for each sentence
        tfidf_sums = np.array(sentence_vectors.sum(axis=1)).flatten()
        
        # Combine both metrics
        combined_scores = 0.7 * similarities + 0.3 * (tfidf_sums / np.max(tfidf_sums))
        
        return combined_scores
    
    def _semantic_features(self, sentences: List[str], document: str) -> np.ndarray:
        """Semantic similarity using sentence transformers"""
        sentence_embeddings = self.semantic_model.encode(sentences)
        document_embedding = self.semantic_model.encode([document])
        
        # Similarity to document
        doc_similarities = cosine_similarity(sentence_embeddings, document_embedding).flatten()
        
        # Centrality in sentence graph
        sentence_similarities = cosine_similarity(sentence_embeddings)
        centrality_scores = np.sum(sentence_similarities, axis=1) / len(sentences)
        
        # Combine document similarity and centrality
        semantic_scores = 0.6 * doc_similarities + 0.4 * centrality_scores
        
        return semantic_scores
    
    def _readability_features(self, sentences: List[str]) -> np.ndarray:
        """Readability-based features"""
        readability_scores = []
        
        for sentence in sentences:
            try:
                # Flesch reading ease (higher is more readable)
                ease_score = flesch_reading_ease(sentence)
                # Normalize to 0-1 range
                normalized_score = min(max(ease_score / 100.0, 0), 1)
                readability_scores.append(normalized_score)
            except:
                readability_scores.append(0.5)  # Default score
        
        return np.array(readability_scores)
    
    def _named_entity_features(self, sentences: List[str]) -> np.ndarray:
        """Named entity density features"""
        if self.nlp is None:
            return np.ones(len(sentences)) * 0.5
        
        entity_scores = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            entities = [ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']]
            
            # Entity density
            entity_density = len(entities) / max(len(sentence.split()), 1)
            entity_scores.append(min(entity_density * 2, 1.0))  # Cap at 1.0
        
        return np.array(entity_scores)
    
    def _discourse_marker_features(self, sentences: List[str]) -> np.ndarray:
        """Discourse marker features for coherence"""
        discourse_markers = {
            'conclusion': ['therefore', 'thus', 'consequently', 'in conclusion', 'finally'],
            'contrast': ['however', 'nevertheless', 'nonetheless', 'on the other hand', 'in contrast'],
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'emphasis': ['indeed', 'in fact', 'notably', 'importantly', 'significantly']
        }
        
        marker_scores = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            for category, markers in discourse_markers.items():
                for marker in markers:
                    if marker in sentence_lower:
                        if category == 'conclusion':
                            score += 0.3
                        elif category == 'emphasis':
                            score += 0.2
                        else:
                            score += 0.1
            
            marker_scores.append(min(score, 1.0))
        
        return np.array(marker_scores)
    
    def calculate_diversity_penalty(self, selected_sentences: List[str], 
                                  candidate_sentence: str) -> float:
        """Calculate diversity penalty to avoid redundancy"""
        if not selected_sentences:
            return 0.0
        
        # Semantic diversity
        candidate_embedding = self.semantic_model.encode([candidate_sentence])
        selected_embeddings = self.semantic_model.encode(selected_sentences)
        
        similarities = cosine_similarity(candidate_embedding, selected_embeddings).flatten()
        max_similarity = np.max(similarities)
        
        # High penalty if too similar to already selected sentences
        if max_similarity > self.config['diversity_threshold']:
            return (max_similarity - self.config['diversity_threshold']) * 2.0
        
        return 0.0
    
    def greedy_selection(self, sentences: List[str], document: str) -> List[int]:
        """Greedy sentence selection with diversity constraints"""
        features = self.extract_features(sentences, document)
        
        # Calculate composite scores
        composite_scores = np.zeros(len(sentences))
        for feature_name, feature_values in features.items():
            weight = self.feature_weights.get(feature_name, 0.1)
            composite_scores += weight * feature_values
        
        selected_indices = []
        selected_sentences = []
        total_words = 0
        
        # Greedy selection
        remaining_indices = set(range(len(sentences)))
        
        while remaining_indices and total_words < self.config['max_summary_length']:
            best_score = -1
            best_idx = -1
            
            for idx in remaining_indices:
                sentence = sentences[idx]
                sentence_words = len(sentence.split())
                
                if total_words + sentence_words > self.config['max_summary_length']:
                    continue
                
                # Base score
                score = composite_scores[idx]
                
                # Diversity penalty
                diversity_penalty = self.calculate_diversity_penalty(
                    selected_sentences, sentence
                )
                score -= diversity_penalty
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            selected_indices.append(best_idx)
            selected_sentences.append(sentences[best_idx])
            total_words += len(sentences[best_idx].split())
            remaining_indices.remove(best_idx)
        
        return sorted(selected_indices)
    
    def metaheuristic_optimization(self, sentences: List[str], document: str) -> List[int]:
        """
        Metaheuristic optimization using Differential Evolution
        """
        features = self.extract_features(sentences, document)
        n_sentences = len(sentences)
        
        def objective_function(solution_vector):
            """Objective function to maximize (we'll minimize the negative)"""
            # Convert continuous solution to binary selection
            threshold = 0.5
            selected_mask = solution_vector > threshold
            selected_indices = np.where(selected_mask)[0]
            
            if len(selected_indices) == 0:
                return 1000  # High penalty for empty selection
            
            # Calculate total words
            total_words = sum(len(sentences[i].split()) for i in selected_indices)
            if total_words > self.config['max_summary_length']:
                # Penalty for exceeding length
                excess_penalty = (total_words - self.config['max_summary_length']) * 10
                return 1000 + excess_penalty
            
            # Calculate feature-based score
            score = 0
            for feature_name, feature_values in features.items():
                weight = self.feature_weights.get(feature_name, 0.1)
                feature_score = np.mean(feature_values[selected_indices])
                score += weight * feature_score
            
            # Diversity bonus
            if len(selected_indices) > 1:
                selected_embeddings = self.semantic_model.encode(
                    [sentences[i] for i in selected_indices]
                )
                pairwise_similarities = cosine_similarity(selected_embeddings)
                avg_similarity = np.mean(pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)])
                diversity_bonus = max(0, 1 - avg_similarity)
                score += 0.2 * diversity_bonus
            
            return -score  # Minimize negative score
        
        # Differential Evolution optimization
        bounds = [(0, 1)] * n_sentences
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42
        )
        
        # Convert solution to selected indices
        threshold = 0.5
        selected_mask = result.x > threshold
        selected_indices = list(np.where(selected_mask)[0])
        
        # Ensure we don't exceed word limit
        selected_sentences = [(i, sentences[i]) for i in selected_indices]
        selected_sentences.sort(key=lambda x: features['tf_idf'][x[0]], reverse=True)
        
        final_selection = []
        total_words = 0
        
        for idx, sentence in selected_sentences:
            sentence_words = len(sentence.split())
            if total_words + sentence_words <= self.config['max_summary_length']:
                final_selection.append(idx)
                total_words += sentence_words
        
        return sorted(final_selection)
    
    def grasp_optimization(self, sentences: List[str], document: str, 
                          alpha: float = 0.3, max_iterations: int = 50) -> List[int]:
        """
        GRASP (Greedy Randomized Adaptive Search Procedure) optimization
        """
        features = self.extract_features(sentences, document)
        best_solution = []
        best_score = -np.inf
        
        for iteration in range(max_iterations):
            # Construction phase
            solution = self._grasp_construction(sentences, features, alpha)
            
            # Local search phase
            solution = self._grasp_local_search(sentences, features, solution)
            
            # Evaluate solution
            score = self._evaluate_solution(sentences, features, solution)
            
            if score > best_score:
                best_score = score
                best_solution = solution.copy()
        
        return sorted(best_solution)
    
    def _grasp_construction(self, sentences: List[str], features: Dict, 
                           alpha: float) -> List[int]:
        """GRASP construction phase"""
        solution = []
        candidates = list(range(len(sentences)))
        total_words = 0
        
        while candidates and total_words < self.config['max_summary_length']:
            # Calculate greedy function values
            greedy_values = []
            for idx in candidates:
                sentence_words = len(sentences[idx].split())
                if total_words + sentence_words > self.config['max_summary_length']:
                    greedy_values.append(-np.inf)
                    continue
                
                # Calculate composite score
                score = 0
                for feature_name, feature_values in features.items():
                    weight = self.feature_weights.get(feature_name, 0.1)
                    score += weight * feature_values[idx]
                
                # Diversity penalty
                if solution:
                    diversity_penalty = self.calculate_diversity_penalty(
                        [sentences[i] for i in solution], sentences[idx]
                    )
                    score -= diversity_penalty
                
                greedy_values.append(score)
            
            greedy_values = np.array(greedy_values)
            
            # Create restricted candidate list (RCL)
            if np.all(greedy_values == -np.inf):
                break
            
            g_max = np.max(greedy_values)
            g_min = np.min(greedy_values[greedy_values != -np.inf])
            threshold = g_max - alpha * (g_max - g_min)
            
            rcl = [i for i, val in enumerate(greedy_values) if val >= threshold]
            
            if not rcl:
                break
            
            # Randomly select from RCL
            selected_idx = np.random.choice(rcl)
            actual_idx = candidates[selected_idx]
            
            solution.append(actual_idx)
            total_words += len(sentences[actual_idx].split())
            candidates.remove(actual_idx)
        
        return solution
    
    def _grasp_local_search(self, sentences: List[str], features: Dict, 
                           solution: List[int]) -> List[int]:
        """GRASP local search phase"""
        improved = True
        
        while improved:
            improved = False
            current_score = self._evaluate_solution(sentences, features, solution)
            
            # Try swapping sentences
            for i in range(len(solution)):
                for candidate in range(len(sentences)):
                    if candidate in solution:
                        continue
                    
                    # Try replacing solution[i] with candidate
                    new_solution = solution.copy()
                    new_solution[i] = candidate
                    
                    # Check word limit
                    total_words = sum(len(sentences[idx].split()) for idx in new_solution)
                    if total_words > self.config['max_summary_length']:
                        continue
                    
                    new_score = self._evaluate_solution(sentences, features, new_solution)
                    
                    if new_score > current_score:
                        solution = new_solution
                        current_score = new_score
                        improved = True
                        break
                
                if improved:
                    break
        
        return solution
    
    def _evaluate_solution(self, sentences: List[str], features: Dict, 
                          solution: List[int]) -> float:
        """Evaluate a solution's quality"""
        if not solution:
            return -np.inf
        
        score = 0
        
        # Feature-based score
        for feature_name, feature_values in features.items():
            weight = self.feature_weights.get(feature_name, 0.1)
            feature_score = np.mean(feature_values[solution])
            score += weight * feature_score
        
        # Diversity bonus
        if len(solution) > 1:
            selected_sentences = [sentences[i] for i in solution]
            selected_embeddings = self.semantic_model.encode(selected_sentences)
            pairwise_similarities = cosine_similarity(selected_embeddings)
            avg_similarity = np.mean(pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)])
            diversity_bonus = max(0, 1 - avg_similarity)
            score += 0.2 * diversity_bonus
        
        return score
    
    def optimize_weights(self, training_data: List[Dict], rouge_evaluator) -> Dict:
        """
        Optimize feature weights using training data and ROUGE scores
        """
        def weight_objective(weights):
            """Objective function for weight optimization"""
            # Update weights
            weight_names = list(self.feature_weights.keys())
            for i, weight in enumerate(weights):
                self.feature_weights[weight_names[i]] = weight
            
            total_rouge = 0
            for data_point in training_data:
                sentences = data_point['sentences']
                document = data_point['document']
                reference_summary = data_point['reference_summary']
                
                # Generate summary
                selected_indices = self.select_sentences(sentences, document)
                generated_summary = ' '.join([sentences[i] for i in selected_indices])
                
                # Calculate ROUGE score
                rouge_score = rouge_evaluator.evaluate(generated_summary, reference_summary)
                total_rouge += rouge_score['rouge-l']['f']
            
            avg_rouge = total_rouge / len(training_data)
            return -avg_rouge  # Minimize negative ROUGE
        
        # Optimize weights
        initial_weights = list(self.feature_weights.values())
        bounds = [(0.01, 1.0)] * len(initial_weights)
        
        result = differential_evolution(
            weight_objective,
            bounds,
            maxiter=30,
            popsize=10,
            seed=42
        )
        
        # Update weights with optimized values
        weight_names = list(self.feature_weights.keys())
        optimized_weights = {}
        for i, weight in enumerate(result.x):
            optimized_weights[weight_names[i]] = weight
            self.feature_weights[weight_names[i]] = weight
        
        return optimized_weights
    
    def select_sentences(self, sentences: List[str], document: str) -> List[int]:
        """
        Main sentence selection method
        """
        if self.config['use_metaheuristic']:
            if self.config['optimization_method'] == 'differential_evolution':
                return self.metaheuristic_optimization(sentences, document)
            elif self.config['optimization_method'] == 'grasp':
                return self.grasp_optimization(sentences, document)
        
        return self.greedy_selection(sentences, document)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'feature_weights': self.feature_weights,
            'config': self.config,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.feature_weights = model_data['feature_weights']
        self.config = model_data['config']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']


# Example usage and evaluation
class Stage2Evaluator:
    """Evaluation framework for Stage 2 optimization"""
    
    def __init__(self, selector: AdvancedSentenceSelector):
        self.selector = selector
    
    def ablation_study(self, test_data: List[Dict], rouge_evaluator) -> Dict:
        """
        Perform ablation study to identify important features
        """
        results = {}
        original_weights = self.selector.feature_weights.copy()
        
        # Test with all features
        all_features_score = self._evaluate_on_dataset(test_data, rouge_evaluator)
        results['all_features'] = all_features_score
        
        # Test removing each feature one by one
        for feature_name in original_weights.keys():
            # Set feature weight to 0
            self.selector.feature_weights[feature_name] = 0
            
            # Evaluate
            score = self._evaluate_on_dataset(test_data, rouge_evaluator)
            results[f'without_{feature_name}'] = score
            
            # Restore original weight
            self.selector.feature_weights[feature_name] = original_weights[feature_name]
        
        return results
    
    def _evaluate_on_dataset(self, test_data: List[Dict], rouge_evaluator) -> float:
        """Evaluate selector on a dataset"""
        total_rouge = 0
        
        for data_point in test_data:
            sentences = data_point['sentences']
            document = data_point['document']
            reference_summary = data_point['reference_summary']
            
            selected_indices = self.selector.select_sentences(sentences, document)
            generated_summary = ' '.join([sentences[i] for i in selected_indices])
            
            rouge_score = rouge_evaluator.evaluate(generated_summary, reference_summary)
            total_rouge += rouge_score['rouge-l']['f']
        
        return total_rouge / len(test_data)
    
    def compare_methods(self, test_data: List[Dict], rouge_evaluator) -> Dict:
        """Compare different optimization methods"""
        results = {}
        original_config = self.selector.config.copy()
        
        # Test greedy method
        self.selector.config['use_metaheuristic'] = False
        greedy_score = self._evaluate_on_dataset(test_data, rouge_evaluator)
        results['greedy'] = greedy_score
        
        # Test differential evolution
        self.selector.config['use_metaheuristic'] = True
        self.selector.config['optimization_method'] = 'differential_evolution'
        de_score = self._evaluate_on_dataset(test_data, rouge_evaluator)
        results['differential_evolution'] = de_score
        
        # Test GRASP
        self.selector.config['optimization_method'] = 'grasp'
        grasp_score = self._evaluate_on_dataset(test_data, rouge_evaluator)
        results['grasp'] = grasp_score
        
        # Restore original config
        self.selector.config = original_config
        
        return results


# Example usage script
if __name__ == "__main__":
    # Initialize the advanced sentence selector
    config = {
        'max_summary_length': 100,
        'use_metaheuristic': True,
        'optimization_method': 'differential_evolution'
    }
    
    selector = AdvancedSentenceSelector(config)
    
    # Example document and sentences
    document = "Your document text here..."
    sentences = [
        "First sentence of the document.",
        "Second sentence with important information.",
        "Third sentence providing context.",
        # ... more sentences
    ]
    
    # Select sentences
    selected_indices = selector.select_sentences(sentences, document)
    
    print("Selected sentence indices:", selected_indices)
    print("Selected sentences:")
    for idx in selected_indices:
        print(f"{idx}: {sentences[idx]}")