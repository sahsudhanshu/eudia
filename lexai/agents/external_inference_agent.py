"""External Inference Agent for legal document similarity and retrieval.

Example:
  agent = ExternalInferenceAgent()
  agent.build_index(candidates_list)  # candidates loaded from data/lecai_baseline subset
  result = agent.infer(target_case, top_k=3, internal_confidence=0.8)
  print(json.dumps(result, indent=2))
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ExternalInferenceAgent:
    """
    Agent for performing similarity-based inference on legal documents.
    
    Uses sentence transformers to create embeddings and FAISS for efficient
    similarity search across a corpus of candidate documents.
    
    Computes Trust Relevance Score (TRS) based on multiple factors:
    - Similarity (S): Semantic similarity via embeddings
    - Context Fit (C): Contextual relevance via TF-IDF
    - Jurisdiction Score (J): Geographic and temporal alignment
    - Internal Confidence (I): Optional confidence from internal models
    - Uncertainty (U): Estimation of prediction uncertainty
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        trs_weights: Optional[Dict[str, float]] = None,
        retriever: Optional[Callable] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the ExternalInferenceAgent.

        Args:
            embedding_model_name: Name of the sentence-transformer model to use.
            trs_weights: Weights for TRS computation. Default: {"w_S":0.5, "w_C":0.2, "w_J":0.1, "w_I":0.15, "w_U":0.05}
            retriever: Optional custom retriever function. If None, builds FAISS index.
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect).
        """
        self.model_name = embedding_model_name
        self.device = device
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.candidates: List[Dict[str, Any]] = []
        self.candidate_embeddings: Optional[np.ndarray] = None
        self.custom_retriever = retriever
        
        # TRS weights with defaults
        if trs_weights is None:
            self.trs_weights = {
                "w_S": 0.5,   # Similarity weight
                "w_C": 0.2,   # Context fit weight
                "w_J": 0.1,   # Jurisdiction score weight
                "w_I": 0.15,  # Internal confidence weight
                "w_U": 0.05   # Uncertainty weight
            }
        else:
            self.trs_weights = trs_weights

    def build_index(
        self,
        candidates: List[Dict[str, Any]],
        text_field: str = "text"
    ) -> None:
        """
        Build FAISS index from candidate documents.

        Creates embeddings for all candidates using sentence-transformers,
        normalizes them, and builds a FAISS IndexFlatIP for inner product search.

        Args:
            candidates: List of candidate documents. Each must contain:
                - case_id, title, text (or text_field), year (optional), jurisdiction (optional)
            text_field: Key in candidate dict containing the text to embed.

        Raises:
            ValueError: If candidates is empty or missing required fields.
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")

        # Validate required fields
        required_fields = ['case_id', 'title']
        for idx, candidate in enumerate(candidates):
            for field in required_fields:
                if field not in candidate:
                    raise ValueError(
                        f"Candidate at index {idx} is missing required field '{field}'"
                    )
            
            if text_field not in candidate:
                raise ValueError(
                    f"Candidate at index {idx} is missing required field '{text_field}'"
                )
            if not isinstance(candidate[text_field], str):
                raise ValueError(
                    f"Candidate at index {idx} has non-string value for '{text_field}'"
                )

        # Initialize model if not already done
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

        # Extract texts from candidates
        texts = [candidate[text_field] for candidate in candidates]

        # Create embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=False  # We'll normalize manually
        )

        # Normalize embeddings for cosine similarity (using inner product)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-10)  # Avoid division by zero

        # Build FAISS index
        dimension = normalized_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors = cosine similarity
        self.index.add(normalized_embeddings.astype('float32'))

        # Store candidates and embeddings
        self.candidates = candidates
        self.candidate_embeddings = normalized_embeddings

    def infer(
        self,
        target: Dict[str, Any],
        top_k: int = 5,
        internal_confidence: Optional[float] = None,
        return_factors: bool = False
    ) -> Dict[str, Any]:
        """
        Perform inference to find similar candidates for a target case.

        Args:
            target: Target case dict with keys: case_id, title, text, year (optional), jurisdiction (optional)
            top_k: Number of top candidates to retrieve.
            internal_confidence: Optional internal confidence score [0,1]
            return_factors: If True, include individual factor scores in output

        Returns:
            Dictionary matching OUTPUT_SCHEMA with retrieved cases sorted by TRS descending.

        Raises:
            ValueError: If index has not been built yet or target is missing required fields.
        """
        if self.index is None or not self.candidates:
            raise ValueError(
                "Index has not been built. Call build_index() first."
            )
        
        # Validate target
        required_fields = ['case_id', 'title', 'text']
        for field in required_fields:
            if field not in target:
                raise ValueError(f"Target is missing required field '{field}'")
        
        target_text = target['text']
        target_year = target.get('year', None)
        target_jurisdiction = target.get('jurisdiction', 'Unknown')
        
        # Use custom retriever if provided, otherwise use FAISS
        if self.custom_retriever is not None:
            retrieved_candidates = self.custom_retriever(target_text, top_k)
        else:
            retrieved_candidates = self._retrieve_candidates(target_text, top_k)
        
        # Encode target for similarity computation
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        
        target_embedding = self.model.encode([target_text], convert_to_numpy=True, normalize_embeddings=False)
        target_norm = np.linalg.norm(target_embedding, axis=1, keepdims=True)
        target_embedding = target_embedding / (target_norm + 1e-10)
        
        # Process each retrieved candidate
        retrieved_cases = []
        for candidate_idx, similarity_score in retrieved_candidates:
            candidate = self.candidates[candidate_idx].copy()
            candidate_text = candidate['text']
            candidate_year = candidate.get('year', None)
            candidate_jurisdiction = candidate.get('jurisdiction', 'Unknown')
            
            # Compute individual scores
            S = float(similarity_score)  # Already computed by FAISS
            C = self._estimate_context_fit(target_text, candidate_text)
            J = self._compute_jurisdiction_score(target_jurisdiction, candidate_jurisdiction, target_year, candidate_year)
            I = internal_confidence if internal_confidence is not None else 0.0
            U = self._estimate_uncertainty(S, C)
            
            # Compute TRS
            trs = self._compute_trs(S, C, J, I, U, return_factors=return_factors)
            
            # Determine alignment
            alignment_type = self._determine_alignment(target_text, candidate_text, S)
            
            # Extract supporting spans
            target_span, candidate_span = self._extract_support_spans(target_text, candidate_text)
            
            # Create justification
            justification = self._generate_justification(S, C, J, alignment_type, candidate_year, candidate_jurisdiction)
            
            # Build result dict
            result_case = {
                "case_id": candidate['case_id'],
                "title": candidate['title'],
                "year": candidate_year if candidate_year is not None else "N/A",
                "jurisdiction": candidate_jurisdiction,
                "similarity_score": float(np.clip(S, 0, 1)),
                "context_fit": float(np.clip(C, 0, 1)),
                "jurisdiction_score": float(np.clip(J, 0, 1)),
                "internal_confidence": float(np.clip(I, 0, 1)),
                "uncertainty": float(np.clip(U, 0, 1)),
                "trs": float(trs) if not return_factors else trs,
                "alignment_type": alignment_type,
                "justification": justification,
                "spans": {
                    "target_span": target_span,
                    "candidate_span": candidate_span
                }
            }
            
            retrieved_cases.append(result_case)
        
        # Sort by TRS (extract float if dict)
        retrieved_cases.sort(key=lambda x: x['trs'] if isinstance(x['trs'], float) else x['trs']['score'], reverse=True)
        
        # Compute overall external coherence score
        if retrieved_cases:
            trs_scores = [c['trs'] if isinstance(c['trs'], float) else c['trs']['score'] for c in retrieved_cases]
            overall_score = float(np.mean(trs_scores))
        else:
            overall_score = 0.0
        
        # Generate summary
        summary = self._generate_summary(target, retrieved_cases, overall_score)
        
        # Build final result
        result = {
            "target": {
                "case_id": target['case_id'],
                "title": target['title'],
                "year": target_year if target_year is not None else "N/A",
                "jurisdiction": target_jurisdiction
            },
            "retrieved_cases": retrieved_cases,
            "overall_external_coherence_score": float(np.clip(overall_score, 0, 1)),
            "short_summary": summary
        }
        
        return result

    def _retrieve_candidates(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Retrieve candidates using FAISS index.

        Args:
            query: Query text.
            top_k: Number of candidates to retrieve.

        Returns:
            List of tuples (candidate_index, distance_score).
        """
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

        # Encode and normalize query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        normalized_query = query_embedding / (query_norm + 1e-10)

        # Search FAISS index
        distances, indices = self.index.search(
            normalized_query.astype('float32'),
            min(top_k, len(self.candidates))
        )

        # Return list of (index, distance) tuples
        results = [
            (int(indices[0][i]), float(distances[0][i]))
            for i in range(len(indices[0]))
        ]

        return results

    def _estimate_context_fit(self, target_text: str, candidate_text: str) -> float:
        """
        Estimate context fit using TF-IDF cosine similarity.

        Args:
            target_text: Target case text.
            candidate_text: Candidate case text.

        Returns:
            Context fit score [0, 1].
        """
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([target_text, candidate_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(np.clip(similarity, 0, 1))
        except:
            # Fallback: simple word overlap
            target_words = set(target_text.lower().split())
            candidate_words = set(candidate_text.lower().split())
            if not target_words or not candidate_words:
                return 0.0
            intersection = len(target_words & candidate_words)
            union = len(target_words | candidate_words)
            return float(intersection / union) if union > 0 else 0.0
    
    def _compute_jurisdiction_score(
        self,
        target_jur: str,
        cand_jur: str,
        target_year: Optional[int],
        cand_year: Optional[int]
    ) -> float:
        """
        Compute jurisdiction alignment score.

        Args:
            target_jur: Target jurisdiction.
            cand_jur: Candidate jurisdiction.
            target_year: Target year (optional).
            cand_year: Candidate year (optional).

        Returns:
            Jurisdiction score [0, 1].
        """
        # Jurisdiction match
        jur_score = 1.0 if target_jur.lower() == cand_jur.lower() else 0.5
        
        # Temporal proximity
        if target_year is not None and cand_year is not None:
            year_diff = abs(target_year - cand_year)
            # Exponential decay: closer years get higher scores
            temporal_score = np.exp(-year_diff / 20.0)  # 20-year half-life
            # Combine jurisdiction and temporal scores
            score = 0.7 * jur_score + 0.3 * temporal_score
        else:
            score = jur_score * 0.7  # Reduce score if year info missing
        
        return float(np.clip(score, 0, 1))
    
    def _estimate_uncertainty(self, similarity: float, context_fit: float) -> float:
        """
        Estimate uncertainty based on score variance.

        Args:
            similarity: Similarity score.
            context_fit: Context fit score.

        Returns:
            Uncertainty score [0, 1]. Lower is better.
        """
        # High variance between S and C suggests uncertainty
        variance = (similarity - context_fit) ** 2
        uncertainty = min(variance, 1.0)
        return float(uncertainty)
    
    def _compute_trs(
        self,
        S: float,
        C: float,
        J: float,
        I: float,
        U: float,
        return_factors: bool = False
    ) -> float:
        """
        Compute Trust Relevance Score using self.trs_weights.

        Args:
            S: Similarity score [0, 1]
            C: Context fit score [0, 1]
            J: Jurisdiction score [0, 1]
            I: Internal confidence [0, 1]
            U: Uncertainty [0, 1] (lower is better)

        Returns:
            TRS score clipped to [0, 1], or dict with factors if return_factors=True
        """
        w_S = self.trs_weights.get("w_S", 0.5)
        w_C = self.trs_weights.get("w_C", 0.2)
        w_J = self.trs_weights.get("w_J", 0.1)
        w_I = self.trs_weights.get("w_I", 0.15)
        w_U = self.trs_weights.get("w_U", 0.05)
        
        # TRS formula: weighted sum, uncertainty is subtracted
        trs = (w_S * S) + (w_C * C) + (w_J * J) + (w_I * I) - (w_U * U)
        
        # Clip to [0, 1]
        trs = float(np.clip(trs, 0, 1))
        
        if return_factors:
            return {
                "score": trs,
                "factors": {
                    "similarity": float(S),
                    "context_fit": float(C),
                    "jurisdiction_score": float(J),
                    "internal_confidence": float(I),
                    "uncertainty": float(U)
                },
                "weights": self.trs_weights
            }
        
        return trs
    
    def _determine_alignment(self, target_text: str, candidate_text: str, similarity: float) -> str:
        """
        Determine alignment type using heuristics.

        Args:
            target_text: Target case text.
            candidate_text: Candidate case text.
            similarity: Similarity score.

        Returns:
            Alignment type: "supports", "contradicts", or "neutral"
        """
        # Simple heuristic based on similarity and keyword analysis
        if similarity >= 0.7:
            # Check for contradiction keywords
            contradiction_keywords = ['however', 'but', 'overruled', 'reversed', 'contrary', 'distinguished']
            candidate_lower = candidate_text.lower()
            
            if any(keyword in candidate_lower for keyword in contradiction_keywords):
                return "contradicts"
            else:
                return "supports"
        elif similarity >= 0.4:
            return "neutral"
        else:
            return "neutral"
    
    def _extract_support_spans(self, target_text: str, candidate_text: str) -> Tuple[str, str]:
        """
        Extract supporting spans (<=40 words) from target and candidate.

        Args:
            target_text: Target case text.
            candidate_text: Candidate case text.

        Returns:
            Tuple of (target_span, candidate_span), each <=40 words.
        """
        # Split into sentences
        target_sentences = re.split(r'[.!?]+', target_text)
        candidate_sentences = re.split(r'[.!?]+', candidate_text)
        
        # Clean sentences
        target_sentences = [s.strip() for s in target_sentences if s.strip()]
        candidate_sentences = [s.strip() for s in candidate_sentences if s.strip()]
        
        if not target_sentences or not candidate_sentences:
            return "N/A", "N/A"
        
        # Use TF-IDF to find most similar sentence pair
        try:
            all_sentences = target_sentences + candidate_sentences
            vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
            vectors = vectorizer.fit_transform(all_sentences)
            
            target_vectors = vectors[:len(target_sentences)]
            candidate_vectors = vectors[len(target_sentences):]
            
            # Find best matching sentence pair
            similarities = cosine_similarity(target_vectors, candidate_vectors)
            best_idx = np.unravel_index(similarities.argmax(), similarities.shape)
            
            target_span = target_sentences[best_idx[0]]
            candidate_span = candidate_sentences[best_idx[1]]
        except:
            # Fallback: use first sentence
            target_span = target_sentences[0]
            candidate_span = candidate_sentences[0]
        
        # Truncate to 40 words
        target_span = ' '.join(target_span.split()[:40])
        candidate_span = ' '.join(candidate_span.split()[:40])
        
        return target_span, candidate_span
    
    def _generate_justification(
        self,
        similarity: float,
        context_fit: float,
        jurisdiction_score: float,
        alignment_type: str,
        year: Optional[int],
        jurisdiction: str
    ) -> str:
        """
        Generate a short justification for the retrieval.

        Args:
            similarity: Similarity score.
            context_fit: Context fit score.
            jurisdiction_score: Jurisdiction score.
            alignment_type: Alignment type.
            year: Candidate year.
            jurisdiction: Candidate jurisdiction.

        Returns:
            Justification string (1-3 sentences).
        """
        justifications = []
        
        if similarity >= 0.7:
            justifications.append(f"High semantic similarity ({similarity:.2f}) indicates strong relevance.")
        elif similarity >= 0.5:
            justifications.append(f"Moderate semantic similarity ({similarity:.2f}) suggests potential relevance.")
        else:
            justifications.append(f"Lower similarity ({similarity:.2f}) indicates limited semantic overlap.")
        
        if jurisdiction_score >= 0.8:
            justifications.append(f"Strong jurisdictional alignment ({jurisdiction}, {year}).")
        elif year is None or jurisdiction == 'Unknown':
            justifications.append("Limited metadata available for jurisdictional assessment.")
        
        justifications.append(f"This case {alignment_type} the target case's reasoning.")
        
        return " ".join(justifications)
    
    def _generate_summary(
        self,
        target: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        overall_score: float
    ) -> str:
        """
        Generate a one-paragraph summary of the inference result.

        Args:
            target: Target case dict.
            retrieved_cases: List of retrieved cases.
            overall_score: Overall external coherence score.

        Returns:
            Summary paragraph.
        """
        num_cases = len(retrieved_cases)
        num_supporting = sum(1 for c in retrieved_cases if c['alignment_type'] == 'supports')
        num_contradicting = sum(1 for c in retrieved_cases if c['alignment_type'] == 'contradicts')
        
        summary = f"Analysis of '{target['title']}' retrieved {num_cases} relevant cases "
        summary += f"with an overall external coherence score of {overall_score:.2f}. "
        
        if num_supporting > 0:
            summary += f"{num_supporting} case(s) support the target reasoning. "
        if num_contradicting > 0:
            summary += f"{num_contradicting} case(s) present contradicting perspectives. "
        
        if overall_score >= 0.7:
            summary += "The external corpus strongly validates the target case's legal reasoning."
        elif overall_score >= 0.5:
            summary += "The external corpus provides moderate support for the target case's legal reasoning."
        else:
            summary += "The external corpus shows limited alignment with the target case's legal reasoning."
        
        return summary

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Dictionary with index statistics.
        """
        if self.index is None:
            return {
                "is_built": False,
                "num_candidates": 0,
                "embedding_dimension": None
            }

        return {
            "is_built": True,
            "num_candidates": len(self.candidates),
            "embedding_dimension": self.index.d,
            "model_name": self.model_name
        }

    def clear_index(self) -> None:
        """Clear the current index and candidate data."""
        self.index = None
        self.candidates = []
        self.candidate_embeddings = None


# ============================================================================
# Demo and Testing Functions
# ============================================================================

def create_sample_data() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create sample candidate and target data for testing.
    
    Returns:
        Tuple of (candidates_list, target_case)
    """
    candidates = [
        {
            "case_id": "CASE_001",
            "title": "K.S. Puttaswamy v. Union of India",
            "text": "The right to privacy is a fundamental right under Article 21 of the Constitution. "
                   "Privacy includes preservation of personal intimacies, the sanctity of family life, "
                   "marriage, procreation, the home and sexual orientation. Privacy also connotes a right "
                   "to be left alone. Privacy safeguards individual autonomy and recognises the ability of "
                   "the individual to control vital aspects of his or her life.",
            "year": 2017,
            "jurisdiction": "Supreme Court of India"
        },
        {
            "case_id": "CASE_002",
            "title": "Maneka Gandhi v. Union of India",
            "text": "Article 21 confers a fundamental right to life and personal liberty. The right to life "
                   "is not merely confined to physical existence but includes the right to live with human "
                   "dignity and all that goes along with it. The procedure established by law must be right, "
                   "just and fair, not arbitrary, fanciful or oppressive.",
            "year": 1978,
            "jurisdiction": "Supreme Court of India"
        },
        {
            "case_id": "CASE_003",
            "title": "Gobind v. State of Madhya Pradesh",
            "text": "The right to privacy in any event will necessarily have to go through a process of "
                   "case-by-case development. The claim to privacy must be weighed against other important "
                   "interests such as public interest. A mere announcement of a privilege against the State "
                   "will not conclude the matter but the question would be whether in a given situation "
                   "a claimed right is entitled to protection.",
            "year": 1975,
            "jurisdiction": "Supreme Court of India"
        },
        {
            "case_id": "CASE_004",
            "title": "Kharak Singh v. State of U.P.",
            "text": "The right to privacy is not a guaranteed right under our Constitution. However, domiciliary "
                   "visits by the police without warrant are an intrusion into a person's right to be left alone. "
                   "The right to personal liberty takes in not only a right to be free from restrictions placed "
                   "on his movements, but also free from encroachments on his private life.",
            "year": 1963,
            "jurisdiction": "Supreme Court of India"
        },
        {
            "case_id": "CASE_005",
            "title": "Justice K.S. Puttaswamy (Retd.) v. Union of India (Aadhaar)",
            "text": "Privacy is a constitutionally protected right which emerges primarily from the guarantee "
                   "of life and personal liberty in Article 21 of the Constitution. Elements of privacy also "
                   "arise from the other facets of freedom and dignity recognised by the fundamental rights "
                   "contained in Part III of the Constitution. The right to privacy must be balanced against "
                   "legitimate state interests and public welfare.",
            "year": 2018,
            "jurisdiction": "Supreme Court of India"
        }
    ]
    
    target = {
        "case_id": "TARGET_001",
        "title": "Test Case on Privacy Rights",
        "text": "The question before this Court is whether the right to privacy is a fundamental right "
               "guaranteed under Part III of the Constitution. Privacy is essential to human dignity and "
               "autonomy. It protects individuals from state and non-state intrusions. The right to privacy "
               "must be read as a part of Article 21 which guarantees the right to life and personal liberty. "
               "This right includes the right to make intimate personal choices and control dissemination "
               "of personal information.",
        "year": 2020,
        "jurisdiction": "Supreme Court of India"
    }
    
    return candidates, target


def test_build_index():
    """Test that index builds correctly."""
    print("\n" + "="*60)
    print("TEST: build_index")
    print("="*60)
    
    candidates, _ = create_sample_data()
    agent = ExternalInferenceAgent()
    
    try:
        agent.build_index(candidates)
        assert agent.index is not None, "Index should be built"
        assert agent.index.ntotal == len(candidates), f"Index should have {len(candidates)} vectors"
        print(f"✓ Index built successfully with {agent.index.ntotal} candidates")
        print(f"✓ Embedding dimension: {agent.index.d}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_infer_schema():
    """Test that infer returns correct schema."""
    print("\n" + "="*60)
    print("TEST: infer schema validation")
    print("="*60)
    
    candidates, target = create_sample_data()
    agent = ExternalInferenceAgent()
    agent.build_index(candidates)
    
    try:
        result = agent.infer(target, top_k=3, internal_confidence=0.8)
        
        # Check top-level keys
        required_keys = ['target', 'retrieved_cases', 'overall_external_coherence_score', 'short_summary']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Check target keys
        target_keys = ['case_id', 'title', 'year', 'jurisdiction']
        for key in target_keys:
            assert key in result['target'], f"Missing target key: {key}"
        
        # Check retrieved cases
        assert len(result['retrieved_cases']) > 0, "Should retrieve at least one case"
        
        case_keys = ['case_id', 'title', 'year', 'jurisdiction', 'similarity_score', 
                    'context_fit', 'jurisdiction_score', 'internal_confidence', 
                    'uncertainty', 'trs', 'alignment_type', 'justification', 'spans']
        
        for case in result['retrieved_cases']:
            for key in case_keys:
                assert key in case, f"Missing case key: {key}"
        
        print(f"✓ Schema validation passed")
        print(f"✓ Retrieved {len(result['retrieved_cases'])} cases")
        print(f"✓ All required fields present")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trs_bounds():
    """Test that TRS and other scores are bounded [0, 1]."""
    print("\n" + "="*60)
    print("TEST: TRS bounds validation")
    print("="*60)
    
    candidates, target = create_sample_data()
    agent = ExternalInferenceAgent()
    agent.build_index(candidates)
    
    try:
        result = agent.infer(target, top_k=3, internal_confidence=0.8)
        
        # Check overall score
        overall_score = result['overall_external_coherence_score']
        assert 0.0 <= overall_score <= 1.0, f"Overall score {overall_score} out of bounds"
        
        # Check each retrieved case
        for case in result['retrieved_cases']:
            trs = case['trs']
            assert 0.0 <= trs <= 1.0, f"TRS {trs} out of bounds for {case['case_id']}"
            
            assert 0.0 <= case['similarity_score'] <= 1.0, "Similarity out of bounds"
            assert 0.0 <= case['context_fit'] <= 1.0, "Context fit out of bounds"
            assert 0.0 <= case['jurisdiction_score'] <= 1.0, "Jurisdiction score out of bounds"
            assert 0.0 <= case['internal_confidence'] <= 1.0, "Internal confidence out of bounds"
            assert 0.0 <= case['uncertainty'] <= 1.0, "Uncertainty out of bounds"
        
        print(f"✓ All scores bounded to [0, 1]")
        print(f"✓ Overall coherence score: {overall_score:.3f}")
        print(f"✓ TRS range: [{min(c['trs'] for c in result['retrieved_cases']):.3f}, "
              f"{max(c['trs'] for c in result['retrieved_cases']):.3f}]")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    tests = [
        ("Build Index", test_build_index),
        ("Infer Schema", test_infer_schema),
        ("TRS Bounds", test_trs_bounds)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("="*60))
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60 + "\n")
    
    return all_passed


def demo():
    """Run a complete demo of the ExternalInferenceAgent."""
    import json
    
    print("\n" + "="*60)
    print("EXTERNAL INFERENCE AGENT DEMO")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample legal cases...")
    candidates, target = create_sample_data()
    print(f"   Created {len(candidates)} candidate cases")
    print(f"   Target case: {target['title']}")
    
    # Initialize agent
    print("\n2. Initializing ExternalInferenceAgent...")
    agent = ExternalInferenceAgent(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        trs_weights={"w_S": 0.5, "w_C": 0.2, "w_J": 0.1, "w_I": 0.15, "w_U": 0.05}
    )
    print("   Agent initialized")
    
    # Build index
    print("\n3. Building FAISS index from candidates...")
    agent.build_index(candidates)
    stats = agent.get_index_stats()
    print(f"   Index built: {stats['num_candidates']} candidates")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    
    # Run inference
    print("\n4. Running inference on target case...")
    result = agent.infer(target, top_k=3, internal_confidence=0.8)
    
    # Display results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2))
    
    print("\n" + "="*60)
    print("KEY METRICS")
    print("="*60)
    print(f"Overall External Coherence Score: {result['overall_external_coherence_score']:.3f}")
    print(f"Retrieved Cases: {len(result['retrieved_cases'])}")
    print(f"\nTop 3 Cases by TRS:")
    for i, case in enumerate(result['retrieved_cases'][:3], 1):
        print(f"\n{i}. {case['title']}")
        print(f"   TRS: {case['trs']:.3f}")
        print(f"   Similarity: {case['similarity_score']:.3f}")
        print(f"   Alignment: {case['alignment_type']}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    import sys
    import json
    
    # Check for test mode
    if "--test" in sys.argv:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        # Run demo
        result = demo()
        
        # Run quick validation
        print("\n" + "="*60)
        print("QUICK VALIDATION")
        print("="*60)
        
        # Create agent for validation
        candidates, target = create_sample_data()
        agent = ExternalInferenceAgent()
        agent.build_index(candidates)
        
        # Assertions
        assert agent.index.ntotal == len(candidates), "Index size mismatch"
        print(f"✓ Index has {agent.index.ntotal} candidates")
        
        for case in result['retrieved_cases']:
            assert 0.0 <= case['trs'] <= 1.0, f"TRS out of bounds: {case['trs']}"
        print(f"✓ All TRS scores in [0, 1]")
        
        assert 0.0 <= result['overall_external_coherence_score'] <= 1.0, "Overall score out of bounds"
        print(f"✓ Overall coherence score in [0, 1]: {result['overall_external_coherence_score']:.3f}")
        
        required_keys = ['target', 'retrieved_cases', 'overall_external_coherence_score', 'short_summary']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        print(f"✓ All required top-level keys present")
        
        print("\n" + "="*60)
        print("ALL VALIDATIONS PASSED ✓")
        print("="*60 + "\n")
