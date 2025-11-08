"""Unit tests for ExternalInferenceAgent."""

import pytest
import numpy as np
from lexai.agents import ExternalInferenceAgent


class TestExternalInferenceAgent:
    """Test suite for ExternalInferenceAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ExternalInferenceAgent(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        self.sample_candidates = [
            {
                "text": "The Constitution of India guarantees fundamental rights to all citizens.",
                "id": 1,
                "category": "constitutional_law"
            },
            {
                "text": "Article 21 protects the right to life and personal liberty.",
                "id": 2,
                "category": "fundamental_rights"
            },
            {
                "text": "Privacy is recognized as a fundamental right under the Constitution.",
                "id": 3,
                "category": "privacy"
            },
            {
                "text": "The Supreme Court has the power of judicial review.",
                "id": 4,
                "category": "judicial_power"
            }
        ]

    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert self.agent.device == "cpu"
        assert self.agent.model is None
        assert self.agent.index is None
        assert self.agent.candidates == []

    def test_build_index_success(self):
        """Test successful index building."""
        self.agent.build_index(self.sample_candidates, text_field="text")
        
        assert self.agent.index is not None
        assert len(self.agent.candidates) == len(self.sample_candidates)
        assert self.agent.candidate_embeddings is not None
        assert self.agent.model is not None

    def test_build_index_empty_candidates(self):
        """Test that empty candidates raises ValueError."""
        with pytest.raises(ValueError, match="Candidates list cannot be empty"):
            self.agent.build_index([])

    def test_build_index_missing_text_field(self):
        """Test that missing text field raises ValueError."""
        invalid_candidates = [{"content": "test"}]
        with pytest.raises(ValueError, match="missing required field"):
            self.agent.build_index(invalid_candidates, text_field="text")

    def test_build_index_non_string_text(self):
        """Test that non-string text field raises ValueError."""
        invalid_candidates = [{"text": 123}]
        with pytest.raises(ValueError, match="non-string value"):
            self.agent.build_index(invalid_candidates, text_field="text")

    def test_infer_success(self):
        """Test successful inference."""
        self.agent.build_index(self.sample_candidates, text_field="text")
        
        results = self.agent.infer("fundamental rights", top_k=2)
        
        assert len(results) == 2
        assert all("similarity_score" in r for r in results)
        assert all("text" in r for r in results)
        assert all("id" in r for r in results)
        
        # Scores should be in descending order
        assert results[0]["similarity_score"] >= results[1]["similarity_score"]

    def test_infer_without_index(self):
        """Test that inference without index raises ValueError."""
        with pytest.raises(ValueError, match="Index has not been built"):
            self.agent.infer("test query")

    def test_infer_top_k_larger_than_candidates(self):
        """Test inference when top_k > number of candidates."""
        self.agent.build_index(self.sample_candidates[:2], text_field="text")
        
        results = self.agent.infer("test query", top_k=10)
        
        # Should return only available candidates
        assert len(results) == 2

    def test_similarity_scores_in_valid_range(self):
        """Test that similarity scores are in valid range."""
        self.agent.build_index(self.sample_candidates, text_field="text")
        
        results = self.agent.infer("constitutional rights", top_k=3)
        
        for result in results:
            score = result["similarity_score"]
            # Cosine similarity should be between -1 and 1
            assert -1.0 <= score <= 1.0

    def test_get_index_stats_before_build(self):
        """Test index stats before building index."""
        stats = self.agent.get_index_stats()
        
        assert stats["is_built"] is False
        assert stats["num_candidates"] == 0
        assert stats["embedding_dimension"] is None

    def test_get_index_stats_after_build(self):
        """Test index stats after building index."""
        self.agent.build_index(self.sample_candidates, text_field="text")
        stats = self.agent.get_index_stats()
        
        assert stats["is_built"] is True
        assert stats["num_candidates"] == len(self.sample_candidates)
        assert stats["embedding_dimension"] == 384  # all-MiniLM-L6-v2 dimension
        assert stats["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_clear_index(self):
        """Test clearing the index."""
        self.agent.build_index(self.sample_candidates, text_field="text")
        
        # Verify index is built
        assert self.agent.index is not None
        
        # Clear index
        self.agent.clear_index()
        
        # Verify index is cleared
        assert self.agent.index is None
        assert self.agent.candidates == []
        assert self.agent.candidate_embeddings is None

    def test_metadata_preservation(self):
        """Test that candidate metadata is preserved in results."""
        self.agent.build_index(self.sample_candidates, text_field="text")
        
        results = self.agent.infer("privacy rights", top_k=1)
        
        # Metadata should be preserved
        assert "id" in results[0]
        assert "category" in results[0]
        assert results[0]["category"] in ["constitutional_law", "fundamental_rights", "privacy", "judicial_power"]

    def test_different_text_field(self):
        """Test using a different text field name."""
        candidates = [
            {"content": "Text about constitutional law", "id": 1},
            {"content": "Text about fundamental rights", "id": 2}
        ]
        
        self.agent.build_index(candidates, text_field="content")
        results = self.agent.infer("law", top_k=1)
        
        assert len(results) == 1
        assert "content" in results[0]

    def test_retrieve_candidates(self):
        """Test internal _retrieve_candidates method."""
        self.agent.build_index(self.sample_candidates, text_field="text")
        
        retrieved = self.agent._retrieve_candidates("privacy", top_k=2)
        
        assert len(retrieved) == 2
        assert all(isinstance(item, tuple) for item in retrieved)
        assert all(len(item) == 2 for item in retrieved)
        
        # Check that indices are valid
        for idx, score in retrieved:
            assert 0 <= idx < len(self.sample_candidates)
            assert isinstance(score, float)

    def test_compute_similarity(self):
        """Test internal _compute_similarity method."""
        self.agent.build_index(self.sample_candidates, text_field="text")
        
        retrieved = [(0, 0.85), (1, 0.75)]
        results = self.agent._compute_similarity("test query", retrieved)
        
        assert len(results) == 2
        assert results[0]["similarity_score"] == 0.85
        assert results[1]["similarity_score"] == 0.75
        assert "text" in results[0]
        assert "id" in results[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
