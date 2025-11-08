"""Integrated Legal AI Pipeline combining External Inference and Legal Reasoning.

This module provides a complete pipeline for legal case analysis:
1. ExternalInferenceAgent - Retrieves relevant precedent cases
2. LegalReasoningAgent - Generates comprehensive legal reasoning

Supports citation graph analysis using NetworkX.
"""

from typing import List, Dict, Any, Optional
import json
import pandas as pd
import networkx as nx
from lexai.agents import ExternalInferenceAgent, LegalReasoningAgent


class LegalAIPipeline:
    """
    Complete legal AI pipeline for case analysis and reasoning.
    
    Combines external inference (precedent retrieval) with legal reasoning
    (multi-aspect analysis) to produce comprehensive legal reports.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        trs_weights: Optional[Dict[str, float]] = None,
        use_llm_reasoning: bool = False,
        llm_model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the Legal AI Pipeline.
        
        Args:
            embedding_model: Sentence transformer model for embeddings.
            trs_weights: Custom TRS weights for external inference.
            use_llm_reasoning: Whether to use LLM for reasoning (vs rule-based).
            llm_model_name: Hugging Face model for reasoning (if use_llm_reasoning=True).
            device: Device to run models on ('cuda', 'cpu', or None).
        """
        self.device = device
        
        # Initialize External Inference Agent
        self.external_agent = ExternalInferenceAgent(
            embedding_model_name=embedding_model,
            trs_weights=trs_weights,
            device=device
        )
        
        # Initialize Legal Reasoning Agent
        self.reasoning_agent = LegalReasoningAgent(
            model_name=llm_model_name,
            use_llm=use_llm_reasoning,
            device=device
        )
        
        # Citation graph (will be built if citation data available)
        self.citation_graph: Optional[nx.DiGraph] = None
        self.candidates_df: Optional[pd.DataFrame] = None
    
    def load_lecai_dataset(
        self,
        candidates_path: str,
        citation_graph_path: Optional[str] = None
    ) -> int:
        """
        Load LecAI dataset from file.
        
        Args:
            candidates_path: Path to candidates JSONL file.
            citation_graph_path: Optional path to citation graph data.
        
        Returns:
            Number of candidates loaded.
        """
        print(f"Loading candidates from: {candidates_path}")
        
        # Load candidates
        candidates = []
        with open(candidates_path, 'r') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
        
        print(f"Loaded {len(candidates)} candidates")
        
        # Build index
        self.external_agent.build_index(candidates)
        
        # Create DataFrame for analysis
        self.candidates_df = pd.DataFrame(candidates)
        
        # Load citation graph if provided
        if citation_graph_path:
            self._load_citation_graph(citation_graph_path)
        
        return len(candidates)
    
    def _load_citation_graph(self, graph_path: str):
        """Load citation graph from file."""
        print(f"Loading citation graph from: {graph_path}")
        
        try:
            # Assume edge list format: source,target
            self.citation_graph = nx.DiGraph()
            
            with open(graph_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            source, target = parts[0], parts[1]
                            self.citation_graph.add_edge(source, target)
            
            print(f"Citation graph loaded: {self.citation_graph.number_of_nodes()} nodes, "
                  f"{self.citation_graph.number_of_edges()} edges")
        except Exception as e:
            print(f"Failed to load citation graph: {e}")
            self.citation_graph = None
    
    def analyze_case(
        self,
        target_case: Dict[str, Any],
        top_k: int = 5,
        internal_confidence: Optional[float] = None,
        use_citation_graph: bool = False
    ) -> Dict[str, Any]:
        """
        Complete case analysis pipeline.
        
        Args:
            target_case: Target case dict with case_id, title, text, etc.
            top_k: Number of precedents to retrieve.
            internal_confidence: Optional internal confidence score.
            use_citation_graph: Whether to incorporate citation graph analysis.
        
        Returns:
            Complete analysis including external inference and legal reasoning.
        """
        print(f"\nAnalyzing case: {target_case.get('title', 'N/A')}")
        print("="*70)
        
        # Step 1: External Inference
        print("Step 1: Running external inference...")
        external_result = self.external_agent.infer(
            target=target_case,
            top_k=top_k,
            internal_confidence=internal_confidence
        )
        
        print(f"  Retrieved {len(external_result['retrieved_cases'])} cases")
        print(f"  External coherence: {external_result['overall_external_coherence_score']:.3f}")
        
        # Step 2: Citation Graph Analysis (if enabled)
        citation_analysis = None
        if use_citation_graph and self.citation_graph:
            print("Step 2: Analyzing citation graph...")
            citation_analysis = self._analyze_citation_graph(
                target_case.get('case_id'),
                [c['case_id'] for c in external_result['retrieved_cases']]
            )
        
        # Step 3: Legal Reasoning
        print("Step 3: Generating legal reasoning...")
        reasoning_result = self.reasoning_agent.generate_reasoning(
            target_case=external_result['target'],
            retrieved_cases=external_result['retrieved_cases'],
            external_coherence_score=external_result['overall_external_coherence_score']
        )
        
        # Step 4: Combine results
        complete_result = {
            "target_case": external_result['target'],
            "external_inference": {
                "retrieved_cases": external_result['retrieved_cases'],
                "external_coherence_score": external_result['overall_external_coherence_score'],
                "short_summary": external_result['short_summary']
            },
            "legal_reasoning": reasoning_result,
            "citation_analysis": citation_analysis,
            "pipeline_metadata": {
                "embedding_model": self.external_agent.model_name,
                "reasoning_method": reasoning_result['reasoning_method'],
                "top_k": top_k,
                "internal_confidence": internal_confidence
            }
        }
        
        print("Analysis complete!")
        return complete_result
    
    def _analyze_citation_graph(
        self,
        target_id: str,
        retrieved_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze citation relationships between target and retrieved cases."""
        if not self.citation_graph:
            return None
        
        analysis = {
            "target_in_graph": target_id in self.citation_graph,
            "retrieved_in_graph": [],
            "citation_links": [],
            "common_citations": [],
            "pagerank_scores": {}
        }
        
        # Check which retrieved cases are in graph
        for case_id in retrieved_ids:
            in_graph = case_id in self.citation_graph
            analysis["retrieved_in_graph"].append({
                "case_id": case_id,
                "in_graph": in_graph
            })
        
        # Find citation links
        for case_id in retrieved_ids:
            if target_id in self.citation_graph and case_id in self.citation_graph:
                # Check if target cites retrieved case
                if self.citation_graph.has_edge(target_id, case_id):
                    analysis["citation_links"].append({
                        "from": target_id,
                        "to": case_id,
                        "type": "target_cites_precedent"
                    })
                # Check if retrieved case cites target
                if self.citation_graph.has_edge(case_id, target_id):
                    analysis["citation_links"].append({
                        "from": case_id,
                        "to": target_id,
                        "type": "precedent_cites_target"
                    })
        
        # Find common citations (cases cited by both target and retrieved)
        if target_id in self.citation_graph:
            target_citations = set(self.citation_graph.successors(target_id))
            
            for case_id in retrieved_ids:
                if case_id in self.citation_graph:
                    case_citations = set(self.citation_graph.successors(case_id))
                    common = target_citations & case_citations
                    if common:
                        analysis["common_citations"].append({
                            "retrieved_case": case_id,
                            "common_cited_cases": list(common),
                            "count": len(common)
                        })
        
        # Compute PageRank scores for retrieved cases
        try:
            pagerank = nx.pagerank(self.citation_graph)
            for case_id in retrieved_ids:
                if case_id in pagerank:
                    analysis["pagerank_scores"][case_id] = pagerank[case_id]
        except:
            pass
        
        return analysis
    
    def batch_analyze(
        self,
        target_cases: List[Dict[str, Any]],
        top_k: int = 5,
        internal_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple cases in batch.
        
        Args:
            target_cases: List of target case dicts.
            top_k: Number of precedents per case.
            internal_confidence: Optional internal confidence score.
        
        Returns:
            List of complete analysis results.
        """
        results = []
        
        print(f"\nBatch analyzing {len(target_cases)} cases...")
        for i, target_case in enumerate(target_cases, 1):
            print(f"\n[{i}/{len(target_cases)}] Processing: {target_case.get('title', 'N/A')}")
            
            result = self.analyze_case(
                target_case=target_case,
                top_k=top_k,
                internal_confidence=internal_confidence
            )
            results.append(result)
        
        print(f"\nBatch analysis complete: {len(results)} cases processed")
        return results
    
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ):
        """
        Export analysis results to file.
        
        Args:
            results: Analysis results dict.
            output_path: Path to save results.
            format: Output format ('json' or 'jsonl').
        """
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format == "jsonl":
            with open(output_path, 'w') as f:
                f.write(json.dumps(results) + '\n')
        
        print(f"Results exported to: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "external_agent": self.external_agent.get_index_stats(),
            "reasoning_agent": {
                "use_llm": self.reasoning_agent.use_llm,
                "device": self.reasoning_agent.device
            }
        }
        
        if self.citation_graph:
            stats["citation_graph"] = {
                "nodes": self.citation_graph.number_of_nodes(),
                "edges": self.citation_graph.number_of_edges(),
                "density": nx.density(self.citation_graph)
            }
        
        if self.candidates_df is not None:
            stats["dataset"] = {
                "total_cases": len(self.candidates_df),
                "columns": list(self.candidates_df.columns)
            }
        
        return stats


def demo_pipeline():
    """Demonstrate the complete Legal AI Pipeline."""
    print("\n" + "="*70)
    print("LEGAL AI PIPELINE DEMO")
    print("="*70)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = LegalAIPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_llm_reasoning=False  # Use rule-based for demo
    )
    
    # Load dataset
    print("\n2. Loading LecAI dataset...")
    num_candidates = pipeline.load_lecai_dataset(
        candidates_path="lexai/data/processed/candidates_sample.jsonl"
    )
    print(f"   Loaded {num_candidates} candidates")
    
    # Create target case
    target_case = {
        "case_id": "TARGET_001",
        "title": "Modern Privacy Rights Case",
        "text": "The question before this Court is whether the right to privacy is a fundamental right "
               "guaranteed under Part III of the Constitution. Privacy is essential to human dignity and "
               "autonomy. It protects individuals from state and non-state intrusions. The right to privacy "
               "must be read as a part of Article 21 which guarantees the right to life and personal liberty.",
        "year": 2020,
        "jurisdiction": "Supreme Court of India"
    }
    
    # Run analysis
    print("\n3. Running complete analysis...")
    result = pipeline.analyze_case(
        target_case=target_case,
        top_k=3,
        internal_confidence=0.8
    )
    
    # Display results
    print("\n" + "="*70)
    print("COMPLETE ANALYSIS RESULTS")
    print("="*70)
    print(json.dumps(result, indent=2))
    
    # Show statistics
    print("\n" + "="*70)
    print("PIPELINE STATISTICS")
    print("="*70)
    stats = pipeline.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return result


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        print("Running Legal AI Pipeline tests...")
        result = demo_pipeline()
        
        # Validation
        print("\n" + "="*70)
        print("VALIDATION")
        print("="*70)
        
        required_keys = [
            'target_case', 'external_inference', 'legal_reasoning',
            'citation_analysis', 'pipeline_metadata'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
            print(f"✓ {key} present")
        
        # Check external inference
        assert 'retrieved_cases' in result['external_inference']
        assert 'external_coherence_score' in result['external_inference']
        print(f"✓ External inference complete")
        
        # Check legal reasoning
        reasoning = result['legal_reasoning']
        assert 'summary_long' in reasoning
        assert 'aspect_analysis' in reasoning
        assert 'detailed_evidence_table' in reasoning
        print(f"✓ Legal reasoning complete")
        
        # Check coherence scores
        ext_score = result['external_inference']['external_coherence_score']
        reason_score = reasoning['overall_coherence_score']
        assert 0 <= ext_score <= 1 and 0 <= reason_score <= 1
        print(f"✓ Coherence scores valid: ext={ext_score:.3f}, reason={reason_score:.3f}")
        
        print("\n" + "="*70)
        print("ALL VALIDATIONS PASSED ✓")
        print("="*70)
    else:
        demo_pipeline()
