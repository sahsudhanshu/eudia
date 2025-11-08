"""
InLegalBERT-based External Inference Agent for Legal AI.

This module implements an ExternalInferenceAgent specifically designed for the LecAI dataset
using InLegalBERT embeddings and citation graph reasoning. It combines:
1. Citation graph analysis (NetworkX)
2. Semantic similarity (InLegalBERT embeddings)
3. Hybrid retrieval (graph + semantic)
4. Legal reasoning generation

Compatible with LecAI baseline dataset format.
"""

import os
import json
import gzip
import csv
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F
except ImportError:
    print("Warning: transformers or torch not installed. Install with: pip install transformers torch")
    raise


class InLegalBERTExternalAgent:
    """
    External Inference Agent using InLegalBERT and citation graph reasoning.
    
    This agent performs hybrid retrieval combining:
    - Citation graph traversal (cited/citing cases)
    - Semantic similarity using InLegalBERT embeddings
    - Citeomatic-style scoring for case ranking
    """
    
    def __init__(
        self,
        model_name: str = "law-ai/InLegalBERT",
        device: Optional[str] = None,
        max_length: int = 512,
        use_citation_weight: float = 0.4,
        use_semantic_weight: float = 0.6
    ):
        """
        Initialize the InLegalBERT External Agent.
        
        Args:
            model_name: HuggingFace model name (default: InLegalBERT)
            device: Device to run on ("cuda", "cpu", or None for auto-detect)
            max_length: Maximum token length for BERT encoding
            use_citation_weight: Weight for citation-based similarity [0, 1]
            use_semantic_weight: Weight for semantic similarity [0, 1]
        """
        self.model_name = model_name
        self.max_length = max_length
        self.citation_weight = use_citation_weight
        self.semantic_weight = use_semantic_weight
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Initializing InLegalBERT External Agent on device: {self.device}")
        
        # Load InLegalBERT model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Data storage
        self.cases = {}  # case_id -> case_data dict
        self.case_embeddings = {}  # case_id -> embedding vector
        self.citation_graph = nx.DiGraph()  # Directed graph: source -> target (cites)
        self.case_id_to_idx = {}  # case_id -> node index
        self.idx_to_case_id = {}  # node index -> case_id
        
        # Metadata
        self.dataset_loaded = False
        self.embeddings_computed = False
        
        print("InLegalBERT External Agent initialized successfully")
    
    def load_dataset(
        self,
        case_mapping_path: str,
        case_text_dir: Optional[str] = None,
        edge_list_path: Optional[str] = None,
        node_features_path: Optional[str] = None,
        max_cases: Optional[int] = None
    ) -> int:
        """
        Load LecAI dataset with case information and citations.
        
        Args:
            case_mapping_path: Path to CSV mapping file (file_name, Case_id, Case_name)
            case_text_dir: Optional directory containing case text files
            edge_list_path: Optional path to edge list file (source, target)
            node_features_path: Optional path to node features (embeddings or metadata)
            max_cases: Optional limit on number of cases to load
        
        Returns:
            Number of cases loaded
        """
        print(f"\nLoading LecAI dataset from: {case_mapping_path}")
        
        # Load case mapping
        df = pd.read_csv(case_mapping_path)
        if max_cases:
            df = df.head(max_cases)
        
        print(f"Found {len(df)} cases in mapping file")
        
        # Load case data
        for idx, row in df.iterrows():
            case_id = str(row['Case_id']).strip()
            case_name = str(row['Case_name']).strip()
            file_name = str(row['file_name']).strip()
            
            # Store case metadata
            self.cases[case_id] = {
                'case_id': case_id,
                'case_name': case_name,
                'file_name': file_name,
                'text': None,  # Will be loaded if text_dir provided
                'year': self._extract_year_from_id(case_id),
                'node_idx': idx
            }
            
            self.case_id_to_idx[case_id] = idx
            self.idx_to_case_id[idx] = case_id
        
        print(f"Loaded {len(self.cases)} cases")
        
        # Load case texts if directory provided
        if case_text_dir and os.path.exists(case_text_dir):
            print(f"Loading case texts from: {case_text_dir}")
            loaded_texts = 0
            for case_id, case_data in self.cases.items():
                file_name = case_data['file_name']
                text_path = os.path.join(case_text_dir, file_name)
                
                if os.path.exists(text_path):
                    try:
                        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                            case_data['text'] = f.read()
                        loaded_texts += 1
                    except Exception as e:
                        print(f"Warning: Failed to load {file_name}: {e}")
            
            print(f"Loaded text for {loaded_texts}/{len(self.cases)} cases")
        
        # Load citation graph if edge list provided
        if edge_list_path and os.path.exists(edge_list_path):
            self._load_edge_list(edge_list_path)
        
        # Load node features if provided
        if node_features_path and os.path.exists(node_features_path):
            self._load_node_features(node_features_path)
        
        self.dataset_loaded = True
        print(f"Dataset loading complete: {len(self.cases)} cases, {self.citation_graph.number_of_edges()} citations")
        
        return len(self.cases)
    
    def _extract_year_from_id(self, case_id: str) -> Optional[int]:
        """Extract year from case ID (format: YYYYMMKSXSC)."""
        try:
            if len(case_id) >= 4:
                year = int(case_id[:4])
                if 1900 <= year <= 2030:
                    return year
        except:
            pass
        return None
    
    def _load_edge_list(self, edge_list_path: str):
        """Load citation graph from edge list file."""
        print(f"Loading citation graph from: {edge_list_path}")
        
        # Check if gzipped
        is_gzipped = edge_list_path.endswith('.gz')
        
        edges_loaded = 0
        
        try:
            if is_gzipped:
                with gzip.open(edge_list_path, 'rt') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # Skip header if present
                    
                    for row in reader:
                        if len(row) >= 2:
                            source_idx, target_idx = int(row[0]), int(row[1])
                            
                            # Convert node indices to case IDs
                            if source_idx in self.idx_to_case_id and target_idx in self.idx_to_case_id:
                                source_id = self.idx_to_case_id[source_idx]
                                target_id = self.idx_to_case_id[target_idx]
                                self.citation_graph.add_edge(source_id, target_id)
                                edges_loaded += 1
            else:
                with open(edge_list_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    
                    for row in reader:
                        if len(row) >= 2:
                            source_idx, target_idx = int(row[0]), int(row[1])
                            
                            if source_idx in self.idx_to_case_id and target_idx in self.idx_to_case_id:
                                source_id = self.idx_to_case_id[source_idx]
                                target_id = self.idx_to_case_id[target_idx]
                                self.citation_graph.add_edge(source_id, target_id)
                                edges_loaded += 1
            
            print(f"Citation graph loaded: {edges_loaded} edges")
        
        except Exception as e:
            print(f"Warning: Failed to load edge list: {e}")
    
    def _load_node_features(self, node_features_path: str):
        """Load pre-computed node features (optional)."""
        print(f"Loading node features from: {node_features_path}")
        # Implementation depends on feature format
        # For now, skip and compute embeddings on-the-fly
        pass
    
    def build_graph(self, cases_data: Optional[List[Dict[str, Any]]] = None):
        """
        Build citation graph from loaded data or provided cases.
        
        Args:
            cases_data: Optional list of case dicts with 'case_id' and 'citations' fields
        """
        if cases_data:
            print(f"Building graph from {len(cases_data)} provided cases")
            
            for case in cases_data:
                case_id = case['case_id']
                
                # Add node
                if case_id not in self.cases:
                    self.cases[case_id] = case
                
                # Add edges
                if 'citations' in case and case['citations']:
                    for cited_case_id in case['citations']:
                        if cited_case_id in self.cases:
                            self.citation_graph.add_edge(case_id, cited_case_id)
        
        print(f"Citation graph: {self.citation_graph.number_of_nodes()} nodes, "
              f"{self.citation_graph.number_of_edges()} edges")
    
    def get_case_embedding(
        self,
        case_id: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Get InLegalBERT embedding for a case.
        
        Args:
            case_id: Case identifier
            use_cache: Whether to use cached embedding
        
        Returns:
            Embedding vector (768-dimensional for BERT)
        """
        # Check cache
        if use_cache and case_id in self.case_embeddings:
            return self.case_embeddings[case_id]
        
        # Get case text
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found in dataset")
        
        case_data = self.cases[case_id]
        text = case_data.get('text') or case_data.get('case_name', '')
        
        if not text:
            print(f"Warning: No text for case {case_id}, using case name")
            text = case_data.get('case_name', 'Unknown case')
        
        # Compute embedding
        embedding = self._encode_text(text)
        
        # Cache
        self.case_embeddings[case_id] = embedding
        
        return embedding
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using InLegalBERT."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        return embedding.flatten()
    
    def compute_all_embeddings(self, batch_size: int = 8):
        """
        Pre-compute embeddings for all cases.
        
        Args:
            batch_size: Batch size for encoding
        """
        print(f"\nComputing InLegalBERT embeddings for {len(self.cases)} cases...")
        
        case_ids = list(self.cases.keys())
        total = len(case_ids)
        
        for i in range(0, total, batch_size):
            batch_ids = case_ids[i:i+batch_size]
            
            for case_id in batch_ids:
                self.get_case_embedding(case_id, use_cache=True)
            
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= total:
                print(f"  Processed {min(i+batch_size, total)}/{total} cases")
        
        self.embeddings_computed = True
        print("All embeddings computed successfully")
    
    def retrieve_similar_cases(
        self,
        query_case_id: str,
        top_k: int = 10,
        use_citations: bool = True,
        use_semantic: bool = True,
        min_citation_hops: int = 1,
        max_citation_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar cases using hybrid graph + semantic retrieval.
        
        Args:
            query_case_id: Query case identifier
            top_k: Number of top cases to retrieve
            use_citations: Whether to use citation graph
            use_semantic: Whether to use semantic similarity
            min_citation_hops: Minimum citation distance
            max_citation_hops: Maximum citation distance for graph traversal
        
        Returns:
            List of retrieved cases with scores and reasoning
        """
        if query_case_id not in self.cases:
            raise ValueError(f"Query case {query_case_id} not found")
        
        print(f"\nRetrieving similar cases for: {query_case_id}")
        print(f"  Query: {self.cases[query_case_id]['case_name']}")
        
        # Get query embedding
        query_embedding = self.get_case_embedding(query_case_id)
        
        # Candidate set
        candidates = set()
        citation_scores = {}
        semantic_scores = {}
        
        # 1. Citation-based retrieval
        if use_citations and self.citation_graph.number_of_edges() > 0:
            citation_candidates = self._get_citation_neighbors(
                query_case_id,
                max_hops=max_citation_hops
            )
            candidates.update(citation_candidates.keys())
            citation_scores = citation_candidates
            print(f"  Found {len(citation_candidates)} citation-based candidates")
        
        # 2. Semantic retrieval (if no citations or use_semantic=True)
        if use_semantic or len(candidates) == 0:
            # Compute similarity with all cases
            for case_id in self.cases.keys():
                if case_id == query_case_id:
                    continue
                
                case_embedding = self.get_case_embedding(case_id)
                similarity = float(np.dot(query_embedding, case_embedding))
                semantic_scores[case_id] = similarity
                candidates.add(case_id)
            
            print(f"  Computed semantic similarity for {len(semantic_scores)} cases")
        
        # 3. Hybrid scoring (Citeomatic-style)
        hybrid_scores = {}
        for case_id in candidates:
            citation_score = citation_scores.get(case_id, 0.0)
            semantic_score = semantic_scores.get(case_id, 0.0)
            
            # Hybrid score
            if use_citations and use_semantic:
                hybrid_score = (
                    self.citation_weight * citation_score +
                    self.semantic_weight * semantic_score
                )
            elif use_citations:
                hybrid_score = citation_score
            else:
                hybrid_score = semantic_score
            
            hybrid_scores[case_id] = hybrid_score
        
        # 4. Rank and select top-k
        ranked_cases = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 5. Build result with reasoning
        results = []
        for rank, (case_id, score) in enumerate(ranked_cases, 1):
            case_data = self.cases[case_id]
            
            result = {
                'rank': rank,
                'case_id': case_id,
                'case_name': case_data['case_name'],
                'year': case_data.get('year'),
                'similarity_score': float(score),
                'semantic_score': semantic_scores.get(case_id, 0.0),
                'citation_score': citation_scores.get(case_id, 0.0),
                'reasoning': self._generate_case_reasoning(
                    query_case_id,
                    case_id,
                    semantic_scores.get(case_id, 0.0),
                    citation_scores.get(case_id, 0.0)
                )
            }
            results.append(result)
        
        print(f"  Retrieved {len(results)} similar cases")
        return results
    
    def _get_citation_neighbors(
        self,
        case_id: str,
        max_hops: int = 2
    ) -> Dict[str, float]:
        """
        Get citation neighbors with distance-based scoring.
        
        Args:
            case_id: Source case
            max_hops: Maximum citation distance
        
        Returns:
            Dict mapping case_id to citation score
        """
        if case_id not in self.citation_graph:
            return {}
        
        neighbors = {}
        
        # Forward citations (cases cited by query)
        if case_id in self.citation_graph:
            for target in self.citation_graph.successors(case_id):
                neighbors[target] = 1.0  # Direct citation
        
        # Backward citations (cases citing query)
        for source in self.citation_graph.predecessors(case_id):
            neighbors[source] = 0.9  # Being cited
        
        # 2-hop neighbors (if max_hops >= 2)
        if max_hops >= 2:
            second_hop = set()
            
            # Forward 2-hop
            for neighbor in list(neighbors.keys()):
                if neighbor in self.citation_graph:
                    for target in self.citation_graph.successors(neighbor):
                        if target != case_id and target not in neighbors:
                            second_hop.add(target)
            
            # Add with lower score
            for target in second_hop:
                neighbors[target] = 0.5
        
        return neighbors
    
    def _generate_case_reasoning(
        self,
        query_id: str,
        candidate_id: str,
        semantic_score: float,
        citation_score: float
    ) -> str:
        """Generate human-readable reasoning for case similarity."""
        query_case = self.cases[query_id]
        candidate_case = self.cases[candidate_id]
        
        reasoning_parts = []
        
        # Semantic reasoning
        if semantic_score > 0.8:
            reasoning_parts.append(
                f"High semantic similarity ({semantic_score:.2f}) indicates strong conceptual overlap in legal reasoning and arguments."
            )
        elif semantic_score > 0.6:
            reasoning_parts.append(
                f"Moderate semantic similarity ({semantic_score:.2f}) suggests related legal concepts and principles."
            )
        elif semantic_score > 0.4:
            reasoning_parts.append(
                f"Some semantic similarity ({semantic_score:.2f}) indicates partial relevance in legal domain."
            )
        
        # Citation reasoning
        if citation_score >= 1.0:
            reasoning_parts.append(
                "This case is directly cited in the query case, establishing precedential relationship."
            )
        elif citation_score >= 0.9:
            reasoning_parts.append(
                "This case cites the query case, showing awareness and engagement with its reasoning."
            )
        elif citation_score >= 0.5:
            reasoning_parts.append(
                "This case is connected through citation network (2-hop), indicating indirect precedential influence."
            )
        
        # Temporal reasoning
        query_year = query_case.get('year')
        candidate_year = candidate_case.get('year')
        
        if query_year and candidate_year:
            year_diff = query_year - candidate_year
            if year_diff > 0:
                reasoning_parts.append(
                    f"This case ({candidate_year}) predates the query case ({query_year}) by {year_diff} years, "
                    "potentially serving as historical precedent."
                )
            elif year_diff < 0:
                reasoning_parts.append(
                    f"This case ({candidate_year}) is {abs(year_diff)} years more recent than the query case ({query_year})."
                )
        
        if not reasoning_parts:
            reasoning_parts.append("Related case based on hybrid retrieval scoring.")
        
        return " ".join(reasoning_parts)
    
    def generate_reasoning_output(
        self,
        query_case_id: str,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate structured output with legal reasoning summary.
        
        Args:
            query_case_id: Query case identifier
            retrieved_cases: List of retrieved cases from retrieve_similar_cases()
        
        Returns:
            Structured output dict with reasoning summary
        """
        query_case = self.cases[query_case_id]
        
        # Generate overall summary
        if len(retrieved_cases) == 0:
            summary = "No similar cases found in the dataset."
        else:
            # Analyze retrieved cases
            high_sim_count = sum(1 for c in retrieved_cases if c['similarity_score'] > 0.7)
            citation_count = sum(1 for c in retrieved_cases if c['citation_score'] > 0.5)
            
            avg_year = np.mean([
                c['year'] for c in retrieved_cases if c['year'] is not None
            ]) if any(c['year'] for c in retrieved_cases) else None
            
            summary = f"The query case '{query_case['case_name']}' shows strong legal connections to {len(retrieved_cases)} precedent cases. "
            
            if high_sim_count > 0:
                summary += f"{high_sim_count} case(s) exhibit high semantic similarity (>0.7), indicating shared legal principles and reasoning patterns. "
            
            if citation_count > 0:
                summary += f"{citation_count} case(s) are connected through the citation network, establishing direct precedential relationships. "
            
            if avg_year:
                summary += f"The retrieved cases span an average year of {int(avg_year)}, providing temporal context for legal evolution. "
            
            summary += "These cases collectively form a robust external knowledge base for legal analysis and argumentation."
        
        # Build output structure
        output = {
            'query_case': query_case_id,
            'query_case_name': query_case['case_name'],
            'query_year': query_case.get('year'),
            'top_references': [
                {
                    'case_id': case['case_id'],
                    'case_name': case['case_name'],
                    'year': case['year'],
                    'similarity_score': round(case['similarity_score'], 3),
                    'semantic_score': round(case['semantic_score'], 3),
                    'citation_score': round(case['citation_score'], 3),
                    'reasoning': case['reasoning']
                }
                for case in retrieved_cases
            ],
            'overall_legal_context_summary': summary,
            'retrieval_metadata': {
                'total_retrieved': len(retrieved_cases),
                'citation_weight': self.citation_weight,
                'semantic_weight': self.semantic_weight,
                'model': self.model_name
            }
        }
        
        return output
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            'total_cases': len(self.cases),
            'embeddings_computed': len(self.case_embeddings),
            'citation_graph': {
                'nodes': self.citation_graph.number_of_nodes(),
                'edges': self.citation_graph.number_of_edges(),
                'density': nx.density(self.citation_graph) if self.citation_graph.number_of_nodes() > 0 else 0
            },
            'model': self.model_name,
            'device': self.device,
            'weights': {
                'citation': self.citation_weight,
                'semantic': self.semantic_weight
            }
        }
        
        # Add year distribution
        years = [c.get('year') for c in self.cases.values() if c.get('year')]
        if years:
            stats['year_range'] = {
                'min': min(years),
                'max': max(years),
                'mean': int(np.mean(years))
            }
        
        return stats


def demo_inlegalbert_agent():
    """Demonstrate InLegalBERT External Agent with LecAI dataset."""
    print("\n" + "="*70)
    print("INLEGALBERT EXTERNAL AGENT DEMO")
    print("="*70)
    
    # Initialize agent
    print("\n1. Initializing InLegalBERT External Agent...")
    agent = InLegalBERTExternalAgent(
        model_name="law-ai/InLegalBERT",
        use_citation_weight=0.4,
        use_semantic_weight=0.6,
        device=None  # Auto-detect
    )
    
    # Load LecAI dataset
    print("\n2. Loading LecAI dataset...")
    
    # Path to LecAI dataset
    base_path = "/home/anand/eudia/LecNet_baseline_models_code_and_README/LecNet_baseline_models_code_and_README"
    case_mapping_path = os.path.join(base_path, "file_Case_id_and_name.csv")
    
    # Load first 100 cases for demo
    num_cases = agent.load_dataset(
        case_mapping_path=case_mapping_path,
        max_cases=100
    )
    
    print(f"\n3. Computing embeddings for {num_cases} cases...")
    agent.compute_all_embeddings(batch_size=4)
    
    # Select a sample query case
    print("\n4. Running retrieval on sample query case...")
    sample_case_ids = list(agent.cases.keys())[:10]
    query_case_id = sample_case_ids[0]
    
    print(f"   Query Case: {query_case_id}")
    print(f"   Query Name: {agent.cases[query_case_id]['case_name']}")
    
    # Retrieve similar cases
    retrieved = agent.retrieve_similar_cases(
        query_case_id=query_case_id,
        top_k=5,
        use_citations=False,  # No edge list loaded in demo
        use_semantic=True
    )
    
    # Generate reasoning output
    print("\n5. Generating structured reasoning output...")
    output = agent.generate_reasoning_output(
        query_case_id=query_case_id,
        retrieved_cases=retrieved
    )
    
    # Display results
    print("\n" + "="*70)
    print("STRUCTURED OUTPUT")
    print("="*70)
    print(json.dumps(output, indent=2))
    
    # Show statistics
    print("\n" + "="*70)
    print("AGENT STATISTICS")
    print("="*70)
    stats = agent.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return output, agent


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        print("Running InLegalBERT External Agent demo...")
        output, agent = demo_inlegalbert_agent()
        
        # Validation
        print("\n" + "="*70)
        print("VALIDATION")
        print("="*70)
        
        required_keys = ['query_case', 'top_references', 'overall_legal_context_summary']
        for key in required_keys:
            assert key in output, f"Missing key: {key}"
            print(f"✓ {key} present")
        
        assert len(output['top_references']) > 0, "No references retrieved"
        print(f"✓ Retrieved {len(output['top_references'])} references")
        
        for ref in output['top_references']:
            assert 'similarity_score' in ref, "Missing similarity_score"
            assert 'reasoning' in ref, "Missing reasoning"
            assert 0 <= ref['similarity_score'] <= 1, "Invalid similarity score"
        
        print("✓ All references have valid scores and reasoning")
        
        print("\n" + "="*70)
        print("ALL VALIDATIONS PASSED ✓")
        print("="*70)
    else:
        demo_inlegalbert_agent()
