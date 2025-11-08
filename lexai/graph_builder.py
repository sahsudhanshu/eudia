"""
Legal Citation Graph Builder using FAISS and Transformer Classification.

This module builds citation graphs from legal documents by:
1. Loading documents and computing embeddings
2. Using FAISS to find nearest neighbor documents
3. Classifying citation relationships using a transformer
4. Building a NetworkX citation graph
5. Saving and visualizing the graph
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("Install with: pip install faiss-cpu sentence-transformers transformers torch networkx matplotlib")
    raise


class GraphBuilder:
    """
    Build legal citation graphs using FAISS similarity and transformer classification.
    
    Features:
    - Document embedding with Sentence Transformers
    - FAISS-based efficient similarity search
    - Citation type classification (supports/contradicts/refers/neutral)
    - NetworkX graph construction
    - Visualization and statistics
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        classifier_model: str = "cross-encoder/nli-deberta-v3-small",
        device: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Initialize the Graph Builder.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            classifier_model: Model for citation type classification
            device: Device to use ('cuda', 'cpu', or None for auto)
            top_k: Number of nearest neighbors to consider per document
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Initializing GraphBuilder on device: {self.device}")
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_model.to(self.device)
        
        # Load classifier (using NLI model as proxy for citation classification)
        print(f"Loading classifier: {classifier_model}")
        self.classifier = pipeline(
            "text-classification",
            model=classifier_model,
            device=0 if self.device == "cuda" else -1
        )
        
        # Parameters
        self.top_k = top_k
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Storage
        self.documents = []  # List of document dicts
        self.embeddings = None  # numpy array of embeddings
        self.faiss_index = None  # FAISS index
        self.graph = nx.DiGraph()  # Citation graph
        
        print(f"GraphBuilder initialized (embedding_dim={self.embedding_dim}, top_k={top_k})")
    
    def load_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'text',
        id_field: str = 'doc_id'
    ) -> int:
        """
        Load legal documents.
        
        Args:
            documents: List of document dicts with text and metadata
            text_field: Field name containing document text
            id_field: Field name containing document ID
        
        Returns:
            Number of documents loaded
        """
        print(f"\nLoading {len(documents)} documents...")
        
        self.documents = []
        for i, doc in enumerate(documents):
            if text_field not in doc:
                print(f"Warning: Document {i} missing '{text_field}' field, skipping")
                continue
            
            # Ensure document has an ID
            if id_field not in doc:
                doc[id_field] = f"DOC_{i:04d}"
            
            self.documents.append(doc)
        
        print(f"Loaded {len(self.documents)} documents")
        return len(self.documents)
    
    def compute_embeddings(
        self,
        text_field: str = 'text',
        batch_size: int = 16,
        show_progress: bool = True
    ):
        """
        Compute embeddings for all documents.
        
        Args:
            text_field: Field containing document text
            batch_size: Batch size for embedding computation
            show_progress: Whether to show progress
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        print(f"\nComputing embeddings for {len(self.documents)} documents...")
        
        # Extract texts
        texts = [doc[text_field] for doc in self.documents]
        
        # Compute embeddings
        self.embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"Embeddings computed: shape={self.embeddings.shape}")
    
    def build_faiss_index(self, use_gpu: bool = False):
        """
        Build FAISS index from embeddings.
        
        Args:
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call compute_embeddings() first.")
        
        print(f"\nBuilding FAISS index...")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        
        # Create FAISS index (inner product = cosine similarity for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings_normalized.astype('float32'))
        
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def find_nearest_neighbors(
        self,
        doc_idx: int,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a document.
        
        Args:
            doc_idx: Document index
            k: Number of neighbors (uses self.top_k if None)
        
        Returns:
            Tuple of (distances, indices)
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")
        
        k = k or self.top_k
        
        # Get embedding
        query_emb = self.embeddings[doc_idx:doc_idx+1].astype('float32')
        
        # Normalize
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Search
        distances, indices = self.faiss_index.search(query_emb, k + 1)  # +1 to exclude self
        
        # Remove self from results
        mask = indices[0] != doc_idx
        distances = distances[0][mask][:k]
        indices = indices[0][mask][:k]
        
        return distances, indices
    
    def classify_citation_type(
        self,
        source_text: str,
        target_text: str,
        max_length: int = 256
    ) -> str:
        """
        Classify the citation relationship between two documents.
        
        Uses NLI model as a proxy:
        - entailment -> supports
        - contradiction -> contradicts
        - neutral -> refers (or neutral)
        
        Args:
            source_text: Source document text
            target_text: Target document text
            max_length: Maximum text length (truncate if longer)
        
        Returns:
            Citation type: 'supports', 'contradicts', 'refers', or 'neutral'
        """
        # Truncate texts
        source_truncated = source_text[:max_length]
        target_truncated = target_text[:max_length]
        
        # Create pair for NLI
        text_pair = f"{source_truncated} [SEP] {target_truncated}"
        
        try:
            # Classify
            result = self.classifier(text_pair, truncation=True, max_length=512)
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            # Map NLI labels to citation types
            if 'entail' in label:
                return 'supports' if score > 0.6 else 'refers'
            elif 'contra' in label:
                return 'contradicts' if score > 0.6 else 'neutral'
            else:
                return 'neutral'
        
        except Exception as e:
            print(f"Warning: Classification failed - {e}")
            return 'neutral'
    
    def build_citation_graph(
        self,
        text_field: str = 'text',
        id_field: str = 'doc_id',
        similarity_threshold: float = 0.3,
        max_edges_per_node: Optional[int] = None
    ):
        """
        Build citation graph using FAISS similarity and classification.
        
        Args:
            text_field: Field containing document text
            id_field: Field containing document ID
            similarity_threshold: Minimum similarity to consider
            max_edges_per_node: Maximum edges per node (None = use top_k)
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")
        
        print(f"\nBuilding citation graph...")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Similarity threshold: {similarity_threshold}")
        
        max_edges = max_edges_per_node or self.top_k
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for doc in self.documents:
            doc_id = doc[id_field]
            self.graph.add_node(
                doc_id,
                text=doc.get(text_field, '')[:200],  # Store snippet
                **{k: v for k, v in doc.items() if k not in [text_field]}
            )
        
        # Build edges
        edges_added = 0
        
        for idx, doc in enumerate(self.documents):
            if idx % 10 == 0:
                print(f"  Processing document {idx}/{len(self.documents)}...")
            
            source_id = doc[id_field]
            source_text = doc[text_field]
            
            # Find nearest neighbors
            distances, neighbor_indices = self.find_nearest_neighbors(idx, k=max_edges)
            
            for dist, neighbor_idx in zip(distances, neighbor_indices):
                # Check similarity threshold
                if dist < similarity_threshold:
                    continue
                
                neighbor_doc = self.documents[neighbor_idx]
                target_id = neighbor_doc[id_field]
                target_text = neighbor_doc[text_field]
                
                # Classify citation type
                citation_type = self.classify_citation_type(source_text, target_text)
                
                # Add edge
                self.graph.add_edge(
                    source_id,
                    target_id,
                    citation_type=citation_type,
                    similarity=float(dist),
                    weight=float(dist)
                )
                
                edges_added += 1
        
        print(f"\nGraph built:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Average degree: {edges_added / len(self.documents):.2f}")
    
    def save_graph(
        self,
        output_path: str,
        format: str = 'json'
    ):
        """
        Save citation graph to file.
        
        Args:
            output_path: Path to save file
            format: Output format ('json', 'gexf', 'graphml')
        """
        print(f"\nSaving graph to: {output_path}")
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        if format == 'json':
            # Convert to JSON-serializable format
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        **self.graph.nodes[node]
                    }
                    for node in self.graph.nodes()
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        **self.graph.edges[u, v]
                    }
                    for u, v in self.graph.edges()
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        
        elif format == 'gexf':
            nx.write_gexf(self.graph, output_path)
        
        elif format == 'graphml':
            nx.write_graphml(self.graph, output_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Graph saved successfully")
    
    def visualize_degree_distribution(
        self,
        output_path: Optional[str] = None,
        title: str = "Citation Graph Degree Distribution"
    ):
        """
        Visualize degree distribution of the citation graph.
        
        Args:
            output_path: Path to save figure (None = show only)
            title: Plot title
        """
        if self.graph.number_of_nodes() == 0:
            print("Warning: Graph is empty, cannot visualize")
            return
        
        print("\nGenerating degree distribution visualization...")
        
        # Compute degrees
        in_degrees = [self.graph.in_degree(node) for node in self.graph.nodes()]
        out_degrees = [self.graph.out_degree(node) for node in self.graph.nodes()]
        total_degrees = [in_deg + out_deg for in_deg, out_deg in zip(in_degrees, out_degrees)]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # In-degree distribution
        axes[0].hist(in_degrees, bins=20, edgecolor='black', alpha=0.7, color='blue')
        axes[0].set_xlabel('In-Degree')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('In-Degree Distribution')
        axes[0].grid(alpha=0.3)
        
        # Out-degree distribution
        axes[1].hist(out_degrees, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_xlabel('Out-Degree')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Out-Degree Distribution')
        axes[1].grid(alpha=0.3)
        
        # Total degree distribution
        axes[2].hist(total_degrees, bins=20, edgecolor='black', alpha=0.7, color='purple')
        axes[2].set_xlabel('Total Degree')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Total Degree Distribution')
        axes[2].grid(alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if self.graph.number_of_nodes() == 0:
            return {'error': 'Graph is empty'}
        
        # Citation type distribution
        citation_types = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            citation_type = data.get('citation_type', 'unknown')
            citation_types[citation_type] += 1
        
        # Degree statistics
        in_degrees = [self.graph.in_degree(node) for node in self.graph.nodes()]
        out_degrees = [self.graph.out_degree(node) for node in self.graph.nodes()]
        
        stats = {
            'num_nodes': int(self.graph.number_of_nodes()),
            'num_edges': int(self.graph.number_of_edges()),
            'density': float(nx.density(self.graph)),
            'citation_types': dict(citation_types),
            'degree_stats': {
                'in_degree': {
                    'mean': float(np.mean(in_degrees)),
                    'median': float(np.median(in_degrees)),
                    'max': int(np.max(in_degrees)),
                    'min': int(np.min(in_degrees))
                },
                'out_degree': {
                    'mean': float(np.mean(out_degrees)),
                    'median': float(np.median(out_degrees)),
                    'max': int(np.max(out_degrees)),
                    'min': int(np.min(out_degrees))
                }
            }
        }
        
        # Try to compute additional metrics (may fail for disconnected graphs)
        try:
            stats['avg_clustering'] = float(nx.average_clustering(self.graph.to_undirected()))
        except:
            pass
        
        try:
            largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
            stats['largest_component_size'] = int(len(largest_cc))
        except:
            pass
        
        return stats


# ===========================================
# Demo Function
# ===========================================
def demo_graph_builder():
    """
    Demo: Build citation graph from 10 LecAI documents.
    """
    print("="*70)
    print("GRAPH BUILDER DEMO - LecAI Dataset")
    print("="*70)
    
    # Load sample documents from LecAI
    print("\n1. Loading LecAI sample documents...")
    
    base_path = "/home/anand/eudia/LecNet_baseline_models_code_and_README/LecNet_baseline_models_code_and_README"
    case_mapping_path = f"{base_path}/file_Case_id_and_name.csv"
    
    # Load case metadata
    df = pd.read_csv(case_mapping_path).head(10)  # First 10 cases for demo
    
    documents = []
    for idx, row in df.iterrows():
        doc = {
            'doc_id': str(row['Case_id']).strip(),
            'case_name': str(row['Case_name']).strip(),
            'text': str(row['Case_name']).strip(),  # Using case name as text for demo
            'file_name': str(row['file_name']).strip()
        }
        documents.append(doc)
    
    print(f"Loaded {len(documents)} documents")
    
    # Initialize GraphBuilder
    print("\n2. Initializing GraphBuilder...")
    builder = GraphBuilder(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        classifier_model="cross-encoder/nli-deberta-v3-small",
        top_k=3
    )
    
    # Load documents
    print("\n3. Loading documents into builder...")
    builder.load_documents(documents, text_field='text', id_field='doc_id')
    
    # Compute embeddings
    print("\n4. Computing embeddings...")
    builder.compute_embeddings(text_field='text', batch_size=8)
    
    # Build FAISS index
    print("\n5. Building FAISS index...")
    builder.build_faiss_index(use_gpu=False)
    
    # Build citation graph
    print("\n6. Building citation graph...")
    builder.build_citation_graph(
        text_field='text',
        id_field='doc_id',
        similarity_threshold=0.2,
        max_edges_per_node=3
    )
    
    # Get statistics
    print("\n7. Graph Statistics:")
    stats = builder.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Save graph
    print("\n8. Saving graph...")
    os.makedirs("lexai/data/graphs", exist_ok=True)
    builder.save_graph("lexai/data/graphs/lecai_citation_graph.json", format='json')
    
    # Visualize degree distribution
    print("\n9. Visualizing degree distribution...")
    builder.visualize_degree_distribution(
        output_path="lexai/data/graphs/degree_distribution.png",
        title="LecAI Citation Graph - Degree Distribution"
    )
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - Graph JSON: lexai/data/graphs/lecai_citation_graph.json")
    print("  - Degree distribution: lexai/data/graphs/degree_distribution.png")
    
    return builder


if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        print("Running GraphBuilder demo...")
        builder = demo_graph_builder()
        
        print("\n" + "="*70)
        print("VALIDATION")
        print("="*70)
        
        assert builder.graph.number_of_nodes() > 0, "Graph has no nodes"
        assert builder.graph.number_of_edges() > 0, "Graph has no edges"
        assert builder.embeddings is not None, "No embeddings computed"
        assert builder.faiss_index is not None, "FAISS index not built"
        
        print("✓ Graph has nodes and edges")
        print("✓ Embeddings computed")
        print("✓ FAISS index built")
        print("\nALL VALIDATIONS PASSED ✓")
    else:
        demo_graph_builder()
