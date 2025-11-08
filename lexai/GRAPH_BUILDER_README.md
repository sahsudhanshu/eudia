# Legal Citation Graph Builder

Build citation graphs from legal documents using **FAISS** similarity search and **transformer-based** citation classification.

## 🎯 Features

✅ **FAISS-based retrieval** - Efficient nearest neighbor search  
✅ **Transformer classification** - Predict citation types (supports/contradicts/refers/neutral)  
✅ **NetworkX graphs** - Flexible graph representation  
✅ **Multiple export formats** - JSON, GEXF, GraphML  
✅ **Visualization** - Degree distribution plots  
✅ **LecAI compatible** - Works with LecAI baseline dataset  

## 🚀 Quick Start

```python
from lexai.graph_builder import GraphBuilder

# Initialize
builder = GraphBuilder(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    classifier_model="cross-encoder/nli-deberta-v3-small",
    top_k=5
)

# Load documents
documents = [
    {'doc_id': 'CASE_001', 'text': 'Legal case text...'},
    {'doc_id': 'CASE_002', 'text': 'Another case...'},
    # ... more documents
]
builder.load_documents(documents)

# Build graph
builder.compute_embeddings()
builder.build_faiss_index()
builder.build_citation_graph()

# Save and visualize
builder.save_graph("citation_graph.json")
builder.visualize_degree_distribution("degree_dist.png")
```

## 📦 Installation

```bash
pip install faiss-cpu sentence-transformers transformers torch networkx matplotlib pandas numpy
```

## 🧪 Run Demo

```bash
# Run with 10 LecAI documents
python lexai/graph_builder.py --test
```

**Outputs:**
- `lexai/data/graphs/lecai_citation_graph.json` - Citation graph
- `lexai/data/graphs/degree_distribution.png` - Degree distribution plot

## 📊 How It Works

### 1. Document Embedding
Uses Sentence Transformers to create semantic embeddings of legal documents.

```python
builder.compute_embeddings(text_field='text', batch_size=16)
```

### 2. FAISS Similarity Search
Builds FAISS index for efficient nearest neighbor retrieval.

```python
builder.build_faiss_index()
distances, indices = builder.find_nearest_neighbors(doc_idx=0, k=5)
```

### 3. Citation Type Classification
Uses NLI transformer to classify citation relationships:

| NLI Label | Citation Type |
|-----------|---------------|
| Entailment (>0.6) | **supports** |
| Contradiction (>0.6) | **contradicts** |
| Neutral or low score | **refers** or **neutral** |

```python
citation_type = builder.classify_citation_type(source_text, target_text)
# Returns: 'supports', 'contradicts', 'refers', or 'neutral'
```

### 4. Graph Construction
Builds NetworkX DiGraph with:
- **Nodes**: Documents with metadata
- **Edges**: Citations with type and similarity score

```python
builder.build_citation_graph(
    similarity_threshold=0.3,
    max_edges_per_node=5
)
```

## 📈 Graph Statistics

```python
stats = builder.get_statistics()
```

Returns:
```json
{
  "num_nodes": 10,
  "num_edges": 30,
  "density": 0.333,
  "citation_types": {
    "supports": 12,
    "contradicts": 8,
    "refers": 7,
    "neutral": 3
  },
  "degree_stats": {
    "in_degree": {"mean": 3.0, "median": 3.0, "max": 6, "min": 0},
    "out_degree": {"mean": 3.0, "median": 3.0, "max": 3, "min": 3}
  },
  "avg_clustering": 0.613,
  "largest_component_size": 10
}
```

## 🎨 Visualization

### Degree Distribution
```python
builder.visualize_degree_distribution(
    output_path="degree_dist.png",
    title="Citation Graph Degree Distribution"
)
```

Creates three histograms:
- **In-Degree**: Number of incoming citations
- **Out-Degree**: Number of outgoing citations  
- **Total Degree**: Sum of in and out degrees

## 💾 Export Formats

### JSON (Default)
```python
builder.save_graph("graph.json", format='json')
```

```json
{
  "nodes": [
    {
      "id": "CASE_001",
      "text": "Case text snippet...",
      "metadata": "..."
    }
  ],
  "edges": [
    {
      "source": "CASE_001",
      "target": "CASE_002",
      "citation_type": "supports",
      "similarity": 0.85,
      "weight": 0.85
    }
  ]
}
```

### GEXF (Gephi Compatible)
```python
builder.save_graph("graph.gexf", format='gexf')
```

### GraphML (Cytoscape Compatible)
```python
builder.save_graph("graph.graphml", format='graphml')
```

## 🔧 Advanced Usage

### Custom Models

```python
# Use legal-specific embeddings
builder = GraphBuilder(
    embedding_model="law-ai/InLegalBERT",  # Legal domain model
    classifier_model="microsoft/deberta-v3-large",  # Better classifier
    top_k=10
)
```

### Fine-tune Classification Thresholds

```python
# Modify classification logic in classify_citation_type()
# Adjust score thresholds for citation types
```

### Load from LecAI Dataset

```python
import pandas as pd

# Load case mapping
df = pd.read_csv("path/to/file_Case_id_and_name.csv")

documents = []
for _, row in df.iterrows():
    doc = {
        'doc_id': row['Case_id'],
        'case_name': row['Case_name'],
        'text': row['Case_name'],  # or load full text
        'file_name': row['file_name']
    }
    documents.append(doc)

builder.load_documents(documents)
```

### Process Large Datasets

```python
# Use batching for large corpora
builder.compute_embeddings(batch_size=32, show_progress=True)

# Use GPU FAISS for large graphs
builder.build_faiss_index(use_gpu=True)  # Requires faiss-gpu

# Limit edges per node
builder.build_citation_graph(max_edges_per_node=10)
```

## 📊 Demo Results

Running on 10 LecAI cases:

```
Graph Statistics:
  Nodes: 10
  Edges: 30
  Density: 0.333
  Average clustering: 0.613
  Citation types: {'contradicts': 30}
  
Degree Stats:
  In-degree: mean=3.0, max=6
  Out-degree: mean=3.0, max=3
```

## 🎓 Use Cases

1. **Citation Network Analysis** - Study how legal cases reference each other
2. **Precedent Discovery** - Find supporting/contradicting precedents
3. **Legal Knowledge Graphs** - Build knowledge graphs from legal corpora
4. **Citation Prediction** - Train link prediction models
5. **Document Clustering** - Cluster documents by citation patterns

## 🔬 Integration with Existing Framework

Combine with InLegalBERT External Agent:

```python
from lexai.graph_builder import GraphBuilder
from lexai.agents import InLegalBERTExternalAgent

# Build graph with InLegalBERT embeddings
builder = GraphBuilder(
    embedding_model="law-ai/InLegalBERT",
    top_k=5
)

# Or use external agent for retrieval + graph building
external_agent = InLegalBERTExternalAgent()
external_agent.load_dataset("lecai_data.csv")

# Extract documents for graph
documents = [
    {'doc_id': cid, 'text': case['case_name']}
    for cid, case in external_agent.cases.items()
]

builder.load_documents(documents)
builder.compute_embeddings()
builder.build_faiss_index()
builder.build_citation_graph()
```

## ⚙️ Configuration

### GraphBuilder Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `classifier_model` | `nli-deberta-v3-small` | Citation classifier |
| `device` | Auto-detect | 'cuda', 'cpu', or None |
| `top_k` | 5 | Nearest neighbors per document |

### build_citation_graph() Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.3 | Minimum similarity to create edge |
| `max_edges_per_node` | None | Max outgoing edges (None = top_k) |
| `text_field` | 'text' | Document text field |
| `id_field` | 'doc_id' | Document ID field |

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Use CPU
builder = GraphBuilder(device="cpu")

# Or reduce batch size
builder.compute_embeddings(batch_size=8)
```

### Slow Classification
```python
# Use smaller classifier
builder = GraphBuilder(
    classifier_model="cross-encoder/nli-MiniLM2-L6-H768"
)

# Or reduce top_k
builder = GraphBuilder(top_k=3)
```

### Empty Graph
```python
# Lower similarity threshold
builder.build_citation_graph(similarity_threshold=0.1)

# Increase top_k
builder = GraphBuilder(top_k=10)
```

## 📚 API Reference

### Class: GraphBuilder

#### Methods

- `load_documents(documents, text_field='text', id_field='doc_id')` - Load documents
- `compute_embeddings(text_field='text', batch_size=16)` - Compute embeddings
- `build_faiss_index(use_gpu=False)` - Build FAISS index
- `find_nearest_neighbors(doc_idx, k=None)` - Find k nearest neighbors
- `classify_citation_type(source_text, target_text)` - Classify citation relationship
- `build_citation_graph(...)` - Build complete citation graph
- `save_graph(output_path, format='json')` - Save graph to file
- `visualize_degree_distribution(output_path, title)` - Plot degree distribution
- `get_statistics()` - Get graph statistics

## 🤝 Contributing

Improvements welcome:
- [ ] Add more citation types (distinguishes/follows/overrules)
- [ ] Fine-tune classifier on legal citation data
- [ ] Add temporal analysis (citation evolution over time)
- [ ] Implement PageRank for case importance
- [ ] Add community detection

## 📄 License

See main LICENSE file.

---

**Version**: 1.0.0  
**Status**: Production Ready ✓  
**Last Updated**: November 2024
