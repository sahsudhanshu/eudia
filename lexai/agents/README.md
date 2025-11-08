# LexAI Agents

This module contains intelligent agents for legal document processing and analysis.

## ExternalInferenceAgent

The `ExternalInferenceAgent` is a similarity-based retrieval agent that uses sentence transformers and FAISS for efficient semantic search across legal documents.

### Features

- **Sentence Transformers**: Uses state-of-the-art embedding models for semantic understanding
- **FAISS Integration**: Efficient similarity search using Facebook's FAISS library
- **Normalized Embeddings**: Cosine similarity computation via normalized inner product
- **Flexible Retrieval**: Support for custom retrievers or built-in FAISS search
- **Type-Safe**: Full type hints for better IDE support and code quality

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For GPU support, install the GPU version of FAISS:

```bash
pip install faiss-gpu
```

### Usage

#### Basic Example

```python
from lexai.agents import ExternalInferenceAgent

# Initialize the agent
agent = ExternalInferenceAgent(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"  # or "cuda" for GPU
)

# Prepare candidate documents
candidates = [
    {"text": "Article 21 guarantees the right to life and personal liberty..."},
    {"text": "The Constitution of India provides fundamental rights..."},
    {"text": "Privacy is a fundamental right under Article 21..."}
]

# Build the index
agent.build_index(candidates, text_field="text")

# Perform similarity search
results = agent.infer("right to privacy", top_k=2)

# Access results
for result in results:
    print(f"Similarity: {result['similarity_score']:.4f}")
    print(f"Text: {result['text']}")
```

#### Advanced Usage with Custom Fields

```python
# Candidates with additional metadata
candidates = [
    {
        "text": "Full text of the judgment...",
        "case_name": "K.S. Puttaswamy v. Union of India",
        "year": 2017,
        "court": "Supreme Court of India"
    },
    # ... more candidates
]

# Build index
agent.build_index(candidates, text_field="text")

# Search and access metadata
results = agent.infer("privacy rights", top_k=5)
for result in results:
    print(f"{result['case_name']} ({result['year']})")
    print(f"Similarity: {result['similarity_score']:.4f}")
```

#### Index Management

```python
# Get index statistics
stats = agent.get_index_stats()
print(f"Candidates indexed: {stats['num_candidates']}")
print(f"Embedding dimension: {stats['embedding_dimension']}")

# Clear the index
agent.clear_index()
```

### API Reference

#### `__init__(model_name, device)`

Initialize the ExternalInferenceAgent.

**Parameters:**
- `model_name` (str): Sentence transformer model name (default: "sentence-transformers/all-MiniLM-L6-v2")
- `device` (str, optional): Device to use ("cpu", "cuda", or None for auto-detect)

#### `build_index(candidates, text_field)`

Build FAISS index from candidate documents.

**Parameters:**
- `candidates` (List[Dict]): List of candidate documents
- `text_field` (str): Key in candidate dict containing text (default: "text")

**Raises:**
- `ValueError`: If candidates is empty or missing required fields

#### `infer(query, top_k, retriever)`

Perform similarity-based inference.

**Parameters:**
- `query` (str): Query text
- `top_k` (int): Number of results to return (default: 5)
- `retriever` (optional): Custom retriever object

**Returns:**
- List[Dict]: Candidate documents with added `similarity_score` field

**Raises:**
- `ValueError`: If index hasn't been built

#### `get_index_stats()`

Get statistics about the current index.

**Returns:**
- Dict with keys: `is_built`, `num_candidates`, `embedding_dimension`, `model_name`

#### `clear_index()`

Clear the current index and candidate data.

### Implementation Details

1. **Embedding Creation**: Uses sentence-transformers to encode text into dense vectors
2. **Normalization**: Embeddings are L2-normalized for cosine similarity computation
3. **FAISS Index**: Uses `IndexFlatIP` (inner product) on normalized vectors
4. **Similarity Score**: Returns inner product scores (equivalent to cosine similarity for normalized vectors)

### Performance Considerations

- **Model Selection**: Smaller models like `all-MiniLM-L6-v2` are faster but less accurate than larger models
- **Batch Processing**: Encoding is done in batches for efficiency
- **GPU Acceleration**: Use GPU for faster encoding of large corpora
- **Index Type**: `IndexFlatIP` provides exact search; consider approximate methods (e.g., `IndexIVFFlat`) for very large datasets

### Example Models

| Model | Dimension | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| legal-bert-base-uncased | 768 | Medium | Domain-specific |

### Error Handling

The agent validates inputs and provides clear error messages:

```python
# Empty candidates
agent.build_index([])  # Raises ValueError

# Missing text field
candidates = [{"content": "text"}]
agent.build_index(candidates, text_field="text")  # Raises ValueError

# Search before building index
agent.infer("query")  # Raises ValueError
```

### Testing

Run the example usage script:

```bash
python example_usage.py
```

This will demonstrate the agent with a legal document from the data directory.

## License

See LICENSE file in the root directory.
