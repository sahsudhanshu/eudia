# ExternalInferenceAgent - Complete Usage Guide

## Quick Start

### Installation

```bash
# Navigate to the eudia directory
cd /home/anand/eudia/eudia

# Install dependencies
pip install -r requirements.txt

# The sentence-transformer model will be downloaded automatically on first use
```

### Basic Example

```python
from lexai.agents import ExternalInferenceAgent
import json

# Initialize agent
agent = ExternalInferenceAgent()

# Prepare candidates (legal cases)
candidates = [
    {
        "case_id": "CASE_001",
        "title": "K.S. Puttaswamy v. Union of India",
        "text": "The right to privacy is a fundamental right...",
        "year": 2017,
        "jurisdiction": "Supreme Court of India"
    },
    # ... more candidates
]

# Build index
agent.build_index(candidates)

# Prepare target case
target = {
    "case_id": "TARGET_001",
    "title": "Privacy Rights Case",
    "text": "The question is whether privacy is fundamental...",
    "year": 2020,
    "jurisdiction": "Supreme Court of India"
}

# Run inference
result = agent.infer(target, top_k=5, internal_confidence=0.8)

# Print results
print(json.dumps(result, indent=2))
```

## Running the Demo

### Standard Demo

```bash
cd /home/anand/eudia/eudia
python lexai/agents/external_inference_agent.py
```

This will:
1. Create sample legal cases
2. Build FAISS index
3. Run inference
4. Display formatted results
5. Run quick validations

### Test Mode

```bash
python lexai/agents/external_inference_agent.py --test
```

Runs comprehensive tests:
- Index building validation
- Schema validation
- TRS bounds checking

## API Reference

### Class: ExternalInferenceAgent

#### Constructor

```python
ExternalInferenceAgent(
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    trs_weights: Optional[Dict[str, float]] = None,
    retriever: Optional[Callable] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `embedding_model_name`: Sentence transformer model name
- `trs_weights`: Custom TRS weights dict (see TRS_IMPLEMENTATION.md)
- `retriever`: Custom retriever function (optional)
- `device`: "cpu", "cuda", or None for auto-detect

**Default TRS Weights:**
```python
{
    "w_S": 0.5,   # Similarity
    "w_C": 0.2,   # Context fit
    "w_J": 0.1,   # Jurisdiction score
    "w_I": 0.15,  # Internal confidence
    "w_U": 0.05   # Uncertainty
}
```

#### Methods

##### build_index()

```python
build_index(
    candidates: List[Dict[str, Any]],
    text_field: str = "text"
) -> None
```

Build FAISS index from candidate documents.

**Required candidate fields:**
- `case_id`: Unique identifier
- `title`: Case title
- `text`: Full case text
- `year`: Year (optional)
- `jurisdiction`: Jurisdiction (optional)

**Example:**
```python
candidates = [
    {
        "case_id": "CASE_001",
        "title": "Sample v. Case",
        "text": "Full text of the legal case...",
        "year": 2020,
        "jurisdiction": "Supreme Court"
    }
]
agent.build_index(candidates)
```

##### infer()

```python
infer(
    target: Dict[str, Any],
    top_k: int = 5,
    internal_confidence: Optional[float] = None,
    return_factors: bool = False
) -> Dict[str, Any]
```

Perform inference to retrieve similar cases.

**Parameters:**
- `target`: Target case dict (same schema as candidates)
- `top_k`: Number of cases to retrieve
- `internal_confidence`: Optional confidence score [0, 1]
- `return_factors`: If True, return detailed TRS breakdown

**Returns:** Dictionary matching OUTPUT_SCHEMA (see below)

##### get_index_stats()

```python
get_index_stats() -> Dict[str, Any]
```

Get statistics about the current index.

**Returns:**
```python
{
    "is_built": True,
    "num_candidates": 5,
    "embedding_dimension": 384,
    "model_name": "sentence-transformers/all-MiniLM-L6-v2"
}
```

##### clear_index()

```python
clear_index() -> None
```

Clear the index and reset candidate data.

## Output Schema

The `infer()` method returns a dictionary with the following structure:

```json
{
  "target": {
    "case_id": "TARGET_001",
    "title": "Case Title",
    "year": 2020,
    "jurisdiction": "Supreme Court"
  },
  "retrieved_cases": [
    {
      "case_id": "CASE_001",
      "title": "Retrieved Case Title",
      "year": 2017,
      "jurisdiction": "Supreme Court",
      "similarity_score": 0.85,
      "context_fit": 0.72,
      "jurisdiction_score": 0.95,
      "internal_confidence": 0.80,
      "uncertainty": 0.15,
      "trs": 0.78,
      "alignment_type": "supports",
      "justification": "High semantic similarity (0.85)...",
      "spans": {
        "target_span": "Relevant excerpt from target...",
        "candidate_span": "Relevant excerpt from candidate..."
      }
    }
  ],
  "overall_external_coherence_score": 0.75,
  "short_summary": "Analysis retrieved 5 cases with overall score 0.75..."
}
```

### Field Descriptions

#### Target
- `case_id`: Identifier
- `title`: Case title
- `year`: Year of the case
- `jurisdiction`: Legal jurisdiction

#### Retrieved Cases (sorted by TRS descending)
- `case_id`: Candidate identifier
- `title`: Candidate title
- `year`: Candidate year
- `jurisdiction`: Candidate jurisdiction
- `similarity_score`: Semantic similarity [0, 1]
- `context_fit`: Contextual relevance [0, 1]
- `jurisdiction_score`: Jurisdictional alignment [0, 1]
- `internal_confidence`: Internal model confidence [0, 1]
- `uncertainty`: Prediction uncertainty [0, 1]
- `trs`: Trust Relevance Score [0, 1]
- `alignment_type`: "supports", "contradicts", or "neutral"
- `justification`: 1-3 sentence explanation
- `spans`: Key excerpts (≤40 words each)

#### Overall Metrics
- `overall_external_coherence_score`: Mean TRS across retrieved cases [0, 1]
- `short_summary`: One-paragraph summary of results

## Advanced Usage

### Custom TRS Weights

```python
# Emphasize semantic similarity
custom_weights = {
    "w_S": 0.6,
    "w_C": 0.25,
    "w_J": 0.05,
    "w_I": 0.1,
    "w_U": 0.0
}

agent = ExternalInferenceAgent(trs_weights=custom_weights)
```

### With Factor Breakdown

```python
result = agent.infer(
    target,
    top_k=3,
    internal_confidence=0.8,
    return_factors=True  # Get detailed TRS breakdown
)

# Access detailed factors
for case in result['retrieved_cases']:
    trs_details = case['trs']
    print(f"Score: {trs_details['score']}")
    print(f"Similarity: {trs_details['factors']['similarity']}")
    print(f"Context Fit: {trs_details['factors']['context_fit']}")
```

### Custom Retriever

```python
def my_custom_retriever(query_text: str, top_k: int):
    # Custom retrieval logic
    # Must return List[Tuple[int, float]] of (index, score)
    return [(0, 0.9), (1, 0.8), (2, 0.7)]

agent = ExternalInferenceAgent(retriever=my_custom_retriever)
agent.build_index(candidates)  # Still needed for metadata

# Will use custom retriever instead of FAISS
result = agent.infer(target, top_k=3)
```

### GPU Acceleration

```python
# Use GPU for faster encoding
agent = ExternalInferenceAgent(device="cuda")

# Or install GPU FAISS
# pip uninstall faiss-cpu
# pip install faiss-gpu
```

## Loading Data from Files

### From JSONL

```python
import json

# Load candidates from JSONL
candidates = []
with open('lexai/data/processed/candidates_sample.jsonl', 'r') as f:
    for line in f:
        candidates.append(json.loads(line))

agent = ExternalInferenceAgent()
agent.build_index(candidates)
```

### From JSON

```python
import json

# Load from JSON file
with open('data/cases.json', 'r') as f:
    data = json.load(f)
    candidates = data['candidates']
    target = data['target']

agent = ExternalInferenceAgent()
agent.build_index(candidates)
result = agent.infer(target, top_k=5)
```

## Common Patterns

### Batch Processing

```python
# Process multiple targets
targets = [target1, target2, target3]
results = []

agent = ExternalInferenceAgent()
agent.build_index(candidates)  # Build once

for target in targets:
    result = agent.infer(target, top_k=5)
    results.append(result)
```

### Filtering Results

```python
# Get only high-confidence results
result = agent.infer(target, top_k=10)

high_confidence_cases = [
    case for case in result['retrieved_cases']
    if case['trs'] >= 0.7
]
```

### Alignment Analysis

```python
result = agent.infer(target, top_k=10)

supporting = [c for c in result['retrieved_cases'] if c['alignment_type'] == 'supports']
contradicting = [c for c in result['retrieved_cases'] if c['alignment_type'] == 'contradicts']

print(f"Supporting cases: {len(supporting)}")
print(f"Contradicting cases: {len(contradicting)}")
```

## Troubleshooting

### Issue: "Index has not been built"

**Solution:** Call `build_index()` before `infer()`
```python
agent.build_index(candidates)  # Must call first
result = agent.infer(target)
```

### Issue: "Missing required field 'case_id'"

**Solution:** Ensure all candidates have required fields
```python
# Required fields
candidate = {
    "case_id": "...",  # Required
    "title": "...",    # Required
    "text": "...",     # Required
    "year": 2020,      # Optional but recommended
    "jurisdiction": "..." # Optional but recommended
}
```

### Issue: Low TRS scores

**Possible causes:**
1. Poor semantic similarity
2. Low context overlap
3. Different jurisdiction
4. Large temporal gap
5. High uncertainty

**Solutions:**
- Adjust TRS weights for your domain
- Increase candidate corpus size
- Filter by jurisdiction first
- Use domain-specific embedding models

### Issue: Slow performance

**Optimizations:**
1. Use GPU: `agent = ExternalInferenceAgent(device="cuda")`
2. Reduce `top_k`: Only retrieve what you need
3. Use smaller embedding model
4. Implement approximate FAISS index for large corpora

## Performance Benchmarks

On a typical laptop (CPU):
- **Index building**: ~1-2 seconds for 1000 candidates
- **Single inference**: ~200-500ms for top_k=5
- **Batch inference**: ~100-200ms per target (amortized)

With GPU:
- **Index building**: ~500ms for 1000 candidates
- **Single inference**: ~50-100ms for top_k=5

## Best Practices

1. **Build index once**, reuse for multiple queries
2. **Validate input data** before building index
3. **Use appropriate top_k** (5-10 for most use cases)
4. **Set internal_confidence** when available
5. **Filter by TRS threshold** for quality control
6. **Monitor uncertainty scores** for reliability
7. **Cache results** for repeated queries
8. **Use return_factors=True** for debugging

## Examples Directory

See `example_usage.py` for a complete working example with legal document processing.

## Testing

```bash
# Run all tests
python lexai/agents/external_inference_agent.py --test

# Run with pytest (if installed)
pytest tests/test_external_inference_agent.py -v
```

## Support

For issues or questions:
1. Check this guide
2. Review TRS_IMPLEMENTATION.md for scoring details
3. Check ARCHITECTURE.md for system design
4. Run demo with `--test` flag for validation

## License

See LICENSE file in the repository root.
