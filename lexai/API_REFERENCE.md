# LexAI API Reference

Complete API documentation for the Legal AI Framework.

## Table of Contents

1. [ExternalInferenceAgent](#externalinferenceagent)
2. [LegalReasoningAgent](#legalreasoningagent)
3. [LegalAIPipeline](#legalaipipeline)
4. [Data Schemas](#data-schemas)

---

## ExternalInferenceAgent

Main class for retrieving relevant precedent cases using semantic similarity and TRS scoring.

### Constructor

```python
ExternalInferenceAgent(
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    trs_weights: Optional[Dict[str, float]] = None,
    retriever: str = "faiss",
    device: Optional[str] = None
)
```

**Parameters:**
- `embedding_model_name`: Sentence Transformer model name or path
- `trs_weights`: Dict with keys: `similarity`, `context_fit`, `jurisdiction`, `internal_confidence`, `uncertainty`. Defaults to balanced weights.
- `retriever`: Retrieval backend (currently only "faiss" supported)
- `device`: Device to run on ("cuda", "cpu", or None for auto-detect)

**Example:**
```python
agent = ExternalInferenceAgent(
    embedding_model_name="all-MiniLM-L6-v2",
    trs_weights={
        'similarity': 0.35,
        'context_fit': 0.20,
        'jurisdiction': 0.20,
        'internal_confidence': 0.15,
        'uncertainty': 0.10
    },
    device="cuda"
)
```

### Methods

#### `build_index(candidates: List[Dict[str, Any]]) -> None`

Build search index from candidate cases.

**Parameters:**
- `candidates`: List of candidate case dicts. Each must have:
  - `case_id` (str): Unique identifier
  - `title` (str): Case title
  - `text` (str): Full case text
  - `year` (int, optional): Year decided
  - `jurisdiction` (str, optional): Court/jurisdiction

**Raises:**
- `ValueError`: If candidates missing required fields

**Example:**
```python
candidates = [
    {
        "case_id": "CASE_001",
        "title": "Landmark Case",
        "text": "Full text...",
        "year": 2020,
        "jurisdiction": "Supreme Court"
    },
    # ... more cases
]
agent.build_index(candidates)
```

#### `infer(target: Dict[str, Any], top_k: int = 5, internal_confidence: Optional[float] = None, return_trs_factors: bool = False) -> Dict[str, Any]`

Retrieve and rank relevant precedent cases.

**Parameters:**
- `target`: Target case dict (same schema as candidates)
- `top_k`: Number of top cases to retrieve
- `internal_confidence`: Optional model confidence score [0, 1]
- `return_trs_factors`: Include TRS component breakdown in output

**Returns:**
Dict with keys:
- `target` (dict): Target case info
- `retrieved_cases` (list): Top-k ranked cases with scores
- `overall_external_coherence_score` (float): Aggregate coherence
- `short_summary` (str): Brief summary

**Example:**
```python
target = {
    "case_id": "TARGET_001",
    "title": "New Case",
    "text": "Case facts...",
    "year": 2023,
    "jurisdiction": "High Court"
}

result = agent.infer(
    target=target,
    top_k=5,
    internal_confidence=0.85,
    return_trs_factors=True
)

print(result['overall_external_coherence_score'])  # 0.75
print(len(result['retrieved_cases']))  # 5
```

#### `get_index_stats() -> Dict[str, Any]`

Get statistics about the built index.

**Returns:**
- `is_built` (bool): Whether index is built
- `num_candidates` (int): Number of indexed cases
- `embedding_dim` (int): Dimension of embeddings
- `model_name` (str): Model name

**Example:**
```python
stats = agent.get_index_stats()
# {'is_built': True, 'num_candidates': 1000, 'embedding_dim': 384, 'model_name': '...'}
```

---

## LegalReasoningAgent

Generates comprehensive multi-aspect legal reasoning from retrieved precedents.

### Constructor

```python
LegalReasoningAgent(
    model_name: Optional[str] = None,
    use_llm: bool = False,
    device: Optional[str] = None
)
```

**Parameters:**
- `model_name`: Hugging Face model name (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
- `use_llm`: Whether to use LLM (True) or rule-based reasoning (False)
- `device`: Device to run on ("cuda", "cpu", or None)

**Example:**
```python
# Rule-based reasoning (fast, deterministic)
agent_rule = LegalReasoningAgent(use_llm=False)

# LLM-based reasoning (flexible, requires model)
agent_llm = LegalReasoningAgent(
    model_name="gpt2",
    use_llm=True,
    device="cuda"
)
```

### Methods

#### `generate_reasoning(target_case: Dict[str, Any], retrieved_cases: List[Dict[str, Any]], external_coherence_score: float) -> Dict[str, Any]`

Generate comprehensive legal reasoning analysis.

**Parameters:**
- `target_case`: Target case dict
- `retrieved_cases`: List of retrieved precedent cases (from ExternalInferenceAgent)
- `external_coherence_score`: Overall coherence score from external inference

**Returns:**
Dict with keys:
- `summary_long` (str): Comprehensive analysis narrative
- `aspect_analysis` (list): 5 aspects with strength ratings
  - Precedent Support
  - Contradicting Authority
  - Jurisdictional Consistency
  - Temporal Relevance
  - Semantic Coherence
- `detailed_evidence_table` (list): Per-case evidence with TRS breakdown
- `uncertainty_and_limits` (str): Uncertainty analysis and limitations
- `recommended_next_steps` (list): Actionable recommendations
- `reasoning_method` (str): "rule_based" or "llm"
- `overall_coherence_score` (float): Aggregate coherence [0, 1]

**Example:**
```python
reasoning = agent.generate_reasoning(
    target_case=target,
    retrieved_cases=inference_result['retrieved_cases'],
    external_coherence_score=inference_result['overall_external_coherence_score']
)

print(reasoning['summary_long'])
for aspect in reasoning['aspect_analysis']:
    print(f"{aspect['aspect']}: {aspect['strength']}")
```

---

## LegalAIPipeline

Complete end-to-end pipeline combining external inference and legal reasoning.

### Constructor

```python
LegalAIPipeline(
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    trs_weights: Optional[Dict[str, float]] = None,
    use_llm_reasoning: bool = False,
    llm_model_name: Optional[str] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `embedding_model`: Sentence Transformer model
- `trs_weights`: Custom TRS weights
- `use_llm_reasoning`: Use LLM for reasoning (vs rule-based)
- `llm_model_name`: LLM model name if use_llm_reasoning=True
- `device`: Device to run on

**Example:**
```python
pipeline = LegalAIPipeline(
    embedding_model="all-MiniLM-L6-v2",
    trs_weights={'similarity': 0.4, 'context_fit': 0.25, ...},
    use_llm_reasoning=False,
    device="cuda"
)
```

### Methods

#### `load_lecai_dataset(candidates_path: str, citation_graph_path: Optional[str] = None) -> int`

Load LecAI dataset from JSONL file.

**Parameters:**
- `candidates_path`: Path to candidates JSONL file
- `citation_graph_path`: Optional path to citation graph CSV (format: source,target)

**Returns:**
- Number of candidates loaded

**Example:**
```python
num_cases = pipeline.load_lecai_dataset(
    candidates_path="data/candidates.jsonl",
    citation_graph_path="data/citations.csv"
)
print(f"Loaded {num_cases} cases")
```

#### `analyze_case(target_case: Dict[str, Any], top_k: int = 5, internal_confidence: Optional[float] = None, use_citation_graph: bool = False) -> Dict[str, Any]`

Complete case analysis pipeline.

**Parameters:**
- `target_case`: Target case dict
- `top_k`: Number of precedents to retrieve
- `internal_confidence`: Optional model confidence [0, 1]
- `use_citation_graph`: Include citation graph analysis

**Returns:**
Complete analysis dict with:
- `target_case`: Target case info
- `external_inference`: External inference results
- `legal_reasoning`: Legal reasoning analysis
- `citation_analysis`: Citation graph analysis (if enabled)
- `pipeline_metadata`: Pipeline configuration

**Example:**
```python
result = pipeline.analyze_case(
    target_case=target,
    top_k=5,
    internal_confidence=0.8,
    use_citation_graph=True
)

print(result['external_inference']['external_coherence_score'])
print(result['legal_reasoning']['overall_coherence_score'])
```

#### `batch_analyze(target_cases: List[Dict[str, Any]], top_k: int = 5, internal_confidence: Optional[float] = None) -> List[Dict[str, Any]]`

Analyze multiple cases in batch.

**Parameters:**
- `target_cases`: List of target case dicts
- `top_k`: Number of precedents per case
- `internal_confidence`: Optional model confidence

**Returns:**
- List of complete analysis results

**Example:**
```python
results = pipeline.batch_analyze(
    target_cases=[case1, case2, case3],
    top_k=5
)
for result in results:
    print(result['target_case']['title'])
```

#### `export_results(results: Dict[str, Any], output_path: str, format: str = "json") -> None`

Export analysis results to file.

**Parameters:**
- `results`: Analysis results dict
- `output_path`: Path to save file
- `format`: Output format ("json" or "jsonl")

**Example:**
```python
pipeline.export_results(result, "output/analysis.json", format="json")
```

#### `get_statistics() -> Dict[str, Any]`

Get pipeline statistics.

**Returns:**
Dict with:
- `external_agent`: External agent stats
- `reasoning_agent`: Reasoning agent config
- `citation_graph`: Graph metrics (if loaded)
- `dataset`: Dataset info (if loaded)

**Example:**
```python
stats = pipeline.get_statistics()
print(stats['external_agent']['num_candidates'])
print(stats['citation_graph']['nodes'])
```

---

## Data Schemas

### Candidate Case Schema

Required fields for all candidate and target cases:

```python
{
    "case_id": str,        # Unique identifier
    "title": str,          # Case title
    "text": str,           # Full case text
    "year": int,           # Year decided (optional but recommended)
    "jurisdiction": str    # Court/jurisdiction (optional but recommended)
}
```

### TRS Weights Schema

Custom weights for Total Relevance Score:

```python
{
    "similarity": float,           # Semantic similarity weight [0, 1]
    "context_fit": float,          # Contextual relevance weight [0, 1]
    "jurisdiction": float,         # Jurisdictional alignment weight [0, 1]
    "internal_confidence": float,  # Model confidence weight [0, 1]
    "uncertainty": float           # Uncertainty penalty weight [0, 1]
}
# Note: Weights should sum to ~1.0 but normalization is automatic
```

### External Inference Output Schema

```python
{
    "target": {
        "case_id": str,
        "title": str,
        "year": int,
        "jurisdiction": str
    },
    "retrieved_cases": [
        {
            "case_id": str,
            "title": str,
            "year": int,
            "jurisdiction": str,
            "similarity_score": float,       # [0, 1]
            "context_fit": float,            # [0, 1]
            "jurisdiction_score": float,     # [0, 1]
            "internal_confidence": float,    # [0, 1]
            "uncertainty": float,            # [0, ∞)
            "trs": float,                    # [0, 1]
            "alignment_type": str,           # "supports"/"contradicts"/"neutral"
            "justification": str,
            "spans": {
                "target_span": str,
                "candidate_span": str
            }
        }
    ],
    "overall_external_coherence_score": float,  # [0, 1]
    "short_summary": str
}
```

### Legal Reasoning Output Schema

```python
{
    "summary_long": str,
    "aspect_analysis": [
        {
            "aspect": str,         # Aspect name
            "analysis": str,       # Analysis text
            "strength": str        # "low"/"medium"/"high"
        }
    ],
    "detailed_evidence_table": [
        {
            "rank": int,
            "case_id": str,
            "case_title": str,
            "year": int,
            "jurisdiction": str,
            "trs": float,
            "similarity_score": float,
            "context_fit": float,
            "jurisdiction_score": float,
            "alignment_type": str,
            "internal_confidence": float,
            "uncertainty": float,
            "key_reasoning": str,
            "target_span": str,
            "candidate_span": str
        }
    ],
    "uncertainty_and_limits": str,
    "recommended_next_steps": [str],
    "reasoning_method": str,              # "rule_based" or "llm"
    "overall_coherence_score": float      # [0, 1]
}
```

### Complete Pipeline Output Schema

```python
{
    "target_case": {...},                      # Target case info
    "external_inference": {...},               # External inference output
    "legal_reasoning": {...},                  # Legal reasoning output
    "citation_analysis": {                     # Optional, if citation graph enabled
        "target_in_graph": bool,
        "retrieved_in_graph": [...],
        "citation_links": [...],
        "common_citations": [...],
        "pagerank_scores": {...}
    },
    "pipeline_metadata": {
        "embedding_model": str,
        "reasoning_method": str,
        "top_k": int,
        "internal_confidence": float
    }
}
```

---

## Error Handling

### Common Exceptions

**ValueError**: Invalid input data
```python
try:
    agent.build_index(candidates)
except ValueError as e:
    print(f"Invalid candidate data: {e}")
```

**RuntimeError**: Index not built
```python
try:
    result = agent.infer(target)
except RuntimeError as e:
    print(f"Must build index first: {e}")
```

**ImportError**: Missing dependencies
```python
try:
    from lexai.agents import ExternalInferenceAgent
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
```

---

## Type Hints

Complete type signatures for all public methods:

```python
from typing import List, Dict, Any, Optional

class ExternalInferenceAgent:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        trs_weights: Optional[Dict[str, float]] = None,
        retriever: str = "faiss",
        device: Optional[str] = None
    ) -> None: ...
    
    def build_index(self, candidates: List[Dict[str, Any]]) -> None: ...
    
    def infer(
        self,
        target: Dict[str, Any],
        top_k: int = 5,
        internal_confidence: Optional[float] = None,
        return_trs_factors: bool = False
    ) -> Dict[str, Any]: ...
    
    def get_index_stats(self) -> Dict[str, Any]: ...

class LegalReasoningAgent:
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_llm: bool = False,
        device: Optional[str] = None
    ) -> None: ...
    
    def generate_reasoning(
        self,
        target_case: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        external_coherence_score: float
    ) -> Dict[str, Any]: ...

class LegalAIPipeline:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        trs_weights: Optional[Dict[str, float]] = None,
        use_llm_reasoning: bool = False,
        llm_model_name: Optional[str] = None,
        device: Optional[str] = None
    ) -> None: ...
    
    def load_lecai_dataset(
        self,
        candidates_path: str,
        citation_graph_path: Optional[str] = None
    ) -> int: ...
    
    def analyze_case(
        self,
        target_case: Dict[str, Any],
        top_k: int = 5,
        internal_confidence: Optional[float] = None,
        use_citation_graph: bool = False
    ) -> Dict[str, Any]: ...
    
    def batch_analyze(
        self,
        target_cases: List[Dict[str, Any]],
        top_k: int = 5,
        internal_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]: ...
    
    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ) -> None: ...
    
    def get_statistics(self) -> Dict[str, Any]: ...
```

---

## Version Information

- **API Version**: 1.0.0
- **Last Updated**: 2024
- **Compatibility**: Python 3.8+

---

## Additional Resources

- **[README](README.md)**: Quick start guide
- **[TRS Implementation](TRS_IMPLEMENTATION.md)**: TRS scoring details
- **[Usage Guide](USAGE_GUIDE.md)**: Examples and use cases
- **[Architecture](ARCHITECTURE.md)**: System design

---

For questions or bug reports, please open a GitHub issue.
