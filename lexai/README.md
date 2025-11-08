# LexAI: Legal Citation and Reasoning Framework

A complete AI-powered legal analysis system combining external inference (precedent retrieval) with legal reasoning (multi-aspect analysis) for comprehensive case evaluation.

## 🎯 Features

### 1. **External Inference Agent**
- Retrieves relevant precedent cases using semantic similarity
- Computes Total Relevance Score (TRS) combining multiple factors:
  - **Similarity (S)**: Semantic embedding similarity
  - **Context Fit (C)**: TF-IDF contextual relevance
  - **Jurisdiction (J)**: Geographic + temporal alignment
  - **Internal Confidence (I)**: Model confidence in prediction
  - **Uncertainty (U)**: Consistency of relevance signals
- Determines case alignment (supports/contradicts/neutral)
- Extracts supporting text spans
- Outputs JSON compliant with standard schema

### 2. **Legal Reasoning Agent**
- Generates comprehensive legal analysis from retrieved precedents
- Supports both **rule-based** and **LLM-based** reasoning
- Multi-aspect analysis:
  - Precedent Support
  - Contradicting Authority
  - Jurisdictional Consistency
  - Temporal Relevance
  - Semantic Coherence
- Detailed evidence tables with TRS component breakdown
- Uncertainty and limitations analysis
- Actionable recommendations

### 3. **Integrated Pipeline**
- Complete end-to-end workflow
- LecAI dataset compatibility
- Citation graph analysis with NetworkX
- Batch processing support
- Export to JSON/JSONL

## 📦 Installation

### Requirements
```bash
# Python 3.8+
pip install -r requirements.txt
```

### Dependencies
```
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
scikit-learn>=1.3.0
transformers>=4.30.0
torch>=2.0.0
pandas>=2.0.0
networkx>=3.0
numpy>=1.24.0
```

## 🚀 Quick Start

### Basic Usage

```python
from lexai.legal_ai_pipeline import LegalAIPipeline

# Initialize pipeline
pipeline = LegalAIPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_llm_reasoning=False  # True for LLM-based reasoning
)

# Load dataset
pipeline.load_lecai_dataset(
    candidates_path="lexai/data/processed/candidates_sample.jsonl"
)

# Analyze case
target_case = {
    "case_id": "TARGET_001",
    "title": "Privacy Rights Case",
    "text": "The question is whether privacy is a fundamental right...",
    "year": 2020,
    "jurisdiction": "Supreme Court of India"
}

result = pipeline.analyze_case(
    target_case=target_case,
    top_k=5,
    internal_confidence=0.8
)

# Export results
pipeline.export_results(result, "output/analysis.json")
```

### Using Individual Agents

```python
from lexai.agents import ExternalInferenceAgent, LegalReasoningAgent

# External Inference
external_agent = ExternalInferenceAgent()
external_agent.build_index(candidates)
inference_result = external_agent.infer(target_case, top_k=5)

# Legal Reasoning
reasoning_agent = LegalReasoningAgent(use_llm=False)
reasoning_result = reasoning_agent.generate_reasoning(
    target_case=inference_result['target'],
    retrieved_cases=inference_result['retrieved_cases'],
    external_coherence_score=inference_result['overall_external_coherence_score']
)
```

## 🧪 Testing

### Run All Tests
```bash
# Test External Inference Agent
python lexai/agents/external_inference_agent.py --test

# Test Legal Reasoning Agent
python lexai/agents/legal_reasoning_agent.py --test

# Test Complete Pipeline
python lexai/legal_ai_pipeline.py --test
```

### Demo Mode
```bash
# Run without --test flag for interactive demo
python lexai/legal_ai_pipeline.py
```

## 📊 Data Format

### Input: Candidates Dataset (JSONL)
```json
{
  "case_id": "CASE_001",
  "title": "Case Title",
  "text": "Full case text...",
  "year": 2017,
  "jurisdiction": "Supreme Court of India"
}
```

### Input: Target Case
```python
target_case = {
    "case_id": "TARGET_001",
    "title": "Target Case Title",
    "text": "Target case text...",
    "year": 2020,
    "jurisdiction": "Supreme Court"
}
```

### Output: Complete Analysis
```json
{
  "target_case": { ... },
  "external_inference": {
    "retrieved_cases": [...],
    "external_coherence_score": 0.75,
    "short_summary": "..."
  },
  "legal_reasoning": {
    "summary_long": "...",
    "aspect_analysis": [...],
    "detailed_evidence_table": [...],
    "uncertainty_and_limits": "...",
    "recommended_next_steps": [...],
    "reasoning_method": "rule_based",
    "overall_coherence_score": 0.68
  },
  "citation_analysis": { ... },
  "pipeline_metadata": { ... }
}
```

## ⚙️ Configuration

### TRS Weights Customization
```python
# Default weights (balanced)
default_weights = {
    'similarity': 0.35,
    'context_fit': 0.20,
    'jurisdiction': 0.20,
    'internal_confidence': 0.15,
    'uncertainty': 0.10
}

# Constitutional law focus (higher temporal weight)
constitutional_weights = {
    'similarity': 0.30,
    'context_fit': 0.15,
    'jurisdiction': 0.30,  # Emphasize jurisdictional alignment
    'internal_confidence': 0.15,
    'uncertainty': 0.10
}

# Contract law focus (higher context fit)
contract_weights = {
    'similarity': 0.30,
    'context_fit': 0.35,  # Emphasize contextual relevance
    'jurisdiction': 0.15,
    'internal_confidence': 0.10,
    'uncertainty': 0.10
}

pipeline = LegalAIPipeline(trs_weights=constitutional_weights)
```

### LLM Configuration
```python
# Use local LLM for reasoning
pipeline = LegalAIPipeline(
    use_llm_reasoning=True,
    llm_model_name="meta-llama/Llama-2-7b-hf",  # or gpt2, etc.
    device="cuda"  # or "cpu"
)
```

### Device Selection
```python
# GPU acceleration
pipeline = LegalAIPipeline(device="cuda")

# CPU only
pipeline = LegalAIPipeline(device="cpu")

# Auto-detect
pipeline = LegalAIPipeline(device=None)  # Uses CUDA if available
```

## 📈 Citation Graph Analysis

```python
# Load with citation graph
pipeline.load_lecai_dataset(
    candidates_path="data/candidates.jsonl",
    citation_graph_path="data/citations.csv"  # Format: source,target
)

# Analyze with citation graph
result = pipeline.analyze_case(
    target_case=target_case,
    use_citation_graph=True
)

# Citation analysis includes:
# - Direct citation links
# - Common citations
# - PageRank scores
```

## 🔍 Advanced Features

### Batch Processing
```python
target_cases = [case1, case2, case3, ...]
results = pipeline.batch_analyze(
    target_cases=target_cases,
    top_k=5
)
```

### Statistics
```python
stats = pipeline.get_statistics()
# Returns:
# - External agent index stats
# - Reasoning agent config
# - Citation graph metrics
# - Dataset info
```

### Custom Retrieval
```python
from lexai.agents import ExternalInferenceAgent

agent = ExternalInferenceAgent()
agent.build_index(candidates)

# Get TRS factor breakdown
result = agent.infer(
    target=target_case,
    top_k=5,
    return_trs_factors=True  # Include factor dict in output
)
```

## 📚 Documentation

- **[TRS Implementation](TRS_IMPLEMENTATION.md)**: Detailed explanation of Total Relevance Score
- **[Usage Guide](USAGE_GUIDE.md)**: Comprehensive examples and use cases
- **[Architecture](ARCHITECTURE.md)**: System design and components
- **[API Reference](API_REFERENCE.md)**: Complete API documentation

## 🎓 Use Cases

1. **Legal Research**: Find relevant precedents for new cases
2. **Case Preparation**: Generate comprehensive legal analysis
3. **Academic Analysis**: Study citation networks and legal reasoning
4. **LecAI Integration**: Compatible with LecAI/Citeomatic datasets
5. **Custom Legal AI**: Build domain-specific legal assistants

## 🔧 Troubleshooting

### Model Download Issues
```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Memory Issues
```python
# Reduce batch size in embedding generation
agent.build_index(candidates, batch_size=16)  # Default: 32

# Use CPU if GPU memory limited
pipeline = LegalAIPipeline(device="cpu")
```

### Low Quality Results
```python
# Adjust TRS weights for your domain
custom_weights = {
    'similarity': 0.40,  # Increase semantic weight
    'context_fit': 0.25,  # Increase contextual weight
    'jurisdiction': 0.15,
    'internal_confidence': 0.10,
    'uncertainty': 0.10
}

# Increase top_k for more candidates
result = pipeline.analyze_case(target_case, top_k=10)

# Use LLM reasoning for better analysis
pipeline = LegalAIPipeline(use_llm_reasoning=True)
```

## 📊 Performance

### Benchmarks (on sample dataset)
- **Index Building**: ~0.5s for 5 cases (CPU)
- **Inference**: ~0.3s per case (top_k=5)
- **Rule-based Reasoning**: ~0.1s per case
- **LLM Reasoning**: ~2-5s per case (depends on model)

### Scalability
- **Tested**: Up to 10,000 candidates
- **FAISS**: Efficient for millions of vectors
- **Memory**: ~100MB per 10,000 candidates

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run tests: `python -m pytest tests/`
4. Submit pull request

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Built for LecAI/Citeomatic legal citation prediction
- Uses Sentence Transformers, FAISS, scikit-learn
- Inspired by legal information retrieval research

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready ✓
