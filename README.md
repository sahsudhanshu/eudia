# LexAI - Legal Document Intelligence System

LexAI is an intelligent system for processing, analyzing, and retrieving legal documents using state-of-the-art NLP and machine learning techniques.

## 🚀 Quick Start

### Installation

```bash
# Navigate to project directory
cd /home/anand/eudia/eudia

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# Run the ExternalInferenceAgent demo
python lexai/agents/external_inference_agent.py

# Run tests
python lexai/agents/external_inference_agent.py --test
```

### Basic Usage

```python
from lexai.agents import ExternalInferenceAgent
import json

# Initialize agent
agent = ExternalInferenceAgent()

# Load and index candidates
agent.build_index(candidates)

# Run inference
result = agent.infer(target_case, top_k=5, internal_confidence=0.8)

# Print results
print(json.dumps(result, indent=2))
```

## ✨ Features

### 📄 OCR and Document Extraction
- Extract text from legal PDFs using pdfplumber
- Identify case titles, citations, and article references
- Parse structured legal document metadata

### 🔍 Semantic Search and Retrieval
- **Sentence transformer-based embeddings** for semantic understanding
- **FAISS-powered efficient similarity search**
- Support for large-scale legal document corpora
- **Trust Relevance Score (TRS)** for multi-factor ranking

### 🤖 External Inference Agent

Advanced legal case retrieval with comprehensive scoring:

- **Similarity Score (S)**: Semantic similarity via embeddings
- **Context Fit (C)**: Contextual relevance via TF-IDF
- **Jurisdiction Score (J)**: Geographic and temporal alignment
- **Internal Confidence (I)**: Optional model confidence
- **Uncertainty (U)**: Prediction reliability estimation

**TRS Formula:**
```
TRS = (w_S × S) + (w_C × C) + (w_J × J) + (w_I × I) - (w_U × U)
Clipped to [0, 1]
```

### 🎯 Key Capabilities

- ✅ Deterministic retrieval (no LLM calls)
- ✅ Configurable TRS weights
- ✅ Alignment detection (supports/contradicts/neutral)
- ✅ Automatic span extraction (≤40 words)
- ✅ Comprehensive justifications
- ✅ GPU acceleration support
- ✅ Custom retriever integration

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
cd eudia

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

For faster embedding generation, install the GPU version of FAISS:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Quick Start

### 1. Extract Text from Legal Documents

```python
from ocr_agent import process_pdf

# Process a legal PDF
result = process_pdf("path/to/legal_document.pdf")

print(f"Title: {result['title']}")
print(f"Citations found: {len(result['citations'])}")
print(f"Articles referenced: {len(result['articles'])}")
```

### 2. Build a Semantic Search Index

```python
from lexai.agents import ExternalInferenceAgent
import json

# Initialize the agent
agent = ExternalInferenceAgent(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Load legal documents
with open("lexai/data/raw/document.json", "r") as f:
    doc = json.load(f)

# Prepare candidates
candidates = [
    {"text": "Constitutional rights are fundamental...", "source": "doc1"},
    {"text": "Article 21 guarantees life and liberty...", "source": "doc2"},
    # ... more documents
]

# Build search index
agent.build_index(candidates, text_field="text")

# Search for similar documents
results = agent.infer("right to privacy", top_k=5)

for result in results:
    print(f"Score: {result['similarity_score']:.4f}")
    print(f"Text: {result['text'][:100]}...")
```

### 3. Run the Example

```bash
python example_usage.py
```

## Project Structure

```
eudia/
├── app.py                      # Main application entry point
├── ocr_agent.py               # PDF text extraction agent
├── example_usage.py           # Example usage demonstration
├── requirements.txt           # Python dependencies
├── lexai/                     # Core LexAI package
│   ├── agents/               # Intelligent agents
│   │   ├── __init__.py
│   │   ├── external_inference_agent.py
│   │   └── README.md         # Agent documentation
│   └── data/                 # Data directory
│       └── raw/              # Raw legal documents
└── tests/                    # Test suite
    ├── __init__.py
    └── test_external_inference_agent.py
```

## Components

### OCR Agent (`ocr_agent.py`)

Extracts structured information from legal PDFs:
- **Text Extraction**: Uses pdfplumber for reliable text extraction
- **Title Detection**: Identifies case titles from document headers
- **Citation Parsing**: Extracts legal citations (case law, AIR references)
- **Article References**: Identifies constitutional articles and sections

### External Inference Agent

Provides semantic search capabilities:
- **Embedding Generation**: Creates dense vector representations using sentence transformers
- **FAISS Indexing**: Efficient similarity search with IndexFlatIP
- **Flexible Retrieval**: Supports custom retrievers or built-in search
- **Metadata Preservation**: Maintains document metadata throughout retrieval

See [lexai/agents/README.md](lexai/agents/README.md) for detailed documentation.

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lexai --cov-report=html

# Run specific test file
pytest tests/test_external_inference_agent.py -v
```

## Configuration

### Embedding Models

Choose different sentence transformer models based on your needs:

| Model | Dimension | Speed | Use Case |
|-------|-----------|-------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | General purpose, quick retrieval |
| all-mpnet-base-v2 | 768 | Medium | Better quality, balanced |
| legal-bert-base-uncased | 768 | Medium | Legal domain-specific |

### Device Selection

```python
# Use CPU
agent = ExternalInferenceAgent(device="cpu")

# Use GPU (if available)
agent = ExternalInferenceAgent(device="cuda")

# Auto-detect
agent = ExternalInferenceAgent(device=None)
```

## API Reference

### ExternalInferenceAgent

#### Methods

- `build_index(candidates, text_field="text")`: Build FAISS index from documents
- `infer(query, top_k=5, retriever=None)`: Retrieve similar documents
- `get_index_stats()`: Get index statistics
- `clear_index()`: Clear the current index

See [lexai/agents/README.md](lexai/agents/README.md) for complete API documentation.

## Performance Tips

1. **Batch Processing**: Process multiple queries together for better throughput
2. **GPU Acceleration**: Use GPU for encoding large document collections
3. **Model Selection**: Choose smaller models for speed, larger for accuracy
4. **Index Optimization**: For very large datasets (>1M documents), use approximate search methods

## Use Cases

### Legal Research
- Find precedents similar to a case description
- Discover related case law based on semantic similarity
- Search for judgments citing specific articles

### Document Analysis
- Identify similar legal arguments across cases
- Cluster related legal documents
- Extract and organize citations

### Knowledge Management
- Build searchable legal knowledge bases
- Organize case law libraries
- Semantic tagging of legal documents

## Roadmap

- [ ] Add support for more document formats (DOCX, HTML)
- [ ] Implement multi-lingual legal document support
- [ ] Add citation graph analysis
- [ ] Integrate with external legal databases
- [ ] Build web interface for document search
- [ ] Add fine-tuned legal domain models

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See LICENSE file for details.

## Acknowledgments

- Built with [sentence-transformers](https://www.sbert.net/)
- Powered by [FAISS](https://github.com/facebookresearch/faiss)
- PDF processing with [pdfplumber](https://github.com/jsvine/pdfplumber)

## Support

For questions or issues, please open an issue on the repository.
