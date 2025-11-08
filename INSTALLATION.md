# Installation Guide for Eudia Legal AI System

This guide will help you set up the Eudia legal AI system from scratch.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- 8GB+ RAM (16GB recommended)
- CUDA-compatible GPU (optional, for faster processing)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/sahsudhanshu/eudia.git
cd eudia
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Run tests to verify everything is working
pytest tests/

# Or run a quick demo
python lexai/quick_start_inlegalbert.py
```

## Detailed Installation

### Install Core Dependencies

```bash
# PDF Processing
pip install pdfplumber>=0.11.0

# Machine Learning
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install sentence-transformers>=5.0.0

# Vector Search
pip install faiss-cpu>=1.8.0

# LangChain
pip install langchain>=0.3.0
pip install langchain-community>=0.4.0

# Graph Processing
pip install networkx>=3.0
pip install matplotlib>=3.7.0

# Scientific Computing
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0
```

### GPU Support (Optional)

If you have a CUDA-compatible GPU:

```bash
# Uninstall CPU version
pip uninstall faiss-cpu

# Install GPU version
pip install faiss-gpu>=1.8.0

# Ensure you have the correct PyTorch version for your CUDA version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Project Structure

```
eudia/
├── lexai/                          # Main LexAI framework
│   ├── agents/                     # AI agents
│   │   ├── external_inference_agent.py
│   │   ├── inlegalbert_external_agent.py
│   │   ├── internal_coherence_agent.py
│   │   ├── ocr_agent.py
│   │   └── query_pdf_rag_ocr.py
│   ├── data/                       # Data files
│   │   ├── graphs/                 # Citation graphs
│   │   ├── processed/              # Processed data
│   │   └── raw/                    # Raw PDF files
│   ├── graph_builder.py            # Citation graph builder
│   └── legal_ai_pipeline.py        # Main pipeline
├── tests/                          # Unit tests
├── outputs/                        # Generated outputs
├── requirements.txt                # Python dependencies
└── README.md                       # Main documentation
```

## Data Setup

### Download Sample Legal Documents

Place your legal PDF documents in:
```
lexai/data/raw/
```

### Citation Graph Data

For citation graph analysis, you'll need:
- Legal case files (PDF or TXT format)
- Optional: LecAI baseline dataset

## Running the System

### 1. OCR and PDF Processing

```python
from lexai.agents.ocr_agent import process_pdf

result = process_pdf("path/to/legal_document.pdf")
print(result["raw_text"])
```

### 2. External Inference with InLegalBERT

```python
from lexai.agents.inlegalbert_external_agent import InLegalBERTExternalAgent

agent = InLegalBERTExternalAgent()
agent.load_dataset("path/to/lecai_data.csv")
result = agent.retrieve_and_reason(query_case, top_k=5)
```

### 3. Citation Graph Building

```bash
python citation_graph_builder.py \
    --input lexai/data/raw \
    --output lexai/data/processed/citation_graph.json \
    --png outputs/graph.png
```

### 4. Query PDF with RAG

```python
from lexai.agents.query_pdf_rag_ocr import query_pdf_with_ocr

result = query_pdf_with_ocr(
    pdf_path="path/to/document.pdf",
    query="What does the judgment say about Article 21?"
)
print(result["Answer"])
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```python
# Use CPU instead
import torch
torch.cuda.is_available = lambda: False

# Or reduce batch size in your code
```

#### 2. Import Errors

```bash
# Ensure you're in the correct directory
cd /path/to/eudia

# Run Python with module syntax
python -m lexai.agents.query_pdf_rag_ocr
```

#### 3. LangChain Deprecation Warnings

The warnings are informational. To use updated imports:
```bash
pip install langchain-huggingface
```

Then update imports:
```python
from langchain_huggingface import HuggingFaceEmbeddings
```

#### 4. Missing Models

Models will be downloaded automatically on first use. Ensure you have:
- Stable internet connection
- Sufficient disk space (~5GB for all models)

### Model Downloads

The system uses these models (downloaded automatically):
- `law-ai/InLegalBERT` (~400MB) - Legal domain embeddings
- `sentence-transformers/all-MiniLM-L6-v2` (~80MB) - General embeddings
- `microsoft/phi-2` (~5GB) - Text generation
- `google/flan-t5-small` (~300MB) - Logical reasoning

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_graph_builder.py

# Run with coverage
pytest --cov=lexai tests/
```

## Environment Variables (Optional)

Create a `.env` file for configuration:

```bash
# GPU settings
CUDA_VISIBLE_DEVICES=0

# Model cache directory
TRANSFORMERS_CACHE=/path/to/model/cache
HF_HOME=/path/to/huggingface/cache

# LangChain settings
LANGCHAIN_TRACING_V2=false
```

## Performance Optimization

### For Faster Processing

1. **Use GPU**: Install faiss-gpu and CUDA-enabled PyTorch
2. **Increase batch size**: Modify batch_size parameters
3. **Cache embeddings**: Store computed embeddings for reuse
4. **Use smaller models**: Replace Phi-2 with smaller alternatives

### Memory Management

```python
# Clear CUDA cache periodically
import torch
torch.cuda.empty_cache()

# Limit text length
max_length = 512  # tokens
```

## Development Setup

For contributing to the project:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy

# Format code
black lexai/

# Run linter
flake8 lexai/

# Type checking
mypy lexai/
```

## Docker Support (Coming Soon)

```bash
# Build Docker image
docker build -t eudia-legal-ai .

# Run container
docker run -p 8000:8000 eudia-legal-ai
```

## Additional Resources

- [LexAI Documentation](lexai/README.md)
- [Graph Builder Guide](lexai/GRAPH_BUILDER_README.md)
- [InLegalBERT Guide](lexai/INLEGALBERT_README.md)
- [API Reference](lexai/API_REFERENCE.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/sahsudhanshu/eudia/issues
- Documentation: See README files in respective directories

## License

See LICENSE file for details.
