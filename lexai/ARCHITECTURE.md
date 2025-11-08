# LexAI System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     LexAI Legal AI Framework                    │
│                                                                 │
│  ┌───────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Data Loading     │→ │ External         │→ │  Legal      │ │
│  │  & Indexing       │  │ Inference        │  │  Reasoning  │ │
│  └───────────────────┘  └──────────────────┘  └─────────────┘ │
│           ↓                      ↓                     ↓        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Complete Legal Analysis Output               │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. External Inference Agent

```
Input: Target Case + Candidate Corpus
        ↓
┌──────────────────────────────────────────┐
│   External Inference Agent               │
│                                          │
│  ┌────────────────────────────────┐    │
│  │  1. Embedding Generation       │    │
│  │     - Sentence Transformers    │    │
│  │     - Normalize vectors        │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  2. FAISS Similarity Search    │    │
│  │     - IndexFlatIP              │    │
│  │     - Top-k retrieval          │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  3. TRS Computation            │    │
│  │     - Similarity (S)           │    │
│  │     - Context Fit (C)          │    │
│  │     - Jurisdiction (J)         │    │
│  │     - Internal Conf (I)        │    │
│  │     - Uncertainty (U)          │    │
│  │     TRS = wS·S + wC·C + wJ·J   │    │
│  │         + wI·I - wU·U          │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  4. Alignment Detection        │    │
│  │     - Keyword-based heuristics │    │
│  │     - supports/contradicts     │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  5. Span Extraction            │    │
│  │     - Sentence-level matching  │    │
│  │     - TF-IDF similarity        │    │
│  └────────────────────────────────┘    │
└──────────────────────────────────────────┘
        ↓
Output: Ranked Cases with TRS Scores
```

### 2. Legal Reasoning Agent

```
Input: Target + Retrieved Cases + Coherence Score
        ↓
┌──────────────────────────────────────────┐
│   Legal Reasoning Agent                  │
│                                          │
│  ┌────────────────────────────────┐    │
│  │  Mode Selection                │    │
│  │  ┌──────────┐   ┌───────────┐ │    │
│  │  │ Rule-    │   │ LLM-      │ │    │
│  │  │ Based    │ or│ Based     │ │    │
│  │  └──────────┘   └───────────┘ │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  1. Summary Generation         │    │
│  │     - Overall analysis         │    │
│  │     - Support/contradict count │    │
│  │     - Key precedents           │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  2. Multi-Aspect Analysis      │    │
│  │     - Precedent Support        │    │
│  │     - Contradicting Authority  │    │
│  │     - Jurisdictional Consist.  │    │
│  │     - Temporal Relevance       │    │
│  │     - Semantic Coherence       │    │
│  │     (Each: low/medium/high)    │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  3. Evidence Table             │    │
│  │     - Per-case TRS breakdown   │    │
│  │     - All score components     │    │
│  │     - Spans + justification    │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  4. Uncertainty Analysis       │    │
│  │     - Avg/max uncertainty      │    │
│  │     - False positive detection │    │
│  │     - Limitations              │    │
│  └────────────────────────────────┘    │
│                ↓                        │
│  ┌────────────────────────────────┐    │
│  │  5. Recommendations            │    │
│  │     - Context-aware actions    │    │
│  │     - Next steps               │    │
│  └────────────────────────────────┘    │
└──────────────────────────────────────────┘
        ↓
Output: Comprehensive Legal Reasoning
```

### 3. Integrated Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   LegalAIPipeline                           │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │  Dataset Loading                                  │    │
│  │  - Load candidates from JSONL                     │    │
│  │  - Build FAISS index                              │    │
│  │  - Optional: Load citation graph (NetworkX)       │    │
│  └───────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  ┌───────────────────────────────────────────────────┐    │
│  │  Analysis Workflow                                │    │
│  │                                                    │    │
│  │  Input: Target Case                               │    │
│  │     ↓                                              │    │
│  │  Step 1: External Inference                       │    │
│  │     - Retrieve top-k precedents                   │    │
│  │     - Compute TRS scores                          │    │
│  │     - Extract spans                               │    │
│  │     ↓                                              │    │
│  │  Step 2: Citation Graph (Optional)                │    │
│  │     - Detect citation links                       │    │
│  │     - Find common citations                       │    │
│  │     - Compute PageRank                            │    │
│  │     ↓                                              │    │
│  │  Step 3: Legal Reasoning                          │    │
│  │     - Multi-aspect analysis                       │    │
│  │     - Evidence tables                             │    │
│  │     - Recommendations                             │    │
│  │     ↓                                              │    │
│  │  Output: Complete Analysis                        │    │
│  └───────────────────────────────────────────────────┘    │
│                         ↓                                   │
│  ┌───────────────────────────────────────────────────┐    │
│  │  Export & Monitoring                              │    │
│  │  - Export to JSON/JSONL                           │    │
│  │  - Get statistics                                 │    │
│  │  - Batch processing                               │    │
│  └───────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Single Case Analysis

```
1. Load Dataset
   ┌──────────────────┐
   │ candidates.jsonl │
   └──────────────────┘
            ↓
   ┌──────────────────┐
   │  Build Index     │
   │  - Embeddings    │
   │  - FAISS Index   │
   └──────────────────┘
            ↓
2. Input Target Case
   ┌──────────────────┐
   │  {              │
   │   case_id,      │
   │   title,        │
   │   text,         │
   │   year,         │
   │   jurisdiction  │
   │  }              │
   └──────────────────┘
            ↓
3. External Inference
   ┌──────────────────┐
   │ Retrieved Cases  │
   │ - Case 1: TRS=.8 │
   │ - Case 2: TRS=.7 │
   │ - Case 3: TRS=.6 │
   └──────────────────┘
            ↓
4. Legal Reasoning
   ┌──────────────────┐
   │ Analysis         │
   │ - Summary        │
   │ - Aspects        │
   │ - Evidence       │
   │ - Recommendations│
   └──────────────────┘
            ↓
5. Output
   ┌──────────────────┐
   │ Complete JSON    │
   │ - External       │
   │ - Reasoning      │
   │ - Citations      │
   │ - Metadata       │
   └──────────────────┘
```

### Batch Processing

```
┌───────────────────┐
│ Multiple Targets  │
│ [case1, case2,..] │
└───────────────────┘
         ↓
    ┌────────┐
    │ Loop   │
    └────────┘
         ↓
┌─────────────────────────────────┐
│ For each case:                  │
│   1. External Inference         │
│   2. Legal Reasoning            │
│   3. Append to results          │
└─────────────────────────────────┘
         ↓
┌───────────────────┐
│ Batch Results     │
│ [result1, ...]    │
└───────────────────┘
```

---

## TRS Computation Detail

```
┌────────────────────────────────────────────────────────┐
│  Total Relevance Score (TRS) Computation               │
│                                                        │
│  Components:                                           │
│                                                        │
│  1. Similarity (S)                                     │
│     ┌────────────────────────────────────┐            │
│     │ query_emb = embed(target_text)     │            │
│     │ cand_emb = embed(candidate_text)   │            │
│     │ S = cosine_sim(query_emb, cand_emb)│            │
│     └────────────────────────────────────┘            │
│                                                        │
│  2. Context Fit (C)                                    │
│     ┌────────────────────────────────────┐            │
│     │ tfidf = TfidfVectorizer()          │            │
│     │ vectors = tfidf.fit_transform(...)  │            │
│     │ C = cosine_sim(target_v, cand_v)   │            │
│     └────────────────────────────────────┘            │
│                                                        │
│  3. Jurisdiction (J)                                   │
│     ┌────────────────────────────────────┐            │
│     │ geo_match = same_court ? 1 : 0.5   │            │
│     │ year_diff = abs(year_t - year_c)   │            │
│     │ temp_decay = exp(-year_diff/20)    │            │
│     │ J = 0.7*geo_match + 0.3*temp_decay │            │
│     └────────────────────────────────────┘            │
│                                                        │
│  4. Internal Confidence (I)                            │
│     ┌────────────────────────────────────┐            │
│     │ I = internal_confidence or 0.5     │            │
│     └────────────────────────────────────┘            │
│                                                        │
│  5. Uncertainty (U)                                    │
│     ┌────────────────────────────────────┐            │
│     │ U = |S - C| + |S - J|              │            │
│     └────────────────────────────────────┘            │
│                                                        │
│  Final TRS:                                            │
│     ┌────────────────────────────────────┐            │
│     │ TRS = w_S·S + w_C·C + w_J·J +      │            │
│     │       w_I·I - w_U·U                │            │
│     │ TRS = clip(TRS, 0, 1)              │            │
│     └────────────────────────────────────┘            │
│                                                        │
│  Default Weights:                                      │
│     w_S = 0.35, w_C = 0.20, w_J = 0.20,               │
│     w_I = 0.15, w_U = 0.10                            │
└────────────────────────────────────────────────────────┘
```

---

## Technology Stack

```
┌─────────────────────────────────────────────────────┐
│                  Application Layer                  │
│  ┌──────────────────────────────────────────────┐  │
│  │  LegalAIPipeline                             │  │
│  │  - ExternalInferenceAgent                    │  │
│  │  - LegalReasoningAgent                       │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                   Core Libraries                    │
│  ┌────────────────┐  ┌─────────────────────────┐  │
│  │ Sentence       │  │ FAISS                   │  │
│  │ Transformers   │  │ - IndexFlatIP           │  │
│  │ - Embeddings   │  │ - Similarity Search     │  │
│  └────────────────┘  └─────────────────────────┘  │
│  ┌────────────────┐  ┌─────────────────────────┐  │
│  │ Transformers   │  │ scikit-learn            │  │
│  │ - LLM Pipeline │  │ - TF-IDF                │  │
│  │ - Text Gen     │  │ - Metrics               │  │
│  └────────────────┘  └─────────────────────────┘  │
│  ┌────────────────┐  ┌─────────────────────────┐  │
│  │ NetworkX       │  │ Pandas                  │  │
│  │ - Graphs       │  │ - DataFrames            │  │
│  │ - PageRank     │  │ - Data Processing       │  │
│  └────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                  Infrastructure                     │
│  ┌────────────────┐  ┌─────────────────────────┐  │
│  │ PyTorch        │  │ NumPy                   │  │
│  │ - Deep Learning│  │ - Array Operations      │  │
│  └────────────────┘  └─────────────────────────┘  │
│  ┌────────────────┐  ┌─────────────────────────┐  │
│  │ CUDA (GPU)     │  │ CPU                     │  │
│  │ - Acceleration │  │ - Fallback              │  │
│  └────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Client Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Python   │  │ REST API │  │ Web Dashboard    │ │
│  │ Script   │  │ (Future) │  │ (Future)         │ │
│  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                  Service Layer                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  LegalAIPipeline                             │  │
│  │  - load_lecai_dataset()                      │  │
│  │  - analyze_case()                            │  │
│  │  - batch_analyze()                           │  │
│  │  - export_results()                          │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                   Agent Layer                       │
│  ┌──────────────────────┐  ┌────────────────────┐ │
│  │ ExternalInference    │  │ LegalReasoning     │ │
│  │ Agent                │  │ Agent              │ │
│  │ - build_index()      │  │ - generate_        │ │
│  │ - infer()            │  │   reasoning()      │ │
│  └──────────────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                    Data Layer                       │
│  ┌────────────────┐  ┌─────────────────────────┐  │
│  │ FAISS Index    │  │ Citation Graph          │  │
│  │ (In-Memory)    │  │ (NetworkX)              │  │
│  └────────────────┘  └─────────────────────────┘  │
│  ┌────────────────┐  ┌─────────────────────────┐  │
│  │ Candidates     │  │ Models                  │  │
│  │ (JSONL)        │  │ (HuggingFace Cache)     │  │
│  └────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Extension Points

```
┌─────────────────────────────────────────────────────┐
│             Customization & Extensions              │
│                                                     │
│  1. Custom Retrieval                                │
│     ┌──────────────────────────────────────────┐   │
│     │ Implement custom retriever:              │   │
│     │ - Dense retrieval (current)              │   │
│     │ - BM25 (add-on)                          │   │
│     │ - Hybrid (future)                        │   │
│     │ - Re-ranking (future)                    │   │
│     └──────────────────────────────────────────┘   │
│                                                     │
│  2. Custom TRS Components                           │
│     ┌──────────────────────────────────────────┐   │
│     │ Add new TRS factors:                     │   │
│     │ - Citation count                         │   │
│     │ - Court authority level                  │   │
│     │ - Legal topic alignment                  │   │
│     │ - Statutory references                   │   │
│     └──────────────────────────────────────────┘   │
│                                                     │
│  3. Custom Reasoning                                │
│     ┌──────────────────────────────────────────┐   │
│     │ Extend LegalReasoningAgent:              │   │
│     │ - Domain-specific prompts                │   │
│     │ - Fine-tuned legal LLMs                  │   │
│     │ - Multi-stage reasoning                  │   │
│     │ - Chain-of-thought                       │   │
│     └──────────────────────────────────────────┘   │
│                                                     │
│  4. Custom Embeddings                               │
│     ┌──────────────────────────────────────────┐   │
│     │ Use specialized models:                  │   │
│     │ - Legal-BERT                             │   │
│     │ - CaseLaw Embeddings                     │   │
│     │ - Domain-specific fine-tuning            │   │
│     └──────────────────────────────────────────┘   │
│                                                     │
│  5. Custom Output                                   │
│     ┌──────────────────────────────────────────┐   │
│     │ Format adapters:                         │   │
│     │ - PDF reports                            │   │
│     │ - HTML dashboards                        │   │
│     │ - LaTeX documents                        │   │
│     │ - Citation formats (Bluebook, etc.)      │   │
│     └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## Performance Optimization

```
┌─────────────────────────────────────────────────────┐
│              Optimization Strategies                │
│                                                     │
│  1. Index Optimization                              │
│     ┌──────────────────────────────────────────┐   │
│     │ - Use IndexIVFFlat for large corpora    │   │
│     │ - Quantization (IndexIVFPQ)             │   │
│     │ - GPU acceleration (FAISS-GPU)          │   │
│     └──────────────────────────────────────────┘   │
│                                                     │
│  2. Batch Processing                                │
│     ┌──────────────────────────────────────────┐   │
│     │ - Batch embedding generation            │   │
│     │ - Parallel case processing              │   │
│     │ - Cache computed embeddings             │   │
│     └──────────────────────────────────────────┘   │
│                                                     │
│  3. Model Optimization                              │
│     ┌──────────────────────────────────────────┐   │
│     │ - Smaller embedding models              │   │
│     │ - Quantized LLMs (8-bit, 4-bit)         │   │
│     │ - Distilled models                      │   │
│     └──────────────────────────────────────────┘   │
│                                                     │
│  4. Caching                                         │
│     ┌──────────────────────────────────────────┐   │
│     │ - Cache embeddings on disk              │   │
│     │ - Cache TF-IDF matrices                 │   │
│     │ - Memoize expensive computations        │   │
│     └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Status**: Complete

This architecture supports:
- ✅ Modular design
- ✅ Easy customization
- ✅ Scalable deployment
- ✅ Production readiness
