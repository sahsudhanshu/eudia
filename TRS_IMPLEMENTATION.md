# Trust Relevance Score (TRS) Implementation Guide

## Overview

The Trust Relevance Score (TRS) is a composite metric that evaluates the relevance and trustworthiness of retrieved legal cases relative to a target case. It combines multiple factors with configurable weights to provide a holistic assessment.

## TRS Formula

```
TRS = (w_S × S) + (w_C × C) + (w_J × J) + (w_I × I) - (w_U × U)
```

Where all scores are bounded to [0, 1].

## Components

### 1. Similarity Score (S)
**Weight: 0.5 (default)**

- **Computation**: Cosine similarity via normalized embeddings using FAISS IndexFlatIP
- **Method**: Sentence-BERT embeddings from `all-MiniLM-L6-v2` model
- **Range**: [0, 1], higher is better
- **Interpretation**:
  - > 0.7: High semantic similarity
  - 0.5-0.7: Moderate similarity
  - < 0.5: Low similarity

```python
# Computed via FAISS search on normalized embeddings
target_embedding = model.encode(target_text)
normalized_target = target_embedding / ||target_embedding||
similarity_score = faiss_index.search(normalized_target, k=1)[0]
```

### 2. Context Fit Score (C)
**Weight: 0.2 (default)**

- **Computation**: TF-IDF cosine similarity
- **Method**: Scikit-learn TfidfVectorizer with 500 max features
- **Range**: [0, 1], higher is better
- **Fallback**: Word overlap ratio if TF-IDF fails
- **Interpretation**:
  - > 0.6: High contextual relevance
  - 0.3-0.6: Moderate context alignment
  - < 0.3: Low contextual overlap

```python
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_matrix = vectorizer.fit_transform([target_text, candidate_text])
context_fit = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
```

### 3. Jurisdiction Score (J)
**Weight: 0.1 (default)**

- **Computation**: Combination of geographic and temporal alignment
- **Formula**: `0.7 × jurisdiction_match + 0.3 × temporal_proximity`
- **Range**: [0, 1], higher is better
- **Components**:
  - **Jurisdiction Match**: 1.0 if exact match, 0.5 otherwise
  - **Temporal Proximity**: `exp(-year_diff / 20)` (20-year half-life)

```python
jur_score = 1.0 if target_jur == cand_jur else 0.5
temporal_score = exp(-abs(target_year - cand_year) / 20.0)
J = 0.7 × jur_score + 0.3 × temporal_score
```

**Temporal Decay Examples**:
- Same year: 1.0
- 5 years apart: 0.778
- 10 years apart: 0.606
- 20 years apart: 0.368
- 50 years apart: 0.082

### 4. Internal Confidence (I)
**Weight: 0.15 (default)**

- **Source**: Optional parameter from internal model predictions
- **Range**: [0, 1], higher indicates more confidence
- **Default**: 0.0 if not provided
- **Purpose**: Incorporate confidence from domain-specific models

```python
I = internal_confidence if internal_confidence is not None else 0.0
```

### 5. Uncertainty (U)
**Weight: 0.05 (default)**

- **Computation**: Variance between Similarity and Context Fit
- **Formula**: `min((S - C)², 1.0)`
- **Range**: [0, 1], **lower is better** (subtracted in TRS)
- **Interpretation**:
  - High variance → High uncertainty → Lower TRS
  - Low variance → Low uncertainty → Higher TRS

```python
variance = (similarity - context_fit) ** 2
uncertainty = min(variance, 1.0)
```

## Default Weights

```python
trs_weights = {
    "w_S": 0.5,   # Similarity - most important
    "w_C": 0.2,   # Context fit - secondary
    "w_J": 0.1,   # Jurisdiction - tertiary
    "w_I": 0.15,  # Internal confidence - optional boost
    "w_U": 0.05   # Uncertainty - small penalty
}
```

**Weight Rationale**:
- **S (0.5)**: Primary signal of relevance; semantic similarity is most direct indicator
- **C (0.2)**: Secondary validation; ensures contextual alignment
- **J (0.1)**: Domain-specific factor; legal precedent value varies by jurisdiction
- **I (0.15)**: Trust in internal models when available
- **U (0.05)**: Small penalty for inconsistent signals

## Implementation Details

### Clipping to [0, 1]

All individual scores AND the final TRS are clipped:

```python
trs = (w_S * S) + (w_C * C) + (w_J * J) + (w_I * I) - (w_U * U)
trs = np.clip(trs, 0, 1)  # Force into valid range
```

### Return Factors Option

When `return_factors=True`, the method returns a detailed breakdown:

```python
{
    "score": 0.747,
    "factors": {
        "similarity": 0.896,
        "context_fit": 0.463,
        "jurisdiction_score": 0.958,
        "internal_confidence": 0.8,
        "uncertainty": 0.187
    },
    "weights": {
        "w_S": 0.5,
        "w_C": 0.2,
        "w_J": 0.1,
        "w_I": 0.15,
        "w_U": 0.05
    }
}
```

## Usage Examples

### Basic Usage

```python
from lexai.agents import ExternalInferenceAgent

agent = ExternalInferenceAgent()
agent.build_index(candidates)

result = agent.infer(
    target=target_case,
    top_k=5,
    internal_confidence=0.8
)

# Access TRS
for case in result['retrieved_cases']:
    print(f"{case['title']}: TRS = {case['trs']:.3f}")
```

### Custom Weights

```python
# Emphasize similarity, reduce jurisdiction importance
custom_weights = {
    "w_S": 0.6,   # Increase similarity weight
    "w_C": 0.25,  # Increase context weight
    "w_J": 0.05,  # Reduce jurisdiction weight
    "w_I": 0.1,   # Adjust internal confidence
    "w_U": 0.0    # Ignore uncertainty
}

agent = ExternalInferenceAgent(trs_weights=custom_weights)
```

### With Factor Breakdown

```python
# Get detailed factor analysis
agent = ExternalInferenceAgent()
agent.build_index(candidates)

result = agent.infer(
    target=target_case,
    top_k=3,
    internal_confidence=0.8,
    return_factors=True  # Enable detailed breakdown
)

# Access detailed TRS
for case in result['retrieved_cases']:
    trs_details = case['trs']
    print(f"Score: {trs_details['score']:.3f}")
    print(f"Factors: {trs_details['factors']}")
```

## TRS Interpretation

### Score Ranges

| TRS Range | Interpretation | Recommendation |
|-----------|----------------|----------------|
| 0.8 - 1.0 | Highly Relevant | Strong precedent; high confidence |
| 0.6 - 0.8 | Moderately Relevant | Good precedent; use with context |
| 0.4 - 0.6 | Somewhat Relevant | Weak precedent; verify carefully |
| 0.2 - 0.4 | Marginally Relevant | Limited value; use cautiously |
| 0.0 - 0.2 | Not Relevant | Likely not applicable |

### Example Scenarios

#### Scenario 1: Perfect Match
```
S = 0.95 (high semantic similarity)
C = 0.85 (strong context fit)
J = 0.95 (same jurisdiction, recent case)
I = 0.90 (high internal confidence)
U = 0.01 (low uncertainty)

TRS = 0.5×0.95 + 0.2×0.85 + 0.1×0.95 + 0.15×0.90 - 0.05×0.01
    = 0.475 + 0.17 + 0.095 + 0.135 - 0.0005
    = 0.8745 ≈ 0.87 (High relevance)
```

#### Scenario 2: Semantic Match, Weak Context
```
S = 0.80 (good similarity)
C = 0.30 (weak context)
J = 0.70 (different jurisdiction)
I = 0.50 (moderate confidence)
U = 0.25 (higher uncertainty due to S-C variance)

TRS = 0.5×0.80 + 0.2×0.30 + 0.1×0.70 + 0.15×0.50 - 0.05×0.25
    = 0.40 + 0.06 + 0.07 + 0.075 - 0.0125
    = 0.5925 ≈ 0.59 (Moderate relevance with caution)
```

#### Scenario 3: Old Case, Different Jurisdiction
```
S = 0.70 (moderate similarity)
C = 0.65 (decent context)
J = 0.35 (different jurisdiction, 50 years old)
I = 0.00 (no internal confidence)
U = 0.00 (low uncertainty)

TRS = 0.5×0.70 + 0.2×0.65 + 0.1×0.35 + 0.15×0.00 - 0.05×0.00
    = 0.35 + 0.13 + 0.035 + 0.0 - 0.0
    = 0.515 ≈ 0.52 (Marginal relevance due to jurisdiction/time)
```

## Validation & Testing

### Automated Checks

The implementation includes built-in validation:

```python
# All individual scores are clipped
S = np.clip(similarity, 0, 1)
C = np.clip(context_fit, 0, 1)
J = np.clip(jurisdiction_score, 0, 1)
I = np.clip(internal_confidence, 0, 1)
U = np.clip(uncertainty, 0, 1)

# Final TRS is clipped
trs = np.clip(trs, 0, 1)
```

### Test Suite

Run tests to verify TRS bounds:

```bash
python lexai/agents/external_inference_agent.py --test
```

Tests verify:
- ✓ All TRS scores ∈ [0, 1]
- ✓ All component scores ∈ [0, 1]
- ✓ Overall coherence score ∈ [0, 1]

## Advanced Customization

### Domain-Specific Weights

Different legal domains may benefit from different weight configurations:

#### Constitutional Law
```python
# Emphasize jurisdiction and precedent
constitutional_weights = {
    "w_S": 0.4,
    "w_C": 0.15,
    "w_J": 0.25,  # Higher jurisdiction weight
    "w_I": 0.15,
    "w_U": 0.05
}
```

#### Contract Law
```python
# Emphasize semantic and contextual similarity
contract_weights = {
    "w_S": 0.5,
    "w_C": 0.3,   # Higher context weight
    "w_J": 0.05,  # Lower jurisdiction weight
    "w_I": 0.1,
    "w_U": 0.05
}
```

#### Criminal Law
```python
# Balanced approach with confidence
criminal_weights = {
    "w_S": 0.45,
    "w_C": 0.2,
    "w_J": 0.15,
    "w_I": 0.2,   # Higher internal confidence weight
    "w_U": 0.0
}
```

### Custom TRS Formula

Extend the agent for custom TRS computation:

```python
class CustomInferenceAgent(ExternalInferenceAgent):
    def _compute_trs(self, S, C, J, I, U, return_factors=False):
        # Custom formula: exponential weighting
        trs = (S**2 * 0.5 + C * 0.2 + J * 0.1 + I * 0.15) / (1 + U)
        trs = np.clip(trs, 0, 1)
        
        if return_factors:
            return {"score": trs, "factors": {...}}
        return trs
```

## Performance Considerations

### Computational Complexity

- **Similarity (S)**: O(1) - FAISS lookup
- **Context Fit (C)**: O(n×m) - TF-IDF vectorization
- **Jurisdiction (J)**: O(1) - Simple comparison
- **Uncertainty (U)**: O(1) - Arithmetic
- **Total per candidate**: O(n×m) dominated by TF-IDF

### Optimization Tips

1. **Cache TF-IDF vectorizers** for repeated queries
2. **Batch encode targets** when processing multiple targets
3. **Pre-compute jurisdiction scores** for static metadata
4. **Use sparse TF-IDF** for very long documents

## References

- Sentence-BERT: [SBERT.net](https://www.sbert.net/)
- FAISS: [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- TF-IDF: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
