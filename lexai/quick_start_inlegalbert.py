#!/usr/bin/env python3
"""
Quick Start: InLegalBERT External Agent

Run this script to see a complete working example.
"""

from lexai.agents import InLegalBERTExternalAgent
import json

# ============================================================================
# STEP 1: Initialize Agent
# ============================================================================
print("Step 1: Initializing InLegalBERT External Agent...")

agent = InLegalBERTExternalAgent(
    model_name="law-ai/InLegalBERT",
    use_citation_weight=0.4,  # Citation graph weight
    use_semantic_weight=0.6,  # Semantic similarity weight
    device="cuda"  # or "cpu"
)

# ============================================================================
# STEP 2: Load LecAI Dataset
# ============================================================================
print("\nStep 2: Loading LecAI dataset...")

# Path to your LecAI dataset
LECAI_BASE = "/home/anand/eudia/LecNet_baseline_models_code_and_README/LecNet_baseline_models_code_and_README"
case_mapping = f"{LECAI_BASE}/file_Case_id_and_name.csv"

num_cases = agent.load_dataset(
    case_mapping_path=case_mapping,
    max_cases=100  # Limit for quick demo
)

print(f"Loaded {num_cases} cases")

# ============================================================================
# STEP 3: Compute Embeddings
# ============================================================================
print("\nStep 3: Computing InLegalBERT embeddings...")

agent.compute_all_embeddings(batch_size=8)
print("Embeddings computed successfully")

# ============================================================================
# STEP 4: Query Similar Cases
# ============================================================================
print("\nStep 4: Retrieving similar cases...")

# Select a query case
query_case_id = list(agent.cases.keys())[0]
query_case = agent.cases[query_case_id]

print(f"\nQuery Case: {query_case['case_name']}")
print(f"Query Year: {query_case.get('year', 'N/A')}")

# Retrieve similar cases
results = agent.retrieve_similar_cases(
    query_case_id=query_case_id,
    top_k=5,
    use_citations=False,  # No citations loaded in quick demo
    use_semantic=True
)

# ============================================================================
# STEP 5: Generate Reasoning Output
# ============================================================================
print("\nStep 5: Generating structured output...")

output = agent.generate_reasoning_output(
    query_case_id=query_case_id,
    retrieved_cases=results
)

# ============================================================================
# STEP 6: Display Results
# ============================================================================
print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\nQuery: {output['query_case_name']}\n")

print("Top Similar Cases:")
for ref in output['top_references']:
    print(f"\n{ref['case_id']}")
    print(f"  Name: {ref['case_name']}")
    print(f"  Year: {ref['year']}")
    print(f"  Similarity: {ref['similarity_score']:.3f}")
    print(f"  Reasoning: {ref['reasoning'][:150]}...")

print(f"\nOverall Summary:")
print(f"  {output['overall_legal_context_summary'][:200]}...")

# ============================================================================
# STEP 7: Save Results
# ============================================================================
print("\n" + "="*70)
print("Saving results to 'quick_start_output.json'...")

with open('quick_start_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Done! ✓")

# ============================================================================
# STATISTICS
# ============================================================================
print("\n" + "="*70)
print("AGENT STATISTICS")
print("="*70)

stats = agent.get_statistics()
print(f"Total cases: {stats['total_cases']}")
print(f"Embeddings computed: {stats['embeddings_computed']}")
print(f"Model: {stats['model']}")
print(f"Device: {stats['device']}")
print(f"Weights: Citation={stats['weights']['citation']}, Semantic={stats['weights']['semantic']}")

print("\n" + "="*70)
print("QUICK START COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Load more cases: agent.load_dataset(..., max_cases=1000)")
print("  2. Add citation graph: agent._load_edge_list('edges.csv')")
print("  3. Enable hybrid retrieval: retrieve_similar_cases(..., use_citations=True)")
print("  4. Integrate with LegalReasoningAgent for deeper analysis")
