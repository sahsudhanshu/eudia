"""
Integration example: Using InLegalBERT External Agent with LecAI Dataset

This script demonstrates how to use the InLegalBERTExternalAgent with the actual
LecAI baseline dataset, including citation graph loading and hybrid retrieval.
"""

import os
import json
from lexai.agents import InLegalBERTExternalAgent


def load_lecai_full_dataset_example():
    """Example: Load full LecAI dataset with citation graph."""
    
    print("="*70)
    print("INLEGALBERT + LECAI INTEGRATION EXAMPLE")
    print("="*70)
    
    # Initialize agent with custom weights
    print("\n1. Initializing agent with domain-specific weights...")
    agent = InLegalBERTExternalAgent(
        model_name="law-ai/InLegalBERT",
        use_citation_weight=0.5,  # Higher for citation-rich queries
        use_semantic_weight=0.5,  # Balanced hybrid approach
        device="cuda"  # Use GPU if available
    )
    
    # Paths to LecAI dataset
    base_path = "/home/anand/eudia/LecNet_baseline_models_code_and_README/LecNet_baseline_models_code_and_README"
    
    # Option 1: Load with case mapping only (no full text)
    print("\n2. Loading LecAI dataset (first 500 cases for demo)...")
    num_cases = agent.load_dataset(
        case_mapping_path=os.path.join(base_path, "file_Case_id_and_name.csv"),
        max_cases=500
    )
    
    print(f"   Loaded {num_cases} cases")
    
    # Option 2: Load with citation graph (if edge list available)
    # Uncomment if you have edge list file:
    # edge_list_path = os.path.join(base_path, "comp1_25/dataset/ogbl_newscprecedent/split/time/train_edges.csv")
    # if os.path.exists(edge_list_path):
    #     agent._load_edge_list(edge_list_path)
    
    # Compute embeddings
    print("\n3. Computing InLegalBERT embeddings...")
    agent.compute_all_embeddings(batch_size=8)
    
    # Example query cases
    query_cases = list(agent.cases.keys())[:5]
    
    print("\n4. Running retrieval on sample queries...")
    results = []
    
    for i, query_id in enumerate(query_cases, 1):
        print(f"\n   Query {i}/{len(query_cases)}: {agent.cases[query_id]['case_name'][:60]}...")
        
        # Retrieve similar cases
        retrieved = agent.retrieve_similar_cases(
            query_case_id=query_id,
            top_k=5,
            use_citations=agent.citation_graph.number_of_edges() > 0,
            use_semantic=True
        )
        
        # Generate reasoning output
        output = agent.generate_reasoning_output(
            query_case_id=query_id,
            retrieved_cases=retrieved
        )
        
        results.append(output)
        
        # Display top result
        if retrieved:
            top = retrieved[0]
            print(f"      Top match: {top['case_name'][:50]}... (score: {top['similarity_score']:.3f})")
    
    # Save results
    print("\n5. Saving results...")
    output_path = "lexai/data/inlegalbert_retrieval_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   Results saved to: {output_path}")
    
    # Show statistics
    print("\n6. Agent Statistics:")
    stats = agent.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n" + "="*70)
    print("INTEGRATION EXAMPLE COMPLETE")
    print("="*70)
    
    return results, agent


def hybrid_retrieval_example():
    """Example: Hybrid retrieval with citation graph."""
    
    print("\n" + "="*70)
    print("HYBRID RETRIEVAL EXAMPLE (Citation + Semantic)")
    print("="*70)
    
    # Initialize agent
    agent = InLegalBERTExternalAgent(
        model_name="law-ai/InLegalBERT",
        use_citation_weight=0.6,  # Prioritize citation network
        use_semantic_weight=0.4,
        device="cuda"
    )
    
    # Load dataset with citation graph
    base_path = "/home/anand/eudia/LecNet_baseline_models_code_and_README/LecNet_baseline_models_code_and_README"
    
    agent.load_dataset(
        case_mapping_path=os.path.join(base_path, "file_Case_id_and_name.csv"),
        max_cases=200
    )
    
    # Manually add some citation relationships (for demo)
    # In production, this would come from edge list file
    print("\nBuilding citation graph (demo with synthetic citations)...")
    case_ids = list(agent.cases.keys())
    
    # Add some demo citations
    for i in range(min(50, len(case_ids)-1)):
        agent.citation_graph.add_edge(case_ids[i], case_ids[i+1])
    
    print(f"Citation graph: {agent.citation_graph.number_of_edges()} edges")
    
    # Compute embeddings
    agent.compute_all_embeddings(batch_size=8)
    
    # Query with hybrid retrieval
    query_id = case_ids[10]
    print(f"\nQuery: {agent.cases[query_id]['case_name']}")
    
    print("\n--- Pure Semantic Retrieval ---")
    semantic_only = agent.retrieve_similar_cases(
        query_case_id=query_id,
        top_k=3,
        use_citations=False,
        use_semantic=True
    )
    
    for r in semantic_only:
        print(f"  {r['rank']}. {r['case_name'][:50]}... (sem: {r['semantic_score']:.3f})")
    
    print("\n--- Hybrid Retrieval (Citation + Semantic) ---")
    hybrid = agent.retrieve_similar_cases(
        query_case_id=query_id,
        top_k=3,
        use_citations=True,
        use_semantic=True
    )
    
    for r in hybrid:
        print(f"  {r['rank']}. {r['case_name'][:50]}...")
        print(f"       Hybrid: {r['similarity_score']:.3f} (sem: {r['semantic_score']:.3f}, cit: {r['citation_score']:.3f})")
    
    print("\n" + "="*70)
    
    return agent


def compare_with_sentence_transformers():
    """Compare InLegalBERT with generic Sentence Transformers."""
    
    print("\n" + "="*70)
    print("MODEL COMPARISON: InLegalBERT vs Generic SBERT")
    print("="*70)
    
    from lexai.agents import ExternalInferenceAgent
    
    # Load same dataset with both agents
    base_path = "/home/anand/eudia/LecNet_baseline_models_code_and_README/LecNet_baseline_models_code_and_README"
    
    # InLegalBERT agent
    print("\n1. InLegalBERT Agent")
    inlegal_agent = InLegalBERTExternalAgent(
        model_name="law-ai/InLegalBERT",
        device="cuda"
    )
    
    # Generic SBERT agent
    print("\n2. Generic SBERT Agent")
    sbert_agent = ExternalInferenceAgent(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda"
    )
    
    # Load small subset for comparison
    print("\n3. Loading dataset (50 cases)...")
    inlegal_agent.load_dataset(
        case_mapping_path=os.path.join(base_path, "file_Case_id_and_name.csv"),
        max_cases=50
    )
    
    # Convert to format for SBERT agent
    candidates = [
        {
            'case_id': c['case_id'],
            'title': c['case_name'],
            'text': c['case_name'],  # Using case name as text for demo
            'year': c.get('year'),
            'jurisdiction': 'Supreme Court of India'
        }
        for c in inlegal_agent.cases.values()
    ]
    
    sbert_agent.build_index(candidates)
    
    # Compute embeddings
    print("\n4. Computing embeddings...")
    inlegal_agent.compute_all_embeddings(batch_size=8)
    
    # Compare retrieval results
    query_id = list(inlegal_agent.cases.keys())[5]
    query_case = inlegal_agent.cases[query_id]
    
    print(f"\n5. Query: {query_case['case_name'][:60]}...")
    
    # InLegalBERT retrieval
    print("\n--- InLegalBERT Results ---")
    inlegal_results = inlegal_agent.retrieve_similar_cases(
        query_case_id=query_id,
        top_k=5,
        use_semantic=True,
        use_citations=False
    )
    
    for r in inlegal_results:
        print(f"  {r['rank']}. {r['case_name'][:50]}... ({r['similarity_score']:.3f})")
    
    # SBERT retrieval
    print("\n--- Generic SBERT Results ---")
    sbert_target = {
        'case_id': query_id,
        'title': query_case['case_name'],
        'text': query_case['case_name'],
        'year': query_case.get('year'),
        'jurisdiction': 'Supreme Court of India'
    }
    
    sbert_results = sbert_agent.infer(target=sbert_target, top_k=5)
    
    for i, r in enumerate(sbert_results['retrieved_cases'], 1):
        print(f"  {i}. {r['title'][:50]}... ({r['similarity_score']:.3f})")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("Note: InLegalBERT is fine-tuned on legal domain, expect better")
    print("      performance on legal text compared to generic models.")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if "--full" in sys.argv:
        # Full dataset example
        results, agent = load_lecai_full_dataset_example()
    
    elif "--hybrid" in sys.argv:
        # Hybrid retrieval example
        agent = hybrid_retrieval_example()
    
    elif "--compare" in sys.argv:
        # Model comparison
        compare_with_sentence_transformers()
    
    else:
        print("Usage:")
        print("  python lexai/inlegalbert_integration.py --full       # Full LecAI dataset")
        print("  python lexai/inlegalbert_integration.py --hybrid     # Hybrid retrieval demo")
        print("  python lexai/inlegalbert_integration.py --compare    # Compare models")
        print("\nRunning all examples...")
        
        # Run all
        results, agent = load_lecai_full_dataset_example()
        hybrid_retrieval_example()
        compare_with_sentence_transformers()
