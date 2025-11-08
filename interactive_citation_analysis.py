"""
Interactive Citation Analysis Pipeline

This script allows you to:
1. Load the citation graph (from citation_graph_builder.py output)
2. Select 2 nodes from the graph
3. Calculate TRS score using the RL model
4. Generate RL-based reasoning summary explaining the TRS score
5. Get internal coherence analysis from the multi-model agent
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lexai.reasoning_pipeline import LegalReasoningPipeline, create_demo_graph
from lexai.embedding_interrogator import EmbeddingInterrogator
from agents.multi_model_internal_coherence_agent_light_pdf import run_internal_coherence_agent


def load_citation_graph(graph_path: str = "lexai/data/processed/citation_graph.json"):
    """Load the citation graph built by citation_graph_builder.py"""
    print(f"\n{'='*80}")
    print("LOADING CITATION GRAPH")
    print(f"{'='*80}")
    
    with open(graph_path, 'r') as f:
        graph = json.load(f)
    
    print(f"\n✓ Loaded graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
    
    return graph


def display_available_nodes(graph: Dict):
    """Display all nodes with their citation counts"""
    print(f"\n{'='*80}")
    print("AVAILABLE NODES (Cases in Citation Graph)")
    print(f"{'='*80}\n")
    
    # Calculate citation counts
    incoming_citations = {}
    outgoing_citations = {}
    
    for edge in graph['edges']:
        source = edge['source']
        target = edge['target']
        
        outgoing_citations[source] = outgoing_citations.get(source, 0) + 1
        incoming_citations[target] = incoming_citations.get(target, 0) + 1
    
    # Display nodes
    nodes_with_citations = []
    for node in graph['nodes']:
        node_id = node['id']
        incoming = incoming_citations.get(node_id, 0)
        outgoing = outgoing_citations.get(node_id, 0)
        
        # Only show nodes that have actual citation relationships
        if incoming > 0 or outgoing > 0:
            nodes_with_citations.append({
                'id': node_id,
                'label': node.get('label', node_id),
                'year': node.get('year', 'N/A'),
                'incoming': incoming,
                'outgoing': outgoing,
                'total': incoming + outgoing
            })
    
    # Sort by total citations
    nodes_with_citations.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"{'#':<4} {'Case ID':<30} {'Year':<8} {'Cites →':<10} {'← Cited':<10} {'Total':<8}")
    print("-" * 80)
    
    for i, node in enumerate(nodes_with_citations, 1):
        year_str = str(node['year']) if node['year'] else 'N/A'
        print(f"{i:<4} {node['label']:<30} {year_str:<8} {node['outgoing']:<10} "
              f"{node['incoming']:<10} {node['total']:<8}")
    
    return nodes_with_citations


def select_nodes_interactive(nodes: List[Dict]) -> Tuple[str, str]:
    """Interactive node selection"""
    print(f"\n{'='*80}")
    print("SELECT TWO NODES FOR ANALYSIS")
    print(f"{'='*80}\n")
    
    while True:
        try:
            print("Enter the number of the FIRST node (citing case): ", end='')
            first_idx = int(input().strip()) - 1
            
            if 0 <= first_idx < len(nodes):
                first_node = nodes[first_idx]['id']
                print(f"✓ Selected: {nodes[first_idx]['label']} ({nodes[first_idx]['year']})")
                break
            else:
                print(f"❌ Invalid number. Please enter 1-{len(nodes)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    while True:
        try:
            print("\nEnter the number of the SECOND node (cited case): ", end='')
            second_idx = int(input().strip()) - 1
            
            if 0 <= second_idx < len(nodes):
                second_node = nodes[second_idx]['id']
                print(f"✓ Selected: {nodes[second_idx]['label']} ({nodes[second_idx]['year']})")
                break
            else:
                print(f"❌ Invalid number. Please enter 1-{len(nodes)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    return first_node, second_node


def select_nodes_auto(nodes: List[Dict], source_idx: int = 0, target_idx: int = 1) -> Tuple[str, str]:
    """Automatic node selection for testing"""
    if source_idx >= len(nodes) or target_idx >= len(nodes):
        source_idx, target_idx = 0, min(1, len(nodes) - 1)
    
    source = nodes[source_idx]['id']
    target = nodes[target_idx]['id']
    
    print(f"\n{'='*80}")
    print("AUTO-SELECTED NODES FOR ANALYSIS")
    print(f"{'='*80}")
    print(f"\n✓ Source (citing): {nodes[source_idx]['label']} ({nodes[source_idx]['year']})")
    print(f"✓ Target (cited):  {nodes[target_idx]['label']} ({nodes[target_idx]['year']})")
    
    return source, target


def calculate_trs_and_reasoning(
    source: str,
    target: str,
    graph: Dict,
    output_dir: Path = Path("results/interactive_analysis")
) -> Dict:
    """
    Calculate TRS score and generate RL-based reasoning summary
    
    Returns:
        {
            'trs_score': float,
            'similarity_score': float,
            'embedding_weights': dict,
            'reasoning_summary': str,
            'detailed_analysis': dict
        }
    """
    print(f"\n{'='*80}")
    print("STEP 1: TRS SCORE CALCULATION (RL Model)")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert graph to format expected by reasoning pipeline
    # For this demo, we'll use mock embeddings since we don't have real document embeddings
    print("Preparing graph data for RL model...")
    
    # Find the edge between source and target
    edge_data = None
    for edge in graph['edges']:
        if edge['source'] == source and edge['target'] == target:
            edge_data = edge
            break
    
    if not edge_data:
        print(f"⚠ Warning: No direct edge found from {source} to {target}")
        print("Creating synthetic edge for analysis...")
        edge_data = {
            'source': source,
            'target': target,
            'citation_type': 'SUPPORTS'
        }
    
    # Create demo graph for the reasoning pipeline
    # This creates mock embeddings and TRS scores
    demo_graph_path = create_demo_graph(output_dir / "temp_graph.json")
    
    # Initialize reasoning pipeline
    pipeline = LegalReasoningPipeline(
        graph_file=str(demo_graph_path),
        agent_path="results/mock_rl_demo/final_agent.pt",  # Trained RL agent
        model_name=None,  # Use template-based reasoning
        output_dir=output_dir
    )
    
    print("\n✓ RL model loaded")
    print(f"✓ Using trained agent from: results/mock_rl_demo/final_agent.pt")
    
    # Analyze citation
    print(f"\nAnalyzing citation: {source} → {target}")
    result = pipeline.analyze_citation(source, target)
    
    print(f"\n{'='*80}")
    print("TRS SCORE RESULTS")
    print(f"{'='*80}")
    print(f"\n📊 TRS Score:        {result.trustworthiness_score:.3f}")
    print(f"📊 Similarity:       {result.similarity_score:.3f}")
    print(f"📊 Confidence:       {result.confidence_score:.3f}")
    print(f"📊 Citation Type:    {result.citation_type}")
    print(f"\n🎯 RL-Selected Weights:")
    print(f"   - Internal Coherence: {result.embedding_weights['w_internal']:.2f}")
    print(f"   - External Network:   {result.embedding_weights['w_external']:.2f}")
    
    return {
        'trs_score': result.trustworthiness_score,
        'similarity_score': result.similarity_score,
        'confidence_score': result.confidence_score,
        'citation_type': result.citation_type,
        'embedding_weights': result.embedding_weights,
        'reasoning_summary': result.legal_reasoning,
        'consistency_analysis': result.consistency_analysis,
        'reliability_assessment': result.reliability_assessment,
        'potential_gaps': result.potential_gaps,
        'full_result': result
    }


def generate_rl_reasoning_summary(trs_result: Dict, output_dir: Path) -> str:
    """
    Generate detailed RL-based reasoning summary explaining WHY the TRS score is what it is
    """
    print(f"\n{'='*80}")
    print("STEP 2: RL REASONING SUMMARY GENERATION")
    print(f"{'='*80}\n")
    
    # The reasoning pipeline already generated this in trs_result
    summary = trs_result['reasoning_summary']
    
    print("✓ Generated RL-based reasoning summary")
    print("\n" + "="*80)
    print("REASONING SUMMARY PREVIEW")
    print("="*80)
    print(summary[:500] + "...\n")
    
    # Save full summary
    summary_path = output_dir / "rl_reasoning_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RL-BASED REASONING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(summary)
        f.write("\n\n" + "="*80 + "\n")
        f.write("CONSISTENCY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(trs_result['consistency_analysis'])
        f.write("\n\n" + "="*80 + "\n")
        f.write("RELIABILITY ASSESSMENT\n")
        f.write("="*80 + "\n\n")
        f.write(trs_result['reliability_assessment'])
    
    print(f"✓ Full summary saved to: {summary_path}")
    
    return summary


def get_internal_coherence_analysis(
    source: str,
    target: str,
    graph: Dict,
    output_dir: Path
) -> Dict:
    """
    Get internal coherence analysis from the multi-model agent
    """
    print(f"\n{'='*80}")
    print("STEP 3: INTERNAL COHERENCE ANALYSIS")
    print(f"{'='*80}\n")
    
    # Find the document paths for source and target
    # For this demo, we'll use the sample documents we created
    doc_dir = Path("lexai/data/sample_legal_docs")
    
    source_doc = None
    target_doc = None
    
    # Try to find matching documents
    for node in graph['nodes']:
        if node['id'] == source:
            source_label = node.get('label', source)
            # Try to find matching file
            for doc_file in doc_dir.glob("*.txt"):
                if source_label.lower() in doc_file.stem.lower():
                    source_doc = str(doc_file)
                    break
        
        if node['id'] == target:
            target_label = node.get('label', target)
            for doc_file in doc_dir.glob("*.txt"):
                if target_label.lower() in doc_file.stem.lower():
                    target_doc = str(doc_file)
                    break
    
    if not source_doc or not target_doc:
        print("⚠ Warning: Could not find document files for analysis")
        print("Using sample document for demonstration...")
        
        # Use any available documents
        available_docs = list(doc_dir.glob("*.txt"))
        if len(available_docs) >= 2:
            source_doc = str(available_docs[0])
            target_doc = str(available_docs[1])
        else:
            print("❌ No sample documents found. Skipping internal coherence analysis.")
            return {}
    
    print(f"📄 Source document: {Path(source_doc).name}")
    print(f"📄 Target document: {Path(target_doc).name}")
    
    # Run internal coherence analysis
    print("\nRunning multi-model internal coherence analysis...")
    print("This may take a few moments...\n")
    
    try:
        # Analyze source document
        print(f"Analyzing {source}...")
        source_result = run_internal_coherence_agent(source_doc)
        
        print(f"Analyzing {target}...")
        target_result = run_internal_coherence_agent(target_doc)
        
        print("\n✓ Internal coherence analysis complete")
        
        # Display summary
        print(f"\n{'='*80}")
        print("INTERNAL COHERENCE RESULTS")
        print(f"{'='*80}\n")
        
        print(f"📊 {source} Coherence Scores:")
        for key, value in source_result.get('coherence_scores', {}).items():
            print(f"   - {key}: {value}")
        
        print(f"\n📊 {target} Coherence Scores:")
        for key, value in target_result.get('coherence_scores', {}).items():
            print(f"   - {key}: {value}")
        
        # Save results
        results = {
            'source': {
                'case_id': source,
                'document': source_doc,
                'analysis': source_result
            },
            'target': {
                'case_id': target,
                'document': target_doc,
                'analysis': target_result
            }
        }
        
        results_path = output_dir / "internal_coherence_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during internal coherence analysis: {e}")
        return {}


def generate_final_report(
    source: str,
    target: str,
    trs_result: Dict,
    internal_coherence: Dict,
    output_dir: Path
):
    """Generate comprehensive final report combining all analyses"""
    print(f"\n{'='*80}")
    print("GENERATING FINAL COMPREHENSIVE REPORT")
    print(f"{'='*80}\n")
    
    report_path = output_dir / "comprehensive_analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Citation Analysis Report\n\n")
        f.write(f"**Citation:** {source} → {target}\n\n")
        f.write("---\n\n")
        
        # Section 1: TRS Score and RL Analysis
        f.write("## 1. TRS Score and RL-Based Analysis\n\n")
        f.write(f"**TRS Score:** {trs_result['trs_score']:.3f}\n\n")
        f.write(f"**Similarity Score:** {trs_result['similarity_score']:.3f}\n\n")
        f.write(f"**Confidence:** {trs_result['confidence_score']:.3f}\n\n")
        f.write(f"**Citation Type:** {trs_result['citation_type']}\n\n")
        
        f.write("### RL-Selected Embedding Weights\n\n")
        f.write(f"- Internal Coherence: {trs_result['embedding_weights']['w_internal']:.2f}\n")
        f.write(f"- External Network: {trs_result['embedding_weights']['w_external']:.2f}\n\n")
        
        f.write("### Legal Reasoning Summary\n\n")
        f.write(trs_result['reasoning_summary'])
        f.write("\n\n---\n\n")
        
        # Section 2: Consistency Analysis
        f.write("## 2. Consistency Analysis\n\n")
        f.write(trs_result['consistency_analysis'])
        f.write("\n\n---\n\n")
        
        # Section 3: Reliability Assessment
        f.write("## 3. Reliability Assessment\n\n")
        f.write(trs_result['reliability_assessment'])
        f.write("\n\n---\n\n")
        
        # Section 4: Internal Coherence Analysis
        f.write("## 4. Internal Coherence Analysis\n\n")
        
        if internal_coherence:
            f.write(f"### {source} (Citing Case)\n\n")
            source_scores = internal_coherence.get('source', {}).get('analysis', {}).get('coherence_scores', {})
            for key, value in source_scores.items():
                f.write(f"- **{key}:** {value}\n")
            
            f.write(f"\n### {target} (Cited Case)\n\n")
            target_scores = internal_coherence.get('target', {}).get('analysis', {}).get('coherence_scores', {})
            for key, value in target_scores.items():
                f.write(f"- **{key}:** {value}\n")
        else:
            f.write("*Internal coherence analysis not available*\n")
        
        f.write("\n\n---\n\n")
        
        # Section 5: Potential Issues and Gaps
        f.write("## 5. Potential Issues and Gaps\n\n")
        for gap in trs_result.get('potential_gaps', []):
            f.write(f"- {gap}\n")
        
        f.write("\n\n---\n\n")
        f.write(f"*Report generated on {trs_result['full_result'].timestamp}*\n")
    
    print(f"✓ Comprehensive report saved to: {report_path}")
    
    # Display report location
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE - OUTPUT FILES")
    print(f"{'='*80}\n")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Comprehensive report: {report_path}")
    print(f"📄 RL reasoning summary: {output_dir / 'rl_reasoning_summary.txt'}")
    if internal_coherence:
        print(f"📄 Internal coherence: {output_dir / 'internal_coherence_analysis.json'}")
    print()


def main(interactive: bool = False):
    """
    Main pipeline function
    
    Args:
        interactive: If True, allows user to select nodes interactively.
                    If False, auto-selects nodes for demo.
    """
    print("\n" + "="*80)
    print("INTERACTIVE CITATION ANALYSIS PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("1. Load the citation graph")
    print("2. Select 2 nodes (cases)")
    print("3. Calculate TRS score using RL model")
    print("4. Generate RL-based reasoning summary")
    print("5. Analyze internal coherence")
    print("6. Generate comprehensive report")
    print("="*80 + "\n")
    
    # Step 1: Load graph
    graph = load_citation_graph()
    
    # Step 2: Display and select nodes
    nodes = display_available_nodes(graph)
    
    if not nodes:
        print("❌ No nodes with citations found in graph!")
        return
    
    if interactive:
        source, target = select_nodes_interactive(nodes)
    else:
        # Auto-select top 2 most cited cases
        source, target = select_nodes_auto(nodes, 0, 1)
    
    # Create output directory
    output_dir = Path(f"results/interactive_analysis/{source}_to_{target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 3: Calculate TRS and get reasoning
    trs_result = calculate_trs_and_reasoning(source, target, graph, output_dir)
    
    # Step 4: Generate RL reasoning summary
    rl_summary = generate_rl_reasoning_summary(trs_result, output_dir)
    
    # Step 5: Get internal coherence analysis
    internal_coherence = get_internal_coherence_analysis(source, target, graph, output_dir)
    
    # Step 6: Generate final report
    generate_final_report(source, target, trs_result, internal_coherence, output_dir)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check if interactive mode requested
    interactive = "--interactive" in sys.argv or "-i" in sys.argv
    
    if interactive:
        print("\n🔍 Running in INTERACTIVE mode - you will select the nodes")
    else:
        print("\n🤖 Running in AUTO mode - nodes will be auto-selected")
        print("   Use --interactive or -i flag for manual selection\n")
    
    main(interactive=interactive)
