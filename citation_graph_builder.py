#!/usr/bin/env python3
"""
Citation Graph Builder for Legal Cases

This module builds citation graphs from plain-text legal judgments. It extracts
citations using regex patterns, matches them to existing cases, and creates a
directed graph where edges represent citations (source cites target).

Usage:
    python citation_graph_builder.py --input data/lecai_baseline --output data/processed/citation_graph.json
    python citation_graph_builder.py --input data/lecai_baseline --output data/processed/citation_graph.json --png outputs/graph.png

Output:
    - JSON file with nodes (cases) and edges (citations)
    - Optional PNG visualization using NetworkX and Matplotlib

Requirements:
    - Required: json, re, pathlib, argparse
    - Optional: networkx, matplotlib (for PNG visualization)
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict


class CitationGraphBuilder:
    """Build citation graphs from legal documents."""
    
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, str]] = []
        self.node_map: Dict[str, Dict[str, Any]] = {}  # id -> node data
        self.citation_patterns = [
            # Pattern: "X v. Y (YEAR) N SCC M"
            r'([A-Z][A-Za-z\s&,\.]+?)\s+v\.?\s+([A-Za-z\s&,\.]+?)(?:\s*\((\d{4})\))?\s+(\d+)\s+SCC\s+\d+',
            # Pattern: "X v Y (YEAR)"
            r'([A-Z][A-Za-z\s&,\.]+?)\s+v\.?\s+([A-Za-z\s&,\.]+?)\s*\((\d{4})\)',
            # Pattern: "X vs. Y" or "X v. Y"
            r'([A-Z][A-Za-z\s&,\.]{3,30}?)\s+v[s]?\.?\s+([A-Za-z\s&,\.]{3,30}?)(?=\s|,|\.|\()',
        ]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for matching (lowercase, remove extra spaces)."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[,\.]', '', text)
        return text
    
    def extract_year(self, text: str) -> Optional[int]:
        """Extract the first 4-digit year between 1900-2099 from text."""
        match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if match:
            return int(match.group(1))
        return None
    
    def extract_title(self, text: str) -> str:
        """
        Extract case title from document text.
        
        Looks for:
        1. "X v. Y" or "X vs Y" pattern in first 500 chars
        2. First non-empty line
        3. First sentence
        """
        # Try to find case title pattern in first 500 chars
        snippet = text[:500]
        
        # Look for "X v. Y" pattern
        patterns = [
            r'([A-Z][A-Za-z\s&,\.]+?)\s+v\.?\s+([A-Za-z\s&,\.]+?)(?=\s*\(|\s*$|\n)',
            r'^([A-Z][A-Za-z\s&,\.\-]+?)$',  # All caps line
        ]
        
        for pattern in patterns:
            match = re.search(pattern, snippet, re.MULTILINE)
            if match:
                title = match.group(0).strip()
                if len(title) > 10 and len(title) < 200:
                    return title
        
        # Fallback: get first non-empty line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            # Skip common headers
            for line in lines[:10]:
                if len(line) > 10 and 'SUPREME COURT' not in line and 'JUDGMENT' not in line:
                    return line[:150]  # Truncate long titles
        
        return "Untitled Case"
    
    def extract_citations(self, text: str) -> List[Tuple[str, Optional[int]]]:
        """
        Extract citations from text.
        
        Returns:
            List of tuples (citation_text, year)
        """
        citations = []
        seen = set()
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Construct citation text
                if len(match.groups()) >= 2:
                    # Extract party names
                    party1 = match.group(1).strip()
                    party2 = match.group(2).strip()
                    citation_text = f"{party1} v. {party2}"
                    
                    # Extract year if present
                    year = None
                    if len(match.groups()) >= 3 and match.group(3):
                        try:
                            year = int(match.group(3))
                        except:
                            pass
                    
                    # Normalize for deduplication
                    normalized = self.normalize_text(citation_text)
                    if normalized not in seen and len(normalized) > 10:
                        seen.add(normalized)
                        citations.append((citation_text, year))
        
        return citations
    
    def generate_node_id(self, title: str, year: Optional[int] = None) -> str:
        """Generate a unique node ID from title and year."""
        normalized = self.normalize_text(title)
        # Remove common words and create slug
        slug = re.sub(r'\s+', '_', normalized)
        slug = re.sub(r'[^\w_]', '', slug)
        slug = slug[:50]  # Limit length
        
        if year:
            return f"{slug}_{year}"
        return slug
    
    def add_node(self, node_id: str, title: str, year: Optional[int], text_snippet: str) -> None:
        """Add a node to the graph if it doesn't exist."""
        if node_id not in self.node_map:
            node = {
                "id": node_id,
                "title": title,
                "year": year,
                "text": text_snippet[:200]  # Short snippet
            }
            self.nodes.append(node)
            self.node_map[node_id] = node
    
    def add_edge(self, source_id: str, target_id: str) -> None:
        """Add an edge (citation) to the graph."""
        edge = {"source": source_id, "target": target_id}
        self.edges.append(edge)
    
    def find_matching_node(self, citation_text: str, year: Optional[int]) -> Optional[str]:
        """
        Find an existing node that matches the citation.
        
        Returns node_id if found, None otherwise.
        """
        normalized_citation = self.normalize_text(citation_text)
        
        # Check for exact match by generated ID
        candidate_id = self.generate_node_id(citation_text, year)
        if candidate_id in self.node_map:
            return candidate_id
        
        # Check for fuzzy match in existing nodes
        for node_id, node in self.node_map.items():
            normalized_title = self.normalize_text(node['title'])
            
            # Check if citation is contained in title or vice versa
            if normalized_citation in normalized_title or normalized_title in normalized_citation:
                # If years match or one is None, consider it a match
                if year is None or node['year'] is None or year == node['year']:
                    return node_id
        
        return None
    
    def process_document(self, file_path: Path) -> Tuple[str, str, Optional[int], str]:
        """
        Process a single document and extract metadata.
        
        Returns:
            (node_id, title, year, text)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise
        
        title = self.extract_title(text)
        year = self.extract_year(text)
        node_id = self.generate_node_id(title, year)
        
        return node_id, title, year, text
    
    def build_graph(self, input_dir: Path) -> Tuple[int, int]:
        """
        Build citation graph from documents in input directory.
        
        Returns:
            (num_nodes, num_edges)
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Get all text files
        text_files = list(input_dir.glob('*.txt'))
        
        if not text_files:
            raise ValueError(f"No .txt files found in {input_dir}")
        
        print(f"Found {len(text_files)} documents to process")
        
        # First pass: create nodes for all documents
        doc_data = {}
        for file_path in text_files:
            try:
                node_id, title, year, text = self.process_document(file_path)
                text_snippet = text[:200].replace('\n', ' ')
                self.add_node(node_id, title, year, text_snippet)
                doc_data[node_id] = text
                print(f"  Processed: {title[:50]}...")
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
        
        # Second pass: extract citations and create edges
        print("\nExtracting citations...")
        for source_id, text in doc_data.items():
            citations = self.extract_citations(text)
            
            for citation_text, year in citations:
                # Try to find matching node
                target_id = self.find_matching_node(citation_text, year)
                
                if target_id:
                    # Found existing case
                    self.add_edge(source_id, target_id)
                    print(f"  {source_id} -> {target_id}")
                else:
                    # Create placeholder node for cited case not in corpus
                    placeholder_id = self.generate_node_id(citation_text, year)
                    if placeholder_id not in self.node_map:
                        self.add_node(placeholder_id, citation_text, year, "")
                    self.add_edge(source_id, placeholder_id)
                    print(f"  {source_id} -> {placeholder_id} (placeholder)")
        
        return len(self.nodes), len(self.edges)
    
    def save_json(self, output_path: Path) -> None:
        """Save graph to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        graph_data = {
            "nodes": self.nodes,
            "edges": self.edges
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
    
    def save_png(self, output_path: Path) -> None:
        """Save graph visualization to PNG file (requires networkx and matplotlib)."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("networkx/matplotlib not available: PNG not created.")
            print("Install with: pip install networkx matplotlib")
            return
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes with labels
        for node in self.nodes:
            label = node['title'][:30]  # Truncate long titles
            if len(node['title']) > 30:
                label += "..."
            G.add_node(node['id'], label=label, year=node['year'])
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge['source'], edge['target'])
        
        # Create visualization
        plt.figure(figsize=(16, 12))
        
        # Use spring layout for better node distribution
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1500, alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.5)
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Legal Citation Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PNG saved: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Build citation graph from legal documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/lecai_baseline --output data/processed/citation_graph.json
  %(prog)s --input data/lecai_baseline --output data/processed/citation_graph.json --png outputs/graph.png
        """
    )
    
    parser.add_argument('--input', required=True, type=Path,
                       help='Input directory containing .txt files')
    parser.add_argument('--output', required=True, type=Path,
                       help='Output JSON file path')
    parser.add_argument('--png', type=Path,
                       help='Optional: output PNG visualization path')
    
    args = parser.parse_args()
    
    # Build graph
    builder = CitationGraphBuilder()
    
    try:
        num_nodes, num_edges = builder.build_graph(args.input)
        
        # Save JSON
        builder.save_json(args.output)
        print(f"\ngraph saved: {args.output} ({num_nodes} nodes, {num_edges} edges)")
        
        # Save PNG if requested
        if args.png:
            builder.save_png(args.png)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
