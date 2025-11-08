"""Citation graph generation service."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any
import tempfile


def generate_citation_graph(ocr_text: str, document_title: str) -> Dict[str, Any]:
    """
    Generate citation graph from OCR text using CitationGraphBuilder.
    
    Args:
        ocr_text: Full extracted text from document
        document_title: Title of the document
        
    Returns:
        Dictionary with 'nodes' and 'edges' keys
    """
    # Add project root to path to import citation_graph_builder
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from citation_graph_builder import CitationGraphBuilder
    
    # Create a temporary file with the OCR text
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(ocr_text)
        tmp_path = Path(tmp_file.name)
    
    try:
        # Create a temporary directory structure for the builder
        tmp_dir = tmp_path.parent
        
        builder = CitationGraphBuilder()
        
        # Process the single document
        node_id, title, year, text = builder.process_document(tmp_path)
        
        # Override title with document title if available
        if document_title and document_title != "Untitled":
            title = document_title
            node_id = builder.generate_node_id(document_title, year)
        
        # Add the main document node
        text_snippet = text[:200].replace('\n', ' ')
        builder.add_node(node_id, title, year, text_snippet)
        
        # Extract citations from the text
        citations = builder.extract_citations(text)
        
        # Create nodes and edges for citations
        for citation_text, citation_year in citations:
            target_id = builder.find_matching_node(citation_text, citation_year)
            
            if not target_id:
                # Create placeholder node for cited case
                target_id = builder.generate_node_id(citation_text, citation_year)
                if target_id not in builder.node_map:
                    builder.add_node(target_id, citation_text, citation_year, "")
            
            # Add edge from main document to cited case
            builder.add_edge(node_id, target_id)
        
        return {
            "nodes": builder.nodes,
            "edges": builder.edges
        }
        
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()
