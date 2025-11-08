"""Example usage of ExternalInferenceAgent with legal documents."""

import json
from lexai.agents import ExternalInferenceAgent


def load_legal_document(file_path: str) -> dict:
    """Load a legal document from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Demonstrate ExternalInferenceAgent with legal documents."""
    
    # Initialize the agent
    print("Initializing ExternalInferenceAgent...")
    agent = ExternalInferenceAgent(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"  # Change to "cuda" if GPU available
    )
    
    # Load sample legal document
    doc_path = "lexai/data/raw/1-266Right_to_Privacy__Puttaswamy_Judgment-Chandrachud.json"
    legal_doc = load_legal_document(doc_path)
    
    # Create sample candidates from the document
    # In a real scenario, you'd have multiple documents
    candidates = []
    
    # Split the document into chunks (simplified example)
    text = legal_doc.get("raw_text", "")
    chunk_size = 500  # characters per chunk
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            candidates.append({
                "text": chunk,
                "source": legal_doc["file_name"],
                "title": legal_doc["title"],
                "chunk_id": i // chunk_size
            })
    
    print(f"Created {len(candidates)} candidate chunks from document")
    
    # Build the index
    print("\nBuilding FAISS index...")
    agent.build_index(candidates, text_field="text")
    
    # Get index stats
    stats = agent.get_index_stats()
    print(f"\nIndex Statistics:")
    print(f"  - Built: {stats['is_built']}")
    print(f"  - Candidates: {stats['num_candidates']}")
    print(f"  - Embedding Dimension: {stats['embedding_dimension']}")
    print(f"  - Model: {stats['model_name']}")
    
    # Perform inference with sample queries
    queries = [
        "right to privacy",
        "fundamental rights",
        "constitutional law"
    ]
    
    print("\n" + "="*60)
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        results = agent.infer(query, top_k=3)
        
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. Similarity: {result['similarity_score']:.4f}")
            print(f"   Chunk ID: {result['chunk_id']}")
            print(f"   Text Preview: {result['text'][:100]}...")
    
    print("\n" + "="*60)
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
