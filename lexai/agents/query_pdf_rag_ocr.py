# lexai/agents/query_pdf_rag_ocr.py
import os
import json
from typing import Dict

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ocr_agent import process_pdf  # Use OCR agent to extract text


# ===========================================
# 1. Model Setup
# ===========================================
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
rag_pipe = pipeline("text-generation", model="microsoft/phi-2", max_new_tokens=400, temperature=0.3)
rag_llm = HuggingFacePipeline(pipeline=rag_pipe)


# ===========================================
# 2. FAISS Index Builder
# ===========================================
def build_faiss_index(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks available for indexing")
    vectordb = FAISS.from_texts(chunks, embedding=embedding_model)
    return vectordb


# ===========================================
# 3. Query RAG
# ===========================================
def query_rag(vectordb: FAISS, query: str, k: int = 3) -> str:
    docs = vectordb.similarity_search(query, k=k)
    context = "\n".join([d.page_content for d in docs])
    
    prompt = f"""
You are a legal assistant. Answer the question based on the following extracted PDF content:

Context:
{context}

Question:
{query}

Answer:
"""
    response = rag_llm.invoke(prompt)
    if isinstance(response, dict) and "generated_text" in response:
        return response["generated_text"]
    if isinstance(response, list):
        return "\n".join(str(item) for item in response)
    return str(response)


# ===========================================
# 4. Master Function: PDF → Answer
# ===========================================
def query_pdf_with_ocr(pdf_path: str, query: str) -> Dict[str, str]:
    ocr_result = process_pdf(pdf_path)
    document_text = ocr_result.get("raw_text", "").strip()

    if not document_text:
        return {
            "File Name": ocr_result.get("file_name", "Unknown"),
            "Title": ocr_result.get("title", "Untitled"),
            "Query": query,
            "Answer": "No textual content could be extracted from this document.",
        }

    try:
        vectordb = build_faiss_index(document_text)
        answer = query_rag(vectordb, query)
    except ValueError:
        answer = "No relevant passages were found to answer this question."

    return {
        "File Name": ocr_result.get("file_name", "Unknown"),
        "Title": ocr_result.get("title", "Untitled"),
        "Query": query,
        "Answer": answer,
    }


# ===========================================
# 5. Optimized Function: Query from Cached Text
# ===========================================
def query_rag_from_text(document_text: str, query: str, title: str = "Document") -> Dict[str, str]:
    """
    Query document using cached OCR text (skips OCR processing).
    This is much faster than query_pdf_with_ocr when text is already extracted.
    
    Args:
        document_text: Pre-extracted text from OCR
        query: Question to answer
        title: Document title
        
    Returns:
        Dictionary with query results
    """
    if not document_text or not document_text.strip():
        return {
            "Title": title,
            "Query": query,
            "Answer": "No textual content available for this document.",
        }

    try:
        vectordb = build_faiss_index(document_text)
        answer = query_rag(vectordb, query)
    except ValueError:
        answer = "No relevant passages were found to answer this question."

    return {
        "Title": title,
        "Query": query,
        "Answer": answer,
    }


# ===========================================
# Example Run
# ===========================================
if __name__ == "__main__":
    sample_pdf = "lexai/data/raw/1-266Right_to_Privacy__Puttaswamy_Judgment-Chandrachud.pdf"
    sample_query = "What does the judgment say about Article 19(1)(a)?"

    result = query_pdf_with_ocr(sample_pdf, sample_query)
    print(json.dumps(result, indent=2))