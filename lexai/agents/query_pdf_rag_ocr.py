# lexai/agents/query_pdf_rag_ocr.py
import os
import json
from typing import Dict

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    return rag_llm.invoke(prompt)


# ===========================================
# 4. Master Function: PDF → Answer
# ===========================================
def query_pdf_with_ocr(pdf_path: str, query: str) -> Dict[str, str]:
    # Extract text from PDF
    ocr_result = process_pdf(pdf_path)
    document_text = ocr_result["raw_text"]

    # Build FAISS index
    vectordb = build_faiss_index(document_text)

    # Query LLM using retrieved context
    answer = query_rag(vectordb, query)

    return {
        "File Name": ocr_result["file_name"],
        "Title": ocr_result["title"],
        "Query": query,
        "Answer": answer
    }


# ===========================================
# Example Run
# ===========================================
if __name__ == "__main__":
    pdf_path = "eudia/lexai/data/raw/1-266Right_to_Privacy__Puttaswamy_Judgment-Chandrachud.pdf"
    user_query = "What does the judgment say about Article 19(1)(a)?"

    result = query_pdf_with_ocr(pdf_path, user_query)
    print(json.dumps(result, indent=2))