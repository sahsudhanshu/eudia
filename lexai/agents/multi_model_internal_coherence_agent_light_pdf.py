import re
import json
from typing import List, Dict, Any

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from .ocr_agent import process_pdf  # Import OCR agent


# ===========================================
# 1. Model Setup (Lightweight Versions)
# ===========================================
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.tools import DuckDuckGoSearchRun

# -----------------------------
# 1. Lightweight Classification & NLI
# -----------------------------
# Use a smaller and faster NLI model for zero-shot tasks
extractor = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",     # moderate accuracy, faster than distilbart-mnli
    device=-1                            # ensure CPU
)

# Or for extreme speed (sacrifices a bit of accuracy):
# extractor = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)

# -----------------------------
# 2. Tiny Embedding Model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # small, fast, robust on CPU
)

# -----------------------------
# 3. Logic Reasoning Model (Small)
# -----------------------------
logic_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",          # decent reasoning, still fast on CPU
    max_new_tokens=200,
    device=-1
)
logic_llm = HuggingFacePipeline(pipeline=logic_pipe)

# -----------------------------
# 4. NLI (Contradiction Detection)
# -----------------------------
nli = pipeline(
    "text-classification",
    model="typeform/distilbert-base-uncased-mnli",  # very small, optimized
    device=-1
)

# -----------------------------
# 5. Final Summarization / Report Model
# -----------------------------
final_pipe = pipeline(
    "text-generation",
    model="microsoft/Phi-1.5",            # lighter than Phi-2, still coherent
    max_new_tokens=300,
    temperature=0.3,
    device=-1
)
final_llm = HuggingFacePipeline(pipeline=final_pipe)

# -----------------------------
# 6. Search Tool (Online Data)
# -----------------------------
search = DuckDuckGoSearchRun()


# ===========================================
# 2. Regex Citation Extractor
# ===========================================
def extract_citations_regex(text: str) -> List[str]:
    patterns = [
        r"Article\s\d+[A-Z]?(?:\(\d+\))?",
        r"(?:Section|Sec\.?)\s\d+[A-Z]?(?:\(\d+\))?",
        r"[A-Z][A-Za-z&\s]+(?:vs\.?|v\.?)\s[A-Z][A-Za-z&\s]+",
        r"[A-Z][a-zA-Z\s]+Act(?:,\s*\d{4})?",
        r"\bAIR\s\d{4}\s[A-Z]{2,}\s\d+\b",
        r"\b\d{4}\s*\(\d+\)\s*SCC\s*\d+\b",
    ]
    found = []
    for p in patterns:
        found += re.findall(p, text, flags=re.IGNORECASE)
    return list(set([f.strip() for f in found]))


# ===========================================
# 3. Claim Extraction
# ===========================================
def extract_claims(text: str) -> List[str]:
    # Limit text length to avoid processing too much data
    text = text[:50000]  # Limit to first 50k characters
    
    sentences = re.split(r'(?<=[\.\n])\s+', text)
    claims = []
    
    # Process only first 100 sentences to avoid long processing
    for s in sentences[:100]:
        if not s.strip():
            continue
        res = extractor(s, candidate_labels=["legal argument", "evidence", "factual statement"], multi_label=False)
        if res["labels"][0] in ["legal argument", "evidence"] and res["scores"][0] > 0.5:
            claims.append(s.strip())
        
        # Stop if we have enough claims
        if len(claims) >= 20:
            break
    
    return claims


# ===========================================
# 4. FAISS Index Builder
# ===========================================
def build_faiss_index(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)
    vectordb = FAISS.from_texts(chunks, embedding=embedding_model)
    return vectordb


# ===========================================
# 5. Logical Flow Analyzer
# ===========================================
def analyze_logical_flows(claims: List[str]) -> str:
    # Limit claims to avoid CUDA OOM - take first 10 claims or truncate text
    limited_claims = claims[:10] if len(claims) > 10 else claims
    joined_claims = "\n".join([f"- {c[:200]}" for c in limited_claims])  # Truncate each claim to 200 chars
    
    # Ensure total prompt is under 400 tokens (roughly 1600 chars)
    if len(joined_claims) > 1400:
        joined_claims = joined_claims[:1400] + "..."
    
    prompt = f"Summarize the logical flow among these legal claims:\n{joined_claims}"
    return logic_llm.invoke(prompt)


# ===========================================
# 6. Contradiction Detector
# ===========================================
def detect_contradictions(claims: List[str]) -> List[str]:
    contradictions = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            pair = f"{claims[i]} </s> {claims[j]}"
            result = nli(pair)
            label = result[0]["label"].lower()
            if "contradiction" in label:
                contradictions.append(f"Contradiction between: '{claims[i]}' AND '{claims[j]}'")
    return contradictions


# ===========================================
# 7. Context Retrieval
# ===========================================
def retrieve_context(vectordb, claims: List[str]) -> Dict[str, List[str]]:
    context = {}
    for c in claims:
        try:
            docs = vectordb.similarity_search(c, k=2)
            snippets = [d.page_content for d in docs]
        except Exception:
            snippets = []
        context[c] = snippets
    return context


# ===========================================
# 8. Coherence Scoring (Safe JSON Handling)
# ===========================================
def generate_coherence_report(document: str, flows: str, contradictions: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
    sample_context = "\n".join([f"{k}: {v[:2]}" for k, v in context.items()])
    prompt = f"""
You are a legal reasoning coherence analyzer.

Document excerpt:
{document[:1000]}

Argument Flows:
{flows}

Contradictions:
{contradictions}

Context:
{sample_context[:700]}

Return JSON with exactly these keys:
- Key Argument Flows (list of strings)
- Detected Contradictions (list of strings)
- Logical Gaps (list of strings)
- Coherence Score (float between 0 and 1)
- Brief Commentary (string)
"""
    output = final_llm.invoke(prompt)

    try:
        cleaned = output.replace("\n", "").replace(",}", "}")
        return json.loads(cleaned)
    except Exception:
        # fallback safe structure
        return {
            "Key Argument Flows": flows.split("\n") if flows else [],
            "Detected Contradictions": contradictions,
            "Logical Gaps": [],
            "Coherence Score": 0.0,
            "Brief Commentary": output
        }


# ===========================================
# 9. Master Orchestrator (PDF Input)
# ===========================================
def run_internal_coherence_agent(pdf_path: str) -> Dict[str, Any]:
    ocr_result = process_pdf(pdf_path)
    document_text = ocr_result["raw_text"]

    citations = extract_citations_regex(document_text)
    claims = extract_claims(document_text)
    vectordb = build_faiss_index(document_text)
    flows = analyze_logical_flows(claims)
    contradictions = detect_contradictions(claims)
    context = retrieve_context(vectordb, claims[:5])
    report = generate_coherence_report(document_text, flows, contradictions, context)

    return {
        "File Name": ocr_result["file_name"],
        "Title": ocr_result["title"],
        "Citations": citations,
        "Claims": claims,
        "Contradictions": contradictions,
        "Final Report": report
    }


# ===========================================
# Example Run
# ===========================================
if __name__ == "__main__":
    pdf_path = "eudia/lexai/data/raw/1-266Right_to_Privacy__Puttaswamy_Judgment-Chandrachud.pdf"
    result = run_internal_coherence_agent(pdf_path)
    print(json.dumps(result, indent=2))