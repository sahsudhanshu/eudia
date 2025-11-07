# multi_model_internal_coherence_agent.py

import re, json
from typing import List, Dict, Any

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# ===========================================
# 1. Model Setup (Multiple Specialized Models)
# ===========================================

# (A) Claim extractor – Zero-shot classification
extractor = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# (B) Embedding model for retrieval
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# (C) Logical flow detector – small reasoning model
logic_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=400)
logic_llm = HuggingFacePipeline(pipeline=logic_pipe)

# (D) Contradiction detector – NLI
nli = pipeline("text-classification", model="roberta-large-mnli")

# (E) Final coherence scorer – advanced generative model
final_pipe = pipeline("text-generation", model="mistralai/Mixtral-8x7B", max_new_tokens=700, temperature=0.3)
final_llm = HuggingFacePipeline(pipeline=final_pipe)

# (F) Optional web retriever
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
    sentences = re.split(r'(?<=[\.\n])\s+', text)
    claims = []
    for s in sentences:
        if not s.strip():
            continue
        res = extractor(s, candidate_labels=["legal argument", "evidence", "factual statement"], multi_label=False)
        if res["labels"][0] in ["legal argument", "evidence"] and res["scores"][0] > 0.6:
            claims.append(s.strip())
    return claims


# ===========================================
# 4. FAISS Index Builder for RAG
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
    joined_claims = "\n".join([f"- {c}" for c in claims])
    prompt = f"Identify the logical flow among these legal claims and summarize step-by-step:\n{joined_claims}"
    return logic_llm.invoke(prompt)


# ===========================================
# 6. Contradiction Detector
# ===========================================
def detect_contradictions(claims: List[str]) -> List[str]:
    contradictions = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            result = nli(f"{claims[i]} </s> {claims[j]}")
            label = result[0]["label"].lower()
            if "contradiction" in label:
                contradictions.append(f"Contradiction between: '{claims[i]}' AND '{claims[j]}'")
    return contradictions


# ===========================================
# 7. Context Retrieval (RAG + Web Search)
# ===========================================
def retrieve_context(vectordb, claims: List[str]) -> Dict[str, List[str]]:
    context = {}
    for c in claims:
        try:
            docs = vectordb.similarity_search(c, k=3)
            snippets = [d.page_content for d in docs]
        except Exception:
            snippets = []
        try:
            web_info = search.invoke(f"{c} Indian legal analysis")
            snippets.insert(0, f"[DuckDuckGo] {web_info}")
        except Exception:
            pass
        context[c] = snippets
    return context


# ===========================================
# 8. Coherence Scoring & Final JSON Output
# ===========================================
def generate_coherence_report(document: str, flows: str, contradictions: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
    sample_context = "\n".join([f"{k}: {v[:2]}" for k, v in context.items()])
    prompt = f"""
You are the INTERNAL COHERENCE AGENT.

Document:
{document[:1500]}...

Argument Flows:
{flows}

Contradictions Detected:
{contradictions}

Retrieved Context:
{sample_context[:1000]}

Analyze the internal legal reasoning and output JSON strictly in this format:
{{
  "Key Argument Flows": [...],
  "Detected Contradictions": [...],
  "Logical Gaps": [...],
  "Coherence Score": <float between 0 and 1>,
  "Brief Commentary": "..."
}}
"""
    output = final_llm.invoke(prompt)
    try:
        return json.loads(output)
    except Exception:
        return {"error": "Failed to parse model output", "raw_output": output}


# ===========================================
# 9. Master Orchestrator
# ===========================================
def run_internal_coherence_agent(document_text: str) -> Dict[str, Any]:
    citations = extract_citations_regex(document_text)
    claims = extract_claims(document_text)
    vectordb = build_faiss_index(document_text)
    flows = analyze_logical_flows(claims)
    contradictions = detect_contradictions(claims)
    context = retrieve_context(vectordb, claims[:5])
    report = generate_coherence_report(document_text, flows, contradictions, context)

    return {
        "Citations": citations,
        "Claims": claims,
        "Contradictions": contradictions,
        "Final Report": report
    }


# ===========================================
# Example Run
# ===========================================
if __name__ == "__main__":
    sample_text = """
    In CBSE vs Aditya Bandopadhyay, the court held that RTI requests must be balanced with privacy.
    The petitioner argued that Article 19(1)(a) guarantees unrestricted speech.
    However, the judgment later states that reasonable restrictions are vital to maintain order.
    It also refers to Section 7(1) RTI Act which mandates 30 days response for transparency.
    """
    result = run_internal_coherence_agent(sample_text)
    print(json.dumps(result, indent=2))
