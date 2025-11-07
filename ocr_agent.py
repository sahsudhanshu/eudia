# lexai/agents/ocr_agent.py
import pdfplumber
import re
import os
import json

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from PDF using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_title(text: str) -> str:
    """
    Try to extract a title from the first 10 lines.
    Usually, it's in uppercase or 'X vs Y' style.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:10]:
        if re.search(r"v\.|vs\.|V\.|Vs\.", line):
            return line
        if line.isupper() and len(line.split()) > 2:
            return line
    return lines[0] if lines else "Untitled Document"

def extract_citations(text: str):
    """
    Extract citations like:
    - ABC v. XYZ
    - State of Kerala vs. Rajesh
    - AIR 1992 SC 604
    """
    citation_patterns = [
        r"[A-Z][A-Za-z&\s,]*v\.?[sS]?\.?\s*[A-Z][A-Za-z&\s,]*",   # e.g., A v. B or A vs. B
        r"AIR\s?\d{4}\s?[A-Z]*\s?\d+",                             # e.g., AIR 1992 SC 604
        r"\d{4}\s?\(\d+\)\s?[A-Z]+",                               # e.g., 2019 (4) SCC
    ]
    citations = []
    for pat in citation_patterns:
        citations += re.findall(pat, text)
    # Deduplicate and clean
    citations = list({c.strip(): None for c in citations}.keys())
    return citations

def extract_articles(text: str):
    """
    Extracts 'Article' and 'Section' mentions.
    e.g., Article 21, Section 34(2), Sec. 302 IPC
    """
    patterns = [
        r"Article\s+\d+[A-Z]?", 
        r"Section\s+\d+[A-Z]?\(?\d*\)?",
        r"Sec\.\s*\d+[A-Z]?"
    ]
    matches = []
    for p in patterns:
        matches += re.findall(p, text, flags=re.IGNORECASE)
    return list({m.strip(): None for m in matches}.keys())

def process_pdf(pdf_path: str, output_dir="lexai/data/raw"):
    """Runs full OCR + extraction pipeline for a PDF."""
    os.makedirs(output_dir, exist_ok=True)
    text = extract_text_from_pdf(pdf_path)
    title = extract_title(text)
    citations = extract_citations(text)
    articles = extract_articles(text)

    result = {
        "file_name": os.path.basename(pdf_path),
        "title": title,
        "citations": citations,
        "articles": articles,
        "raw_text": text
    }

    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(pdf_path))[0] + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Extracted: {pdf_path} → {out_path}")
    print(f"Title: {title}")
    print(f"Found {len(citations)} citations and {len(articles)} articles.")
    return result
