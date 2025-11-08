# test_ocr_agent.py
from eudia.lexai.agents.ocr_agent import process_pdf

pdf_path = "1-266Right_to_Privacy__Puttaswamy_Judgment-Chandrachud.pdf"
output = process_pdf(pdf_path)

print(output["title"])
print(output["citations"])
print(output["articles"])
