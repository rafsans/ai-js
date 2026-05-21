"""
extractors.py
=============
Utilitas ekstraksi teks dari file PDF dan DOCX.
"""

import fitz   # PyMuPDF
import docx


def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []
    doc = fitz.open(file_path)
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_path: str) -> str:
    document   = docx.Document(file_path)
    text_parts = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(text_parts).strip()
