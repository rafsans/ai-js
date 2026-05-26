import fitz   # PyMuPDF
import docx

_MIN_TEXT_LENGTH = 50  # Minimum karakter agar dianggap bukan PDF kosong/scan


def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []
    doc = fitz.open(file_path)
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    result = "\n".join(text_parts).strip()
    if len(result) < _MIN_TEXT_LENGTH:
        raise ValueError(
            "Teks tidak dapat diekstrak dari file PDF ini. "
            "Kemungkinan file berbasis gambar/scan dan tidak mengandung teks yang dapat dibaca. "
            "Silakan gunakan PDF dengan teks yang dapat diseleksi."
        )
    return result


def extract_text_from_docx(file_path: str) -> str:
    document   = docx.Document(file_path)
    text_parts = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(text_parts).strip()


def extract_text_from_file(file_path: str) -> str:
    """Ekstrak teks dari file PDF, DOCX, atau TXT."""
    import os
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Format file tidak didukung: {ext}")