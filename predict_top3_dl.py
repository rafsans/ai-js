import os
import pickle

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from extractors import extract_text_from_docx, extract_text_from_pdf
from project_utils import safe_clean

from logger import get_logger
log = get_logger("predict")

# Import untuk deteksi bahasa dan translasi
try:
    from langdetect import detect
except ImportError:
    log.warning("'langdetect' belum di-install. Jalankan: pip install langdetect")
    detect = None

try:
    from translator import translate_to_english
except ImportError:
    log.warning("Modul 'translator' tidak ditemukan atau gagal dimuat.")
    translate_to_english = lambda x: x

# ===========================================================================
# Paths — hasil output dari 3b_train_bert.py
# ===========================================================================
BERT_MODEL_DIR      = "models/bert_jobcategory"
LABEL_ENCODER_FILE  = "models/bert_label_encoder.pkl"


# ===========================================================================
# Helpers
# ===========================================================================

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Format file tidak didukung: {ext}. Gunakan PDF, DOCX, atau TXT.")


def confidence_level(prob: float) -> str:
    if prob > 0.50:
        return "High"
    if prob >= 0.30:
        return "Medium"
    return "Low"


# ===========================================================================
# Load model assets
# ===========================================================================

def load_inference_assets() -> dict:
    if not os.path.isdir(BERT_MODEL_DIR):
        raise FileNotFoundError(
            f"Folder model BERT tidak ditemukan: {BERT_MODEL_DIR}\n"
            "Jalankan 3b_train_bert.py untuk melatih model terlebih dahulu."
        )
    if not os.path.exists(LABEL_ENCODER_FILE):
        raise FileNotFoundError(
            f"Label encoder tidak ditemukan: {LABEL_ENCODER_FILE}\n"
            "Jalankan 3b_train_bert.py untuk melatih model terlebih dahulu."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Memuat tokenizer BERT dari: {BERT_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)

    log.info(f"Memuat model BERT dari: {BERT_MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.to(device)
    model.eval()

    with open(LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)

    log.info(f"BERT siap. Device: {device} | Kelas: {len(label_encoder.classes_)}")
    return {
        "model":         model,
        "tokenizer":     tokenizer,
        "label_encoder": label_encoder,
        "device":        device,
    }


# ===========================================================================
# Inference
# ===========================================================================

def predict_top3_text(input_text: str, assets: dict) -> tuple[list, str]:
    """
    Prediksi Top-3 job category dari teks input menggunakan BERT dengan Auto-Translate.

    Returns:
        tuple: (list of dict prediksi, str teks yang sudah di-translate/di-proses)
    """
    if not input_text or not input_text.strip():
        return [{"category": "Unknown", "confidence": 0.0}], input_text

    # 1. Otomatis deteksi bahasa & translasi ke English jika perlu
    translated_text = input_text
    try:
        if detect:
            lang = detect(input_text)
            if lang != "en":
                log.info(f"Deteksi bahasa: {lang}. Menerjemahkan ke English...")
                translated_text = translate_to_english(input_text)
            else:
                log.info("Deteksi bahasa: EN (English). Melompati translasi.")
        else:
            translated_text = translate_to_english(input_text)
    except Exception as e:
        log.warning(f"Gagal mendeteksi bahasa atau translasi: {e}. Menggunakan teks asli.")
        translated_text = input_text

    # 2. Text Preprocessing (Melindungi token teknologi & normalisasi)
    cleaned = safe_clean(translated_text)
    if not cleaned.strip():
        return [{"category": "Unknown", "confidence": 0.0}], translated_text

    model         = assets["model"]
    tokenizer     = assets["tokenizer"]
    label_encoder = assets["label_encoder"]
    device        = assets["device"]

    # 3. Tokenisasi
    inputs = tokenizer(
        cleaned,
        max_length=256,  # Pastikan sama dengan max_length saat training BERT Anda
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4. Forward pass model
    with torch.no_grad():
        logits = model(**inputs).logits

    probs   = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_k   = min(3, len(label_encoder.classes_))
    top_idx = np.argsort(probs)[-top_k:][::-1]

    results = []
    for idx in top_idx:
        label = label_encoder.inverse_transform([idx])[0]
        results.append({
            "category":   label,
            "confidence": float(probs[idx]),
        })

    if not results:
        results = [{"category": "Unknown", "confidence": 0.0}]

    return results, translated_text


def predict_from_file(file_path: str, assets: dict | None = None) -> dict:
    """
    Prediksi Top-3 job category dari file PDF/DOCX/TXT secara independen tanpa Gemini API.
    """
    assets   = assets or load_inference_assets()
    raw_text = extract_text_from_file(file_path)

    # Menyatukan pipeline ke predict_top3_text murni untuk klasifikasi
    results, translated_text = predict_top3_text(raw_text, assets)

    return {
        "filename":         os.path.basename(file_path),
        "confidence_level": confidence_level(results[0]["confidence"]),
        "top3_predictions": results,
    }


# ===========================================================================
# CLI test
# ===========================================================================

def main():
    assets = load_inference_assets()
    
    # Test teks menggunakan Bahasa Indonesia untuk membuktikan fungsi auto-translate
    text   = "Saya bisa pemrograman Python, SQL, machine learning dan membuat dashboard statistik"
    results, translated = predict_top3_text(text, assets)

    print("\n=== HASIL TRANSLASI ===")
    print(f"Original: {text}")
    print(f"Translated: {translated}")
    print("\n=== TOP-3 JOB CATEGORY PREDICTION (BERT) ===")
    for i, item in enumerate(results, start=1):
        print(f"Top {i}: {item['category']} -> {item['confidence']:.4f}")


if __name__ == "__main__":
    main()