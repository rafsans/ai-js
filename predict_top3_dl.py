"""
predict_top3_dl.py
==================
Inferensi Top-3 Job Category dari teks atau file (PDF/DOCX/TXT).
Menggunakan HuggingFace BERT.
"""

import json
import os
import pickle
import warnings

# Menyembunyikan peringatan (FutureWarning/InconsistentVersion) dari pustaka bawaan
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from extractors import extract_text_from_docx, extract_text_from_pdf
from project_utils import clean_noise

MODEL_PATH         = "models/bert_jobcategory"
LABEL_ENCODER_FILE = "models/bert_label_encoder.pkl"


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


def load_inference_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Folder model tidak ditemukan: {MODEL_PATH}\n"
            "Pastikan model BERT sudah dilatih dan disimpan di folder tersebut."
        )
    
    if not os.path.exists(LABEL_ENCODER_FILE):
        raise FileNotFoundError(
            f"File label encoder tidak ditemukan: {LABEL_ENCODER_FILE}\n"
            "Pastikan file label encoder sudah ada."
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    with open(LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder
    }


def predict_top3_text(input_text: str, assets: dict) -> list:
    """
    Prediksi Top-3 job category dari teks input menggunakan BERT.
    """
    cleaned_text = clean_noise(input_text)

    model = assets["model"]
    tokenizer = assets["tokenizer"]
    label_encoder = assets["label_encoder"]

    # Tokenisasi input
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Inference tanpa gradient
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]

    top_k = min(6, len(label_encoder.classes_))
    top_probs, top_indices = torch.topk(probs, top_k)

    results = []

    for prob, idx in zip(top_probs, top_indices):
        label = label_encoder.inverse_transform([idx.item()])[0]
        
        results.append({
            "category": label,
            "confidence": round(prob.item(), 4)
        })

        if len(results) == 3:
            break

    if not results:
        results.append({
            "category": "Unknown",
            "confidence": 0.0
        })
    
    # Tambahkan confidence threshold untuk prediksi yang tidak pasti
    if results[0]["confidence"] < 0.45:
        results.insert(0, {
            "category": "UNCERTAIN",
            "confidence": round(results[0]["confidence"], 4)
        })

    return results


def predict_from_file(file_path: str, assets: dict | None = None) -> dict:
    assets = assets or load_inference_assets()
    raw_text = extract_text_from_file(file_path)
    
    # Translasi dan Analisis CV secara paralel/sekuensial
    from translator import translate_to_english, analyze_resume
    translated_text = translate_to_english(raw_text)
    cv_feedback     = analyze_resume(raw_text)
    
    results = predict_top3_text(translated_text, assets)

    return {
        "filename": os.path.basename(file_path),
        "confidence_level": confidence_level(results[0]["confidence"]),
        "top3_predictions": results,
        "translated_text": translated_text,
        "cv_feedback": cv_feedback
    }


def main():
    # Contoh penggunaan
    assets = load_inference_assets()
    text = "Python SQL machine learning data analysis dashboard statistics"
    result = predict_top3_text(text, assets)

    print("=== TOP-3 JOB CATEGORY PREDICTION (BERT) ===")
    for i, item in enumerate(result, start=1):
        print(f"Top {i}: {item['category']} -> {item['confidence']:.4f}")


if __name__ == "__main__":
    main()