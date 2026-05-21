"""
app.py
======
Flask API untuk Job Category Classifier.

Endpoints:
  GET  /             - info API
  POST /predict-text - prediksi Top-3 kategori dari teks JSON
  POST /predict      - prediksi Top-3 kategori dari file upload (PDF/DOCX/TXT)
  POST /match-jobs   - ranking pekerjaan berdasarkan cosine similarity resume
"""

import os
import pandas as pd

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from matching import (
    load_jobs,
    rank_jobs_by_resume_text,
    rank_jobs_by_category,
    rank_jobs_from_file_by_category,
)
from predict_top3_dl import predict_from_file, predict_top3_text, load_inference_assets

app = Flask(__name__)

UPLOAD_FOLDER      = "data/uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_ASSETS = None
JOBS_DF      = None

try:
    MODEL_ASSETS = load_inference_assets()
    print("[INFO] Model inference berhasil di-load.")
except Exception as e:
    print(f"[WARNING] Model inference belum bisa di-load: {e}")
    print("[INFO] Jalankan 2_train_deep_learning.py untuk melatih model terlebih dahulu.")

try:
    # Memuat jobs_recommendation.csv (diatur di matching.py)
    JOBS_DF = load_jobs()
    print(f"[INFO] Dataset job berhasil di-load ({len(JOBS_DF):,} baris).")
except Exception as e:
    print(f"[WARNING] Dataset job belum bisa di-load: {e}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_job_recommendations(df) -> list:
    """
    Format output job recommendations.
    Menarik semua kolom relevan dari jobs_recommendation.csv dan memberikan 
    formatting presentasi (misal: penambahan USD).
    """
    output_cols = [
        "rank", 
        "job_title", 
        "job_type", 
        "gaji_perbulan", 
        "job_skill", 
        "label", 
        "text", 
        "cosine_similarity_score", 
        "match_percentage"
    ]
    available   = [col for col in output_cols if col in df.columns]
    result_df   = df[available].copy()
    
    # [PENAMBAHAN LOGIKA FORMATTING]
    # Memastikan angka gaji memiliki konteks mata uang sebelum disajikan
    if "gaji_perbulan" in result_df.columns:
        result_df["gaji_perbulan"] = result_df["gaji_perbulan"].apply(
            lambda x: f"USD {x}" if pd.notna(x) and str(x).strip() and "USD" not in str(x).upper() else x
        )

    return result_df.to_dict(orient="records")


def get_top_category(predictions: list) -> str | None:
    if not predictions:
        return None
    first = predictions[0]
    return first.get("category") or first.get("job_category") or first.get("label")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Job Category Classifier API",
        "status": "active",
        "model_loaded": MODEL_ASSETS is not None,
        "dataset_loaded": JOBS_DF is not None,
        "dataset_mapping": {
            "input_column": "text",
            "target_column": "label (job_category)",
        },
        "endpoints": {
            "GET  /":             "Info API ini",
            "POST /predict-text": "JSON {text}: prediksi Top-3 job category",
            "POST /predict":      "Upload file PDF/DOCX/TXT: prediksi Top-3 job category",
            "POST /match-jobs":   "JSON {resume_text, top_n}: ranking pekerjaan berdasarkan similarity",
        },
        "example_predict_text": {
            "request": {"text": "python sql data analyst dashboard machine learning"},
            "response": {
                "top3_predictions": [
                    {"category": "Information Technology", "confidence": 0.85},
                    {"category": "Engineering",            "confidence": 0.10},
                    {"category": "Finance",                "confidence": 0.05},
                ]
            }
        },
    })


@app.route("/predict-text", methods=["POST"])
def predict_text():
    if MODEL_ASSETS is None:
        return jsonify({
            "error": "Model belum tersedia. Jalankan 2_train_deep_learning.py terlebih dahulu."
        }), 500

    payload = request.get_json(silent=True) or {}
    text    = payload.get("text", "")
    top_n   = int(payload.get("top_n", 5))

    if not str(text).strip():
        return jsonify({"error": "Field JSON 'text' wajib diisi dan tidak boleh kosong."}), 400

    from translator import translate_to_english, analyze_resume
    translated_text = translate_to_english(text)
    cv_feedback     = analyze_resume(text)

    predictions        = predict_top3_text(translated_text, MODEL_ASSETS)
    predicted_category = get_top_category(predictions)

    response = {
        "top3_predictions":  predictions,
        "predicted_category": predicted_category,
        "translated_text": translated_text,
        "cv_feedback": cv_feedback
    }

    if JOBS_DF is not None and predicted_category:
        specific_jobs = rank_jobs_by_category(
            resume_text=translated_text,
            predicted_category=predicted_category,
            jobs_df=JOBS_DF,
            top_n=top_n
        )
        response["job_recommendations"] = format_job_recommendations(specific_jobs)

    return jsonify(response), 200


@app.route("/predict", methods=["POST"])
def predict_file():
    if MODEL_ASSETS is None:
        return jsonify({
            "error": "Model belum tersedia. Jalankan 2_train_deep_learning.py terlebih dahulu."
        }), 500

    if "file" not in request.files:
        return jsonify({"error": "Field 'file' tidak ditemukan di request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Format file harus PDF, DOCX, atau TXT."}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        result             = predict_from_file(save_path, assets=MODEL_ASSETS)
        predicted_category = get_top_category(result.get("top3_predictions", []))
        result["predicted_category"] = predicted_category

        if JOBS_DF is not None and predicted_category:
            specific_jobs = rank_jobs_by_category(
                resume_text=result.get("translated_text", ""),
                predicted_category=predicted_category,
                jobs_df=JOBS_DF,
                top_n=5
            )
            result["job_recommendations"] = format_job_recommendations(specific_jobs)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/match-jobs", methods=["POST"])
def match_jobs():
    if JOBS_DF is None:
        return jsonify({
            "error": "Dataset job belum tersedia. Pastikan file jobs_recommendation.csv ada."
        }), 500

    payload     = request.get_json(silent=True) or {}
    resume_text = payload.get("resume_text", "")
    top_n       = int(payload.get("top_n", 10))

    if not str(resume_text).strip():
        return jsonify({"error": "Field JSON 'resume_text' wajib diisi dan tidak boleh kosong."}), 400

    from translator import translate_to_english
    translated_text = translate_to_english(resume_text)

    ranked = rank_jobs_by_resume_text(translated_text, jobs_df=JOBS_DF, top_n=top_n)
    return jsonify({
        "total_matches": len(ranked),
        "matches": format_job_recommendations(ranked),
        "translated_text": translated_text
    }), 200


if __name__ == "__main__":
    app.run(debug=True)