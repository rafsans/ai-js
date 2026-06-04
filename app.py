import os
import uuid
import requests

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

from matching import (
    load_jobs,
    rank_jobs_by_resume_text,
    rank_jobs_by_category,
    rank_jobs_from_file_by_category,
)
from predict_top3_dl import predict_from_file, predict_top3_text, load_inference_assets
from extractors import extract_text_from_pdf, extract_text_from_docx, extract_text_from_file
from logger import get_logger

app = Flask(__name__)
log = get_logger("app")

_redis_url     = os.getenv("REDIS_URL", "")
_storage_uri   = _redis_url if _redis_url else "memory://"

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=_storage_uri,
)

if not _redis_url:
    import logging
    logging.getLogger("app").warning(
        "REDIS_URL tidak dikonfigurasi. Rate limiter menggunakan in-memory storage "
        "(counter tidak persisten antar restart dan tidak efektif di multi-worker)."
    )

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER      = "data/uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_ASSETS = None
JOBS_DF      = None

try:
    MODEL_ASSETS = load_inference_assets()
    log.info("Model BERT berhasil di-load.")
except Exception as e:
    log.warning(f"Model BERT belum bisa di-load: {e}")
    log.info("Jalankan 3b_train_bert.py untuk melatih model terlebih dahulu.")

try:
    JOBS_DF = load_jobs()
    log.info(f"Dataset job berhasil di-load ({len(JOBS_DF):,} baris).")
except Exception as e:
    log.warning(f"Dataset job belum bisa di-load: {e}")


@app.before_request
def check_api_key():
    
    expected_api_key = os.getenv("API_KEY")
    if expected_api_key:
        api_key = request.headers.get("api-key")
        if not api_key or api_key != expected_api_key:
            return jsonify({"error": "Unauthorized. Invalid or missing api-key in headers."}), 401


@app.errorhandler(413)
def request_entity_too_large(error):
    """Menangani error jika user mengupload file lebih dari 5MB."""
    return jsonify({
        "error": "Ukuran file terlalu besar. Batas maksimal yang diizinkan adalah 5MB."
    }), 413


@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Menangani error rate limit — kembalikan JSON bukan HTML."""
    return jsonify({
        "error": "Terlalu banyak permintaan. Silakan tunggu sebelum mencoba lagi.",
        "detail": str(error.description)
    }), 429


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_job_recommendations(df) -> list:
    output_cols = ["rank", "label", "text", "cosine_similarity_score", "match_percentage","job_title","job_type","gaji_perbulan","salary"]
    available   = [col for col in output_cols if col in df.columns]
    return df[available].copy().to_dict(orient="records")


def get_top_category(predictions: list) -> str | None:
    if not predictions:
        return None
    first = predictions[0]
    return first.get("category") or first.get("job_category") or first.get("label")





def save_uploaded_file(file) -> str:
    """Simpan file upload dan kembalikan path-nya."""
    filename  = secure_filename(file.filename)
    if not filename:
        raise ValueError("Nama file tidak valid setelah sanitasi.")
    ext       = os.path.splitext(filename)[1]
    filename  = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    return save_path


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Job Category Classifier API",
        "model":   "BERT (bert-base-uncased)",
        "status":  "active",
        "model_loaded":   MODEL_ASSETS is not None,
        "dataset_loaded": JOBS_DF is not None,
        "endpoints": {
            "GET  /":              "Info API ini",
            "POST /translate":     "JSON {text} → translasi ke Bahasa Inggris",
            "POST /predict-text":  "JSON {text} → prediksi Top-3 job category",
            "POST /predict":       "Upload file PDF/DOCX/TXT → prediksi Top-3 job category",
            "POST /match-jobs":    "JSON {resume_text, top_n} → ranking pekerjaan",
            "POST /analyze-cv":    "JSON {text} atau upload file → feedback & saran perbaikan CV",
        },
    })


@app.route("/predict-text", methods=["POST"])
def predict_text():
    if MODEL_ASSETS is None:
        return jsonify({"error": "Model BERT belum tersedia. Jalankan 3b_train_bert.py terlebih dahulu."}), 500

    payload = request.get_json(silent=True) or {}
    text    = payload.get("text", "")
    top_n   = max(1, min(int(payload.get("top_n", 5)), 50))

    if not str(text).strip():
        return jsonify({"error": "Field JSON 'text' wajib diisi dan tidak boleh kosong."}), 400

    predictions, translated_text = predict_top3_text(
        text,
        MODEL_ASSETS
    )

    predicted_category = get_top_category(
        predictions
    )

    response = {
        "original_text": text,
        "top3_predictions": predictions,
        "predicted_category": predicted_category,
    }

    if JOBS_DF is not None and predicted_category:
        specific_jobs = rank_jobs_by_category(
            resume_text=translated_text,
            predicted_category=predicted_category,
            jobs_df=JOBS_DF,
            top_n=top_n,
        )
        response["job_recommendations"] = format_job_recommendations(specific_jobs)

    return jsonify(response), 200


@app.route("/predict", methods=["POST"])
@limiter.limit("5 per minute; 20 per hour")
def predict_file():
    if MODEL_ASSETS is None:
        return jsonify({"error": "Model BERT belum tersedia. Jalankan 3b_train_bert.py terlebih dahulu."}), 500

    top_n = max(1, min(int(request.form.get("top_n", 5)), 50))

    if "file" not in request.files:
        return jsonify({"error": "Field 'file' tidak ditemukan di request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Format file harus PDF, DOCX, atau TXT."}), 400

    save_path = None
    try:
        save_path          = save_uploaded_file(file)
        log.info(f"File diterima: {file.filename} ({os.path.getsize(save_path) / 1024:.1f} KB)")
        result             = predict_from_file(save_path, assets=MODEL_ASSETS)
        predicted_category = get_top_category(result.get("top3_predictions", []))
        result["predicted_category"] = predicted_category
        log.info(f"Prediksi selesai: {file.filename} → {predicted_category}")

        if JOBS_DF is not None and predicted_category:
            specific_jobs = rank_jobs_from_file_by_category(
                file_path=save_path,
                predicted_category=predicted_category,
                jobs_df=JOBS_DF,
                top_n=top_n,
            )
            result["job_recommendations"] = format_job_recommendations(specific_jobs)

        return jsonify(result), 200

    except Exception as e:
        log.error(f"Predict error untuk file '{getattr(file, 'filename', '?')}': {e}")
        return jsonify({"error": "Terjadi kesalahan saat memproses file. Silakan coba lagi."}), 500
    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)


@app.route("/predict-url", methods=["POST"])
@limiter.limit("5 per minute; 20 per hour")
def predict_url():
    if MODEL_ASSETS is None:
        return jsonify({"error": "Model BERT belum tersedia."}), 500

    payload = request.get_json(silent=True) or {}
    file_url = payload.get("url")
    top_n = max(1, min(int(payload.get("top_n", 5)), 50))

    if not file_url:
        return jsonify({"error": "URL wajib diisi."}), 400

    save_path = None
    try:
        resp = requests.get(file_url, stream=True, timeout=15)
        resp.raise_for_status()
        
        ext = ".pdf"
        if ".docx" in file_url.lower(): ext = ".docx"
        elif ".txt" in file_url.lower(): ext = ".txt"
            
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                
        log.info(f"File didownload dari URL: {file_url} -> {filename}")
        
        result = predict_from_file(save_path, assets=MODEL_ASSETS)
        predicted_category = get_top_category(result.get("top3_predictions", []))
        result["predicted_category"] = predicted_category
        
        if JOBS_DF is not None and predicted_category:
            specific_jobs = rank_jobs_from_file_by_category(
                file_path=save_path,
                predicted_category=predicted_category,
                jobs_df=JOBS_DF,
                top_n=top_n,
            )
            result["job_recommendations"] = format_job_recommendations(specific_jobs)

        return jsonify(result), 200
        
    except Exception as e:
        log.error(f"Predict URL error: {e}")
        return jsonify({"error": f"Gagal memproses file dari URL: {str(e)}"}), 500
    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

@app.route("/match-jobs", methods=["POST"])
@limiter.limit("10 per minute; 50 per hour")
def match_jobs():
    if JOBS_DF is None:
        return jsonify({"error": "Dataset job belum tersedia. Pastikan data/ds_jobs_ready.csv ada."}), 500

    payload     = request.get_json(silent=True) or {}
    resume_text = payload.get("resume_text", "")
    top_n       = max(1, min(int(payload.get("top_n", 10)), 50))

    if not str(resume_text).strip():
        return jsonify({"error": "Field JSON 'resume_text' wajib diisi dan tidak boleh kosong."}), 400

    ranked = rank_jobs_by_resume_text(resume_text, jobs_df=JOBS_DF, top_n=top_n)
    return jsonify({
        "total_matches": len(ranked),
        "matches":       format_job_recommendations(ranked),
    }), 200


@app.route("/analyze-cv", methods=["POST"])
@limiter.limit("5 per minute; 20 per hour")
def analyze_cv():
    try:
        from translator import analyze_resume, translate_to_english
    except ImportError as e:
        return jsonify({"error": f"Modul translator tidak tersedia: {e}"}), 500

    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Nama file kosong."}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Format file harus PDF, DOCX, atau TXT."}), 400

        save_path = None
        try:
            save_path = save_uploaded_file(file)
            log.info(f"File diterima: {file.filename} ({os.path.getsize(save_path) / 1024:.1f} KB)")
            raw_text  = extract_text_from_file(save_path)
            source    = "file"
            filename  = os.path.basename(save_path)
            log.info(f"Analisis CV selesai diekstrak: {file.filename}")
        except Exception as e:
            return jsonify({"error": f"Gagal membaca file: {e}"}), 500
        finally:
            if save_path and os.path.exists(save_path):
                os.remove(save_path)

    else:
        payload  = request.get_json(silent=True) or {}
        raw_text = payload.get("text", "")
        source   = "text"
        filename = None

    if not str(raw_text).strip():
        return jsonify({"error": "Teks CV tidak boleh kosong."}), 400

    try:
        translated_text = translate_to_english(raw_text)
        feedback        = analyze_resume(translated_text)
    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Analisis gagal: {e}"}), 500

    response = {
        "cv_feedback": feedback,
        "source":      source,
    }
    if filename:
        response["filename"] = filename

    return jsonify(response), 200


@app.route("/analyze-cv-url", methods=["POST"])
@limiter.limit("5 per minute; 20 per hour")
def analyze_cv_url():
    try:
        from translator import analyze_resume, translate_to_english
    except ImportError as e:
        return jsonify({"error": f"Modul translator tidak tersedia: {e}"}), 500

    payload = request.get_json(silent=True) or {}
    file_url = payload.get("url")
    if not file_url:
        return jsonify({"error": "URL wajib diisi."}), 400

    save_path = None
    try:
        resp = requests.get(file_url, stream=True, timeout=15)
        resp.raise_for_status()
        
        ext = ".pdf"
        if ".docx" in file_url.lower(): ext = ".docx"
        elif ".txt" in file_url.lower(): ext = ".txt"
            
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                
        log.info(f"File didownload dari URL untuk dianalisis: {file_url} -> {filename}")
        
        raw_text = extract_text_from_file(save_path)
        if not str(raw_text).strip():
            return jsonify({"error": "Teks CV kosong setelah diekstrak."}), 400
            
        translated_text = translate_to_english(raw_text)
        feedback = analyze_resume(translated_text)
        
        return jsonify({
            "cv_feedback": feedback,
            "source": "url",
            "filename": filename
        }), 200
        
    except Exception as e:
        log.error(f"Analyze URL error: {e}")
        return jsonify({"error": f"Gagal menganalisis CV dari URL: {str(e)}"}), 500
    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)


@app.route("/translate", methods=["POST"])
@limiter.limit("10 per minute; 50 per hour")
def translate_text():
    try:
        from translator import translate_to_english
    except ImportError as e:
        return jsonify({
            "error": f"Translator module error: {e}"
        }), 500

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not str(text).strip():
        return jsonify({
            "error": "Field 'text' wajib diisi."
        }), 400

    try:
        translated = translate_to_english(text)
        return jsonify({
            "original_text": text,
            "translated_text": translated
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port       = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=port)
