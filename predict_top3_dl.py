import os
import pickle

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from extractors import extract_text_from_docx, extract_text_from_pdf
from project_utils import safe_clean
from logger import get_logger

log = get_logger("predict")

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
BERT_MODEL_DIR       = "models/bert_jobcategory"
BERT_SAVEDMODEL_DIR  = "models/bert_jobcategory_savedmodel"
LABEL_ENCODER_FILE   = "models/bert_label_encoder.pkl"
MAX_LENGTH           = 128


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
# Custom components — harus didefinisikan ulang agar SavedModel bisa di-load
# ===========================================================================

class ClassificationHead(tf.keras.layers.Layer):
    def __init__(self, num_classes: int, dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate
        self.dropout      = tf.keras.layers.Dropout(dropout_rate)
        self.dense        = tf.keras.layers.Dense(
            num_classes,
            activation="softmax",
            name="classification_output",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, cls_output, training=False):
        x = self.dropout(cls_output, training=training)
        return self.dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "dropout_rate": self.dropout_rate})
        return config


class TFBertClassifier(tf.keras.Model):
    def __init__(self, bert_encoder, num_classes: int, dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.bert_encoder = bert_encoder
        self.cls_head     = ClassificationHead(num_classes, dropout_rate, name="cls_head")

    def call(self, inputs, training=False):
        bert_output = self.bert_encoder(inputs, training=training)
        cls_token   = bert_output.pooler_output
        return self.cls_head(cls_token, training=training)


# ===========================================================================
# Load model assets
# ===========================================================================

def load_inference_assets() -> dict:
    """
    Load TF-BERT model + tokenizer + label encoder untuk inference.
    Prioritas: SavedModel → weights checkpoint.
    """
    if not os.path.exists(LABEL_ENCODER_FILE):
        raise FileNotFoundError(
            f"Label encoder tidak ditemukan: {LABEL_ENCODER_FILE}\n"
            "Jalankan 3b_train_bert.py untuk melatih model terlebih dahulu."
        )

    with open(LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)

    log.info(f"Memuat tokenizer dari: {BERT_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)

    # ── Coba load SavedModel terlebih dahulu ──────────────────────────────
    if os.path.isdir(BERT_SAVEDMODEL_DIR):
        log.info(f"Memuat SavedModel dari: {BERT_SAVEDMODEL_DIR}")
        model = tf.keras.models.load_model(
            BERT_SAVEDMODEL_DIR,
            custom_objects={
                "TFBertClassifier": TFBertClassifier,
                "ClassificationHead": ClassificationHead,
            },
        )
        log.info("SavedModel berhasil di-load.")

    # ── Fallback ke weights checkpoint ────────────────────────────────────
    elif os.path.isdir(BERT_MODEL_DIR):
        weights_path = os.path.join(BERT_MODEL_DIR, "tf_bert_weights")
        log.info(f"SavedModel tidak ditemukan. Memuat weights dari: {weights_path}")
        bert_base = TFAutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_DIR,
            num_labels=num_classes,
        ).layers[0]
        model = TFBertClassifier(bert_base, num_classes)
        dummy = {
            "input_ids":      tf.zeros((1, MAX_LENGTH), dtype=tf.int32),
            "attention_mask": tf.zeros((1, MAX_LENGTH), dtype=tf.int32),
            "token_type_ids": tf.zeros((1, MAX_LENGTH), dtype=tf.int32),
        }
        _ = model(dummy, training=False)
        model.load_weights(weights_path)
        log.info("Weights berhasil di-load.")

    else:
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{BERT_SAVEDMODEL_DIR}' maupun '{BERT_MODEL_DIR}'.\n"
            "Jalankan 3b_train_bert.py untuk melatih model terlebih dahulu."
        )

    log.info(f"TF-BERT siap | Kelas: {num_classes}")
    return {
        "model":         model,
        "tokenizer":     tokenizer,
        "label_encoder": label_encoder,
    }


# ===========================================================================
# Inference
# ===========================================================================

def predict_top3_text(input_text: str, assets: dict) -> tuple[list, str]:
    """
    Prediksi Top-3 job category dari teks input menggunakan TF-BERT + auto-translate.

    Returns:
        tuple: (list of dict prediksi, str teks yang sudah di-translate/di-proses)
    """
    if not input_text or not input_text.strip():
        return [{"category": "Unknown", "confidence": 0.0}], input_text

    # 1. Deteksi bahasa & translasi ke English jika perlu
    translated_text = input_text
    try:
        if detect:
            lang = detect(input_text)
            if lang != "en":
                log.info(f"Deteksi bahasa: {lang}. Menerjemahkan ke English...")
                translated_text = translate_to_english(input_text)
        else:
            translated_text = translate_to_english(input_text)
    except Exception as e:
        log.warning(f"Gagal mendeteksi bahasa atau translasi: {e}. Menggunakan teks asli.")
        translated_text = input_text

    # 2. Preprocessing
    cleaned = safe_clean(translated_text)
    if not cleaned.strip():
        return [{"category": "Unknown", "confidence": 0.0}], translated_text

    model         = assets["model"]
    tokenizer     = assets["tokenizer"]
    label_encoder = assets["label_encoder"]

    # 3. Tokenisasi → tf.Tensor
    enc = tokenizer(
        cleaned,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )
    inputs = {k: tf.cast(v, tf.int32) for k, v in enc.items()}

    # 4. Inference
    output = model(inputs, training=False)

    # Handle TFSequenceClassifierOutput atau plain tensor
    if hasattr(output, "logits"):
        probs = tf.nn.softmax(output.logits, axis=-1).numpy().squeeze()
    else:
        probs = output.numpy().squeeze()

    top_k   = min(3, len(label_encoder.classes_))
    top_idx = np.argsort(probs)[-top_k:][::-1]

    results = [
        {
            "category":   label_encoder.inverse_transform([idx])[0],
            "confidence": float(probs[idx]),
        }
        for idx in top_idx
    ]

    return results or [{"category": "Unknown", "confidence": 0.0}], translated_text


def predict_from_file(file_path: str, assets: dict | None = None) -> dict:
    """
    Prediksi Top-3 job category dari file PDF/DOCX/TXT.
    """
    assets   = assets or load_inference_assets()
    raw_text = extract_text_from_file(file_path)
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
    text   = "Saya bisa pemrograman Python, SQL, machine learning dan membuat dashboard statistik"
    results, translated = predict_top3_text(text, assets)

    print("\n=== HASIL TRANSLASI ===")
    print(f"Original  : {text}")
    print(f"Translated: {translated}")
    print("\n=== TOP-3 JOB CATEGORY PREDICTION (TF-BERT) ===")
    for i, item in enumerate(results, start=1):
        print(f"Top {i}: {item['category']} → {item['confidence']:.4f}")


if __name__ == "__main__":
    main()
