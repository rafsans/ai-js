"""
4b_evaluate_bert.py
===================
Evaluasi model TF-BERT (TFBertClassifier) yang sudah ditraining oleh 3b_train_bert.py.

Output:
  results/bert_eval_report.txt          — laporan teks (accuracy, top-3, MAE, classification report)
  results/bert_eval_predictions.csv     — prediksi per sample (top-3 kategori + probabilitas)
  results/bert_confusion_matrix.png     — confusion matrix
  results/bert_eval_summary.json        — ringkasan metrik dalam format JSON

Cara pakai:
  python 4b_evaluate_bert.py                    # pakai test set (default)
  python 4b_evaluate_bert.py --split val        # pakai validation set
  python 4b_evaluate_bert.py --split both       # evaluasi keduanya
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# ============================================================================
# Paths — harus sesuai dengan output 3b_train_bert.py
# ============================================================================
BERT_MODEL_DIR      = "models/bert_jobcategory"
BERT_SAVEDMODEL_DIR = "models/bert_jobcategory_savedmodel"
LABEL_ENCODER_FILE  = "models/bert_label_encoder.pkl"
BERT_CONFIG_FILE    = "models/bert_train_config.json"

TEST_FILE           = "data/test_split.csv"
VAL_FILE            = "data/val_split.csv"

RESULTS_DIR           = "results"
REPORT_FILE           = "results/bert_eval_report.txt"
PREDICTION_FILE       = "results/bert_eval_predictions.csv"
CONFUSION_MATRIX_FILE = "results/bert_confusion_matrix.png"
EVAL_SUMMARY_FILE     = "results/bert_eval_summary.json"

# Default config (digunakan bila bert_train_config.json tidak ditemukan)
MAX_LENGTH_DEFAULT  = 256
BATCH_SIZE_DEFAULT  = 16
PRETRAINED_DEFAULT  = "bert-base-uncased"


# ============================================================================
# Custom components — WAJIB didefinisikan ulang agar SavedModel bisa dimuat
# ============================================================================

class ClassificationHead(tf.keras.layers.Layer):
    """
    Custom classification head di atas [CLS] token BERT.
    Identik dengan definisi di 3b_train_bert.py.
    """
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


class LabelSmoothingCategoricalCrossentropy(tf.keras.losses.Loss):
    """Custom loss dengan label smoothing. Diperlukan saat memuat SavedModel."""
    def __init__(self, smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true      = tf.cast(y_true, tf.float32)
        smooth_true = y_true * (1.0 - self.smoothing) + (self.smoothing / num_classes)
        return tf.reduce_mean(
            -tf.reduce_sum(smooth_true * tf.math.log(tf.clip_by_value(y_pred, 1e-9, 1.0)), axis=-1)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"smoothing": self.smoothing})
        return config


class TFBertClassifier(tf.keras.Model):
    """
    Model Subclassing: BERT + ClassificationHead kustom.
    Diperlukan untuk merekonstruksi model saat load dari weights.
    """
    def __init__(self, bert_encoder, num_classes: int, dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.bert_encoder = bert_encoder
        self.cls_head     = ClassificationHead(num_classes, dropout_rate, name="cls_head")

    def call(self, inputs, training=False):
        bert_output = self.bert_encoder(inputs, training=training)
        cls_token   = bert_output.pooler_output
        return self.cls_head(cls_token, training=training)

    def get_config(self):
        return super().get_config()


# ============================================================================
# Helpers
# ============================================================================

def top_k_accuracy_from_probs(probs: np.ndarray, y_true_int: np.ndarray, k: int = 3) -> float:
    """Hitung Top-K Accuracy dari matriks probabilitas."""
    top_k_indices = np.argsort(-probs, axis=1)[:, :k]
    correct = sum(
        y_true_int[i] in top_k_indices[i]
        for i in range(len(y_true_int))
    )
    return correct / len(y_true_int) if len(y_true_int) > 0 else 0.0


def mae_confidence(probs: np.ndarray, y_true_int: np.ndarray) -> float:
    """
    MAE berbasis probabilitas untuk klasifikasi.
    Rumus: rata-rata |1.0 - P(kelas_benar)| per sample.
    Interpretasi:
      - Mendekati 0.0 → model sangat yakin pada prediksi yang benar
      - Mendekati 1.0 → model tidak yakin sama sekali
    """
    correct_class_probs = probs[np.arange(len(y_true_int)), y_true_int]
    return float(np.mean(np.abs(1.0 - correct_class_probs)))


def load_config() -> dict:
    """Muat konfigurasi training; gunakan default jika file tidak ada."""
    if os.path.exists(BERT_CONFIG_FILE):
        with open(BERT_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[WARNING] {BERT_CONFIG_FILE} tidak ditemukan. Menggunakan nilai default.")
    return {
        "max_length":       MAX_LENGTH_DEFAULT,
        "batch_size":       BATCH_SIZE_DEFAULT,
        "pretrained_model": PRETRAINED_DEFAULT,
        "num_classes":      None,
    }


def load_model_and_artifacts(config: dict, num_classes: int):
    """
    Muat model, tokenizer, dan label encoder.
    Strategi:
      1. Coba muat dari SavedModel (lebih cepat, tidak perlu rebuild)
      2. Jika gagal, rekonstruksi dari weights + tokenizer
    """
    pretrained = config.get("pretrained_model", PRETRAINED_DEFAULT)

    # ── Load Label Encoder ────────────────────────────────────────────────
    if not os.path.exists(LABEL_ENCODER_FILE):
        raise FileNotFoundError(
            f"Label encoder tidak ditemukan: {LABEL_ENCODER_FILE}\n"
            "Pastikan 3b_train_bert.py sudah dijalankan."
        )
    with open(LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"[INFO] Label encoder dimuat: {len(label_encoder.classes_)} kelas")

    # ── Load Tokenizer ────────────────────────────────────────────────────
    tokenizer_path = BERT_MODEL_DIR if os.path.isdir(BERT_MODEL_DIR) else pretrained
    print(f"[INFO] Memuat tokenizer dari: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ── Load Model ────────────────────────────────────────────────────────
    custom_objects = {
        "TFBertClassifier":                      TFBertClassifier,
        "ClassificationHead":                    ClassificationHead,
        "LabelSmoothingCategoricalCrossentropy": LabelSmoothingCategoricalCrossentropy,
    }

    # Strategi 1: SavedModel
    if os.path.isdir(BERT_SAVEDMODEL_DIR):
        try:
            print(f"[INFO] Memuat SavedModel dari: {BERT_SAVEDMODEL_DIR}")
            model = tf.keras.models.load_model(
                BERT_SAVEDMODEL_DIR,
                custom_objects=custom_objects,
                compile=False,
            )
            print("[INFO] SavedModel berhasil dimuat.")
            return model, tokenizer, label_encoder
        except Exception as e:
            print(f"[WARNING] Gagal muat SavedModel: {e}")
            print("[INFO] Mencoba rekonstruksi dari weights...")

    # Strategi 2: Rekonstruksi dari weights
    weights_path = os.path.join(BERT_MODEL_DIR, "tf_bert_weights")
    if not os.path.exists(weights_path + ".index"):
        raise FileNotFoundError(
            "Tidak ditemukan SavedModel maupun weights.\n"
            f"  Dicari: {BERT_SAVEDMODEL_DIR} atau {weights_path}\n"
            "Pastikan 3b_train_bert.py sudah dijalankan."
        )

    print(f"[INFO] Merekonstruksi TFBertClassifier dari: {pretrained}")
    bert_encoder = TFAutoModelForSequenceClassification.from_pretrained(
        pretrained,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    ).layers[0]

    model = TFBertClassifier(
        bert_encoder=bert_encoder,
        num_classes=num_classes,
        dropout_rate=0.3,
        name="tf_bert_classifier",
    )

    # Build dengan dummy input sebelum load weights
    max_len = config.get("max_length", MAX_LENGTH_DEFAULT)
    dummy = {
        "input_ids":      tf.zeros((1, max_len), dtype=tf.int32),
        "attention_mask": tf.zeros((1, max_len), dtype=tf.int32),
        "token_type_ids": tf.zeros((1, max_len), dtype=tf.int32),
    }
    _ = model(dummy, training=False)
    model.load_weights(weights_path)
    print(f"[INFO] Weights dimuat dari: {weights_path}")

    return model, tokenizer, label_encoder


def prepare_dataframe(df: pd.DataFrame, label_encoder) -> pd.DataFrame:
    """Bersihkan dataframe dan filter label yang tidak dikenal encoder."""
    df = df.copy()
    df["text"]  = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df = df[df["label"] != ""].reset_index(drop=True)

    known_mask = df["label"].isin(set(label_encoder.classes_))
    removed    = int((~known_mask).sum())
    if removed:
        print(f"[WARNING] {removed} baris dibuang karena label tidak dikenal encoder.")
        df = df[known_mask].reset_index(drop=True)

    return df


def tokenize_batch(texts: list, tokenizer, max_length: int, batch_size: int) -> tf.data.Dataset:
    """Tokenisasi teks menjadi tf.data.Dataset tanpa label (untuk inferensi)."""
    enc = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ds = tf.data.Dataset.from_tensor_slices({
        "input_ids":      enc["input_ids"].astype(np.int32),
        "attention_mask": enc["attention_mask"].astype(np.int32),
        "token_type_ids": enc.get(
            "token_type_ids",
            np.zeros_like(enc["input_ids"])
        ).astype(np.int32),
    })
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def run_inference(model, ds: tf.data.Dataset) -> np.ndarray:
    """Jalankan inferensi dan kembalikan matriks probabilitas."""
    all_probs = []
    for batch_inputs in ds:
        preds = model(batch_inputs, training=False)
        if hasattr(preds, "logits"):
            probs = tf.nn.softmax(preds.logits, axis=-1).numpy()
        else:
            probs = preds.numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def evaluate_split(
    model,
    tokenizer,
    label_encoder,
    eval_df: pd.DataFrame,
    split_name: str,
    config: dict,
    top_k: int,
) -> dict:
    """
    Evaluasi model pada satu split data.
    Mengembalikan dict berisi semua metrik dan hasil prediksi.
    """
    max_length = config.get("max_length", MAX_LENGTH_DEFAULT)
    batch_size = config.get("batch_size", BATCH_SIZE_DEFAULT)

    texts      = eval_df["text"].tolist()
    y_true_raw = eval_df["label"].tolist()
    y_true_int = label_encoder.transform(y_true_raw)

    print(f"[INFO] Tokenisasi {len(texts)} sample...")
    ds    = tokenize_batch(texts, tokenizer, max_length, batch_size)
    probs = run_inference(model, ds)

    # Potong probs sesuai jumlah sample aktual (batch terakhir bisa di-pad)
    probs = probs[:len(texts)]

    y_pred_int = np.argmax(probs, axis=1)
    y_pred_raw = label_encoder.inverse_transform(y_pred_int)

    acc       = accuracy_score(y_true_raw, y_pred_raw)
    top_k_acc = top_k_accuracy_from_probs(probs, y_true_int, k=top_k)
    mae       = mae_confidence(probs, y_true_int)
    report    = classification_report(y_true_raw, y_pred_raw, zero_division=0)

    # Ringkasan per kelas: precision, recall, f1
    report_dict = classification_report(
        y_true_raw, y_pred_raw, zero_division=0, output_dict=True
    )

    # Bangun DataFrame prediksi
    top_indices = np.argsort(-probs, axis=1)[:, :top_k]
    pred_df     = eval_df[["text", "label"]].copy()
    pred_df["split"]            = split_name
    pred_df["y_true"]           = y_true_raw
    pred_df["y_pred"]           = y_pred_raw
    pred_df["correct"]          = pred_df["y_true"] == pred_df["y_pred"]
    pred_df["top1_probability"] = probs.max(axis=1)

    for rank in range(top_k):
        idxs = top_indices[:, rank]
        pred_df[f"top{rank+1}_category"]    = label_encoder.inverse_transform(idxs)
        pred_df[f"top{rank+1}_probability"] = probs[np.arange(len(probs)), idxs]

    return {
        "split_name":  split_name,
        "n_samples":   len(eval_df),
        "accuracy":    acc,
        "top_k_acc":   top_k_acc,
        "mae":         mae,
        "top_k":       top_k,
        "report_str":  report,
        "report_dict": report_dict,
        "pred_df":     pred_df,
        "y_true_raw":  y_true_raw,
        "y_pred_raw":  y_pred_raw,
    }


def save_confusion_matrix(y_true, y_pred, label_encoder, split_name: str, out_path: str):
    """Simpan confusion matrix sebagai gambar PNG."""
    labels = sorted(
        set(y_true) | set(y_pred),
        key=lambda x: list(label_encoder.classes_).index(x)
        if x in label_encoder.classes_ else 999
    )
    if not labels:
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n  = len(labels)
    fig_size = max(10, n * 0.75)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=90, colorbar=False
    )
    ax.set_title(
        f"Confusion Matrix — TFBertClassifier ({split_name})",
        fontsize=14, pad=14
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Confusion matrix disimpan ke: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi TFBertClassifier (output dari 3b_train_bert.py)"
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "both"],
        default="test",
        help="Split yang dievaluasi: test (default), val, atau both",
    )
    # parse_known_args() agar tidak crash saat dijalankan di Jupyter/Colab
    args, _ = parser.parse_known_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load konfigurasi ──────────────────────────────────────────────────
    config      = load_config()
    num_classes = config.get("num_classes")

    # ── Load model & artifacts ────────────────────────────────────────────
    if num_classes is None:
        with open(LABEL_ENCODER_FILE, "rb") as f:
            _le = pickle.load(f)
        num_classes = len(_le.classes_)

    model, tokenizer, label_encoder = load_model_and_artifacts(config, num_classes)
    top_k = min(3, num_classes)

    # ── Tentukan split yang akan dievaluasi ───────────────────────────────
    splits_to_eval = []

    if args.split in ("test", "both"):
        if os.path.exists(TEST_FILE):
            raw = pd.read_csv(TEST_FILE)
            df  = prepare_dataframe(raw, label_encoder)
            if len(df) > 0:
                splits_to_eval.append((df, "test"))
            else:
                print("[WARNING] Test set kosong setelah filtering — dilewati.")
        else:
            print(f"[WARNING] File tidak ditemukan: {TEST_FILE}")

    if args.split in ("val", "both"):
        if os.path.exists(VAL_FILE):
            raw = pd.read_csv(VAL_FILE)
            df  = prepare_dataframe(raw, label_encoder)
            if len(df) > 0:
                splits_to_eval.append((df, "validation"))
            else:
                print("[WARNING] Validation set kosong setelah filtering — dilewati.")
        else:
            print(f"[WARNING] File tidak ditemukan: {VAL_FILE}")

    if not splits_to_eval:
        print("[ERROR] Tidak ada split yang bisa dievaluasi. Periksa path data.")
        sys.exit(1)

    # ── Evaluasi setiap split ─────────────────────────────────────────────
    all_results  = []
    all_pred_dfs = []

    for eval_df, split_name in splits_to_eval:
        print(f"\n{'='*52}")
        print(f"  EVALUASI SPLIT: {split_name.upper()} ({len(eval_df)} sample)")
        print(f"{'='*52}")

        result = evaluate_split(
            model, tokenizer, label_encoder,
            eval_df, split_name, config, top_k
        )

        # Cetak metrik utama
        print(f"\n  Accuracy        : {result['accuracy']:.4f}")
        print(f"  Top-{top_k} Accuracy  : {result['top_k_acc']:.4f}")
        print(f"  MAE             : {result['mae']:.4f}  (confidence error, ↓ lebih baik)")
        print(f"\n  Classification Report:\n")
        print(result["report_str"])

        all_results.append(result)
        all_pred_dfs.append(result["pred_df"])

        # Simpan confusion matrix per split
        cm_path = (
            CONFUSION_MATRIX_FILE
            if len(splits_to_eval) == 1
            else f"results/bert_confusion_matrix_{split_name}.png"
        )
        save_confusion_matrix(
            result["y_true_raw"],
            result["y_pred_raw"],
            label_encoder,
            split_name,
            cm_path,
        )

    # ── Simpan laporan teks ───────────────────────────────────────────────
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  TF-BERT (TFBertClassifier) — EVALUATION REPORT\n")
        f.write(f"  Pretrained : {config.get('pretrained_model', PRETRAINED_DEFAULT)}\n")
        f.write(f"  Task       : Job Category Classification\n")
        f.write(f"  Max Length : {config.get('max_length', MAX_LENGTH_DEFAULT)}\n")
        f.write("=" * 60 + "\n\n")

        for res in all_results:
            header = f"SPLIT: {res['split_name'].upper()} — {res['n_samples']} samples"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            f.write(f"Accuracy        : {res['accuracy']:.4f}\n")
            f.write(f"Top-{res['top_k']} Accuracy  : {res['top_k_acc']:.4f}\n")
            f.write(f"MAE             : {res['mae']:.4f}  (confidence error, lebih kecil lebih baik)\n\n")
            f.write("Classification Report:\n")
            f.write(res["report_str"])
            f.write("\n\n")

    print(f"\n[INFO] Laporan disimpan ke: {REPORT_FILE}")

    # ── Simpan CSV prediksi ───────────────────────────────────────────────
    combined_pred = pd.concat(all_pred_dfs, ignore_index=True)
    combined_pred.to_csv(PREDICTION_FILE, index=False, encoding="utf-8")
    print(f"[INFO] Prediksi disimpan ke: {PREDICTION_FILE}")

    # ── Simpan ringkasan JSON ─────────────────────────────────────────────
    summary = {
        "model_type":       "TFBertClassifier (Model Subclassing)",
        "pretrained_model": config.get("pretrained_model", PRETRAINED_DEFAULT),
        "num_classes":      num_classes,
        "max_length":       config.get("max_length", MAX_LENGTH_DEFAULT),
        "label_classes":    label_encoder.classes_.tolist(),
        "evaluations": [
            {
                "split":                   res["split_name"],
                "n_samples":               res["n_samples"],
                "accuracy":                round(res["accuracy"], 6),
                f"top_{top_k}_accuracy":   round(res["top_k_acc"], 6),
                "mae":                     round(res["mae"], 6),
                "macro_f1":                round(
                    res["report_dict"].get("macro avg", {}).get("f1-score", 0.0), 6
                ),
                "weighted_f1":             round(
                    res["report_dict"].get("weighted avg", {}).get("f1-score", 0.0), 6
                ),
            }
            for res in all_results
        ],
    }

    with open(EVAL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Ringkasan JSON disimpan ke: {EVAL_SUMMARY_FILE}")

    # ── Cetak ringkasan akhir ─────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  RINGKASAN EVALUASI BERT")
    print("=" * 52)
    for ev in summary["evaluations"]:
        print(f"  [{ev['split'].upper()}]")
        print(f"    Accuracy        : {ev['accuracy']:.4f}")
        print(f"    Top-{top_k} Accuracy  : {ev[f'top_{top_k}_accuracy']:.4f}")
        print(f"    MAE             : {ev['mae']:.4f}")
        print(f"    Macro F1        : {ev['macro_f1']:.4f}")
        print(f"    Weighted F1     : {ev['weighted_f1']:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
