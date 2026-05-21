"""
3_evaluate_deep_learning.py
===========================
Evaluasi model BiLSTM Job Category Classifier.
Membaca kolom: text, label
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score,
    classification_report, confusion_matrix
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

from project_utils import top_k_accuracy_from_probs

MODEL_FILE         = "models/model_jobcategory_dl.keras"
TOKENIZER_FILE     = "models/tokenizer.pkl"
LABEL_ENCODER_FILE = "models/label_encoder.pkl"
TRAIN_CONFIG_FILE  = "models/train_config.json"
TEST_FILE          = "data/test_split.csv"
VAL_FILE           = "data/val_split.csv"
RESULTS_DIR        = "results"
REPORT_FILE        = "results/deep_learning_report.txt"
PREDICTION_FILE    = "results/deep_learning_test_predictions.csv"
CONFUSION_MATRIX_FILE = "results/confusion_matrix_deep_learning.png"


def load_eval_split():
    if os.path.exists(TEST_FILE):
        test_df = pd.read_csv(TEST_FILE)
        if len(test_df) > 0:
            return test_df, "test"
    if os.path.exists(VAL_FILE):
        val_df = pd.read_csv(VAL_FILE)
        if len(val_df) > 0:
            return val_df, "validation"
    return pd.DataFrame(), "empty"


def prepare_eval_dataframe(eval_df):
    if "text" not in eval_df.columns:
        raise ValueError("Kolom 'text' tidak ditemukan di eval split.")
    if "label" not in eval_df.columns:
        raise ValueError("Kolom 'label' tidak ditemukan di eval split.")
    eval_df = eval_df.copy()
    eval_df["text"]  = eval_df["text"].fillna("").astype(str)
    eval_df["label"] = eval_df["label"].fillna("").astype(str).str.strip()
    return eval_df[eval_df["label"] != ""].reset_index(drop=True)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    required = [MODEL_FILE, TOKENIZER_FILE, LABEL_ENCODER_FILE, TRAIN_CONFIG_FILE]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "File model belum lengkap. Jalankan 2_train_deep_learning.py dulu.\n"
            "File yang kurang: " + ", ".join(missing)
        )

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)
    with open(TRAIN_CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    max_length = config.get("max_length", 400)
    eval_df, split_name = load_eval_split()

    if len(eval_df) == 0:
        msg = "Test dan validation set kosong. Evaluasi dilewati."
        print("[WARNING]", msg)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(msg)
        return

    eval_df  = prepare_eval_dataframe(eval_df)
    X_eval   = eval_df["text"].astype(str)
    y_true_raw = eval_df["label"].astype(str)

    # Hapus label yang tidak dikenal encoder
    known_mask = y_true_raw.isin(set(label_encoder.classes_))
    if not known_mask.all():
        removed = int((~known_mask).sum())
        print(f"[WARNING] {removed} baris dibuang karena label tidak ada di encoder.")
        eval_df    = eval_df[known_mask].reset_index(drop=True)
        X_eval     = eval_df["text"].astype(str)
        y_true_raw = eval_df["label"].astype(str)

    if len(eval_df) == 0:
        raise ValueError("Tidak ada data evaluasi dengan label yang dikenal encoder.")

    y_true_int = label_encoder.transform(y_true_raw)
    eval_pad   = pad_sequences(
        tokenizer.texts_to_sequences(X_eval),
        maxlen=max_length, padding="post", truncating="post"
    )

    probs      = model.predict(eval_pad, verbose=0)
    y_pred_int = np.argmax(probs, axis=1)
    y_pred_raw = label_encoder.inverse_transform(y_pred_int)

    acc    = accuracy_score(y_true_raw, y_pred_raw)
    top_k  = min(3, len(label_encoder.classes_))
    top3_acc = top_k_accuracy_from_probs(probs, y_true_int, k=top_k)
    report = classification_report(y_true_raw, y_pred_raw, zero_division=0)

    print("\n======================================")
    print(f"DEEP LEARNING {split_name.upper()} EVALUATION")
    print("Target: label (job_category)")
    print("======================================")
    print("Accuracy     :", round(acc, 4))
    print(f"Top-{top_k} Accuracy:", round(top3_acc, 4))
    print(report)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== DEEP LEARNING {split_name.upper()} EVALUATION ===\n")
        f.write("Target: label (job_category)\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Top-{top_k} Accuracy: {top3_acc}\n\n")
        f.write(report)

    pred_df = eval_df[["text", "label"]].copy()
    pred_df["eval_split"]       = split_name
    pred_df["y_true"]           = y_true_raw.values
    pred_df["y_pred"]           = y_pred_raw
    pred_df["pred_probability"] = probs.max(axis=1)
    top_indices = np.argsort(-probs, axis=1)[:, :top_k]
    for rank in range(top_k):
        idxs = top_indices[:, rank]
        pred_df[f"top_{rank+1}_category"]    = label_encoder.inverse_transform(idxs)
        pred_df[f"top_{rank+1}_probability"] = probs[np.arange(len(probs)), idxs]
    pred_df.to_csv(PREDICTION_FILE, index=False, encoding="utf-8")

    labels = [l for l in label_encoder.classes_
              if l in set(y_true_raw) or l in set(y_pred_raw)]
    if labels:
        cm = confusion_matrix(y_true_raw, y_pred_raw, labels=labels)
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), max(10, len(labels) * 0.7)))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
            ax=ax, xticks_rotation=90, colorbar=False
        )
        plt.title("Confusion Matrix - Job Category Classifier (BiLSTM)")
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_FILE, dpi=200)
        plt.close(fig)
        print(f"[INFO] Confusion matrix disimpan ke: {CONFUSION_MATRIX_FILE}")

    print(f"[INFO] Report disimpan ke         : {REPORT_FILE}")
    print(f"[INFO] Prediksi disimpan ke       : {PREDICTION_FILE}")


if __name__ == "__main__":
    main()
