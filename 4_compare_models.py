"""
4_compare_models.py
===================
Membandingkan model BiLSTM vs TF-IDF Logistic Regression.
Keduanya menggunakan kolom: text (input), label (target)
"""

import json
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences

from project_utils import top_k_accuracy_from_probs

DL_MODEL_FILE      = "models/model_jobcategory_dl.keras"
TOKENIZER_FILE     = "models/tokenizer.pkl"
LABEL_ENCODER_FILE = "models/label_encoder.pkl"
TRAIN_CONFIG_FILE  = "models/train_config.json"
BASELINE_MODEL_FILE = "models/jobcategory_tfidf_logreg.joblib"
TEST_FILE          = "data/test_split.csv"
VAL_FILE           = "data/val_split.csv"
RESULTS_DIR        = "results"
COMPARISON_JSON    = "results/model_comparison.json"
COMPARISON_CSV     = "results/model_comparison.csv"


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


def top_k_accuracy_baseline(model, X, y_true_raw, k=3):
    if not hasattr(model, "predict_proba"):
        return None
    probs   = model.predict_proba(X)
    classes = np.array(model.classes_)
    k       = min(k, len(classes))
    topk_idx    = np.argsort(-probs, axis=1)[:, :k]
    topk_labels = classes[topk_idx].astype(str)
    y_true_arr  = np.array(y_true_raw).astype(str)
    return float(np.mean([true in row for true, row in zip(y_true_arr, topk_labels)]))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    required = [DL_MODEL_FILE, TOKENIZER_FILE, LABEL_ENCODER_FILE, TRAIN_CONFIG_FILE, BASELINE_MODEL_FILE]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "File berikut belum ada: " + ", ".join(missing) +
            "\nJalankan 2_train_deep_learning.py dan 5_train_baseline_tfidf.py dulu."
        )

    eval_df, split_name = load_eval_split()
    if len(eval_df) == 0:
        msg = "Tidak ada test/validation set. Perbandingan model dilewati."
        print("[WARNING]", msg)
        with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
            json.dump({"comparison_created": False, "reason": msg}, f, indent=2)
        pd.DataFrame(columns=["model", "eval_split", "target", "accuracy", "top_3_accuracy"])\
            .to_csv(COMPARISON_CSV, index=False)
        return

    eval_df    = prepare_eval_dataframe(eval_df)
    X_eval     = eval_df["text"].astype(str)
    y_true_raw = eval_df["label"].astype(str)

    # --- Deep Learning ---
    dl_model = tf.keras.models.load_model(DL_MODEL_FILE, compile=False)
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_FILE, "rb") as f:
        label_encoder = pickle.load(f)
    with open(TRAIN_CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    known_mask = y_true_raw.isin(set(label_encoder.classes_))
    if not known_mask.all():
        removed = int((~known_mask).sum())
        print(f"[WARNING] {removed} baris dibuang karena label tidak ada di encoder DL.")
        eval_df    = eval_df[known_mask].reset_index(drop=True)
        X_eval     = eval_df["text"].astype(str)
        y_true_raw = eval_df["label"].astype(str)

    if len(eval_df) == 0:
        raise ValueError("Tidak ada data evaluasi yang cocok dengan label encoder.")

    max_length = config.get("max_length", 400)
    y_true_int = label_encoder.transform(y_true_raw)
    eval_pad   = pad_sequences(
        tokenizer.texts_to_sequences(X_eval),
        maxlen=max_length, padding="post", truncating="post"
    )
    dl_probs = dl_model.predict(eval_pad, verbose=0)
    dl_pred  = label_encoder.inverse_transform(np.argmax(dl_probs, axis=1))

    top_k    = min(3, len(label_encoder.classes_))
    dl_acc   = float(accuracy_score(y_true_raw, dl_pred))
    dl_top3  = float(top_k_accuracy_from_probs(dl_probs, y_true_int, k=top_k))

    # --- Baseline ---
    baseline_model = joblib.load(BASELINE_MODEL_FILE)
    base_pred      = baseline_model.predict(X_eval).astype(str)
    base_acc       = float(accuracy_score(y_true_raw, base_pred))
    base_top3      = top_k_accuracy_baseline(baseline_model, X_eval, y_true_raw, k=top_k)

    comparison = [
        {
            "model": "BiLSTM Deep Learning",
            "eval_split": split_name,
            "target": "label (job_category)",
            "accuracy": dl_acc,
            f"top_{top_k}_accuracy": dl_top3,
        },
        {
            "model": "TF-IDF + Logistic Regression",
            "eval_split": split_name,
            "target": "label (job_category)",
            "accuracy": base_acc,
            f"top_{top_k}_accuracy": None if base_top3 is None else float(base_top3),
        },
    ]

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8")

    details = {
        "comparison_created": True,
        "eval_split_used": split_name,
        "target": "label (job_category)",
        "input_column": "text",
        "comparison": comparison,
        "deep_learning_classification_report":
            classification_report(y_true_raw, dl_pred, zero_division=0),
        "baseline_classification_report":
            classification_report(y_true_raw, base_pred, zero_division=0),
    }
    with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("MODEL COMPARISON")
    print("Target: label (job_category)")
    print("==============================")
    print(comparison_df.to_string(index=False))
    print(f"\n[INFO] CSV  disimpan ke: {COMPARISON_CSV}")
    print(f"[INFO] JSON disimpan ke: {COMPARISON_JSON}")


if __name__ == "__main__":
    main()
