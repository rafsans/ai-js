
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

DL_MODEL_FILE       = "models/model_jobcategory_dl.keras"
TOKENIZER_FILE      = "models/tokenizer.pkl"
LABEL_ENCODER_FILE  = "models/label_encoder.pkl"
TRAIN_CONFIG_FILE   = "models/train_config.json"
BASELINE_MODEL_FILE = "models/jobcategory_tfidf_logreg.joblib"
BERT_MODEL_DIR      = "models/bert_jobcategory"
BERT_LABEL_ENCODER  = "models/bert_label_encoder.pkl"
BERT_CONFIG_FILE    = "models/bert_train_config.json"
TEST_FILE           = "data/test_split.csv"
VAL_FILE            = "data/val_split.csv"
RESULTS_DIR         = "results"
COMPARISON_JSON     = "results/model_comparison.json"
COMPARISON_CSV      = "results/model_comparison.csv"


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
    probs       = model.predict_proba(X)
    classes     = np.array(model.classes_)
    k           = min(k, len(classes))
    topk_idx    = np.argsort(-probs, axis=1)[:, :k]
    topk_labels = classes[topk_idx].astype(str)
    y_true_arr  = np.array(y_true_raw).astype(str)
    return float(np.mean([true in row for true, row in zip(y_true_arr, topk_labels)]))


def run_bilstm(X_eval, y_true_raw, top_k):
    required = [DL_MODEL_FILE, TOKENIZER_FILE, LABEL_ENCODER_FILE, TRAIN_CONFIG_FILE]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        print("[WARNING] BiLSTM — file tidak ditemukan:", missing)
        return None

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
        print(f"[WARNING] BiLSTM: {removed} baris dibuang — label tidak dikenal encoder.")
        X_eval     = X_eval[known_mask].reset_index(drop=True)
        y_true_raw = y_true_raw[known_mask].reset_index(drop=True)

    if len(X_eval) == 0:
        print("[WARNING] BiLSTM: tidak ada data evaluasi setelah filter label.")
        return None

    max_length = config.get("max_length", 400)
    y_true_int = label_encoder.transform(y_true_raw)
    eval_pad   = pad_sequences(
        tokenizer.texts_to_sequences(X_eval),
        maxlen=max_length, padding="post", truncating="post",
    )
    dl_probs = dl_model.predict(eval_pad, verbose=0)
    dl_pred  = label_encoder.inverse_transform(np.argmax(dl_probs, axis=1))

    return {
        "accuracy":              float(accuracy_score(y_true_raw, dl_pred)),
        f"top_{top_k}_accuracy": float(top_k_accuracy_from_probs(dl_probs, y_true_int, k=top_k)),
        "cls_report":            classification_report(y_true_raw, dl_pred, zero_division=0),
        "n_eval":                len(X_eval),
    }


def run_baseline(X_eval, y_true_raw, top_k):
    if not os.path.exists(BASELINE_MODEL_FILE):
        print("[WARNING] Baseline — model tidak ditemukan:", BASELINE_MODEL_FILE)
        return None

    baseline_model = joblib.load(BASELINE_MODEL_FILE)
    base_pred      = baseline_model.predict(X_eval).astype(str)
    base_top3      = top_k_accuracy_baseline(baseline_model, X_eval, y_true_raw, k=top_k)

    return {
        "accuracy":              float(accuracy_score(y_true_raw, base_pred)),
        f"top_{top_k}_accuracy": None if base_top3 is None else float(base_top3),
        "cls_report":            classification_report(y_true_raw, base_pred, zero_division=0),
        "n_eval":                len(X_eval),
    }


def run_bert(X_eval, y_true_raw, top_k):
    required = [BERT_MODEL_DIR, BERT_LABEL_ENCODER, BERT_CONFIG_FILE]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        print("[WARNING] BERT — file tidak ditemukan:", missing)
        print("          Jalankan dulu: python 3b_train_bert.py")
        return None

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("[WARNING] BERT — transformers/torch belum terinstall.")
        print("          pip install transformers torch")
        return None

    with open(BERT_LABEL_ENCODER, "rb") as f:
        label_encoder = pickle.load(f)
    with open(BERT_CONFIG_FILE, "r", encoding="utf-8") as f:
        bert_config = json.load(f)

    max_length = bert_config.get("max_length", 256)

    known_mask = y_true_raw.isin(set(label_encoder.classes_))
    if not known_mask.all():
        removed = int((~known_mask).sum())
        print(f"[WARNING] BERT: {removed} baris dibuang — label tidak dikenal encoder.")
        X_eval     = X_eval[known_mask].reset_index(drop=True)
        y_true_raw = y_true_raw[known_mask].reset_index(drop=True)

    if len(X_eval) == 0:
        print("[WARNING] BERT: tidak ada data evaluasi setelah filter label.")
        return None

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    bert_model.to(device)
    bert_model.eval()

    batch_size = 32
    all_logits = []
    texts      = X_eval.tolist()

    print(f"[INFO] BERT inference ({len(texts)} sampel, device={device})...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = bert_model(**enc).logits
        all_logits.append(logits.cpu().numpy())

    logits_np     = np.concatenate(all_logits, axis=0)
    preds_int     = np.argmax(logits_np, axis=1)
    probs         = __import__("torch").softmax(
        __import__("torch").tensor(logits_np, dtype=__import__("torch").float32), dim=1
    ).numpy()
    y_pred_labels = label_encoder.inverse_transform(preds_int)
    y_true_int    = label_encoder.transform(y_true_raw)

    return {
        "accuracy":              float(accuracy_score(y_true_raw, y_pred_labels)),
        f"top_{top_k}_accuracy": float(top_k_accuracy_from_probs(probs, y_true_int, k=top_k)),
        "cls_report":            classification_report(y_true_raw, y_pred_labels, zero_division=0),
        "n_eval":                len(X_eval),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    eval_df, split_name = load_eval_split()
    if len(eval_df) == 0:
        msg = "Tidak ada test/validation set. Perbandingan model dilewati."
        print("[WARNING]", msg)
        with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
            json.dump({"comparison_created": False, "reason": msg}, f, indent=2)
        pd.DataFrame(columns=["model", "eval_split", "target", "n_eval", "accuracy", "top_3_accuracy"])\
            .to_csv(COMPARISON_CSV, index=False)
        return

    eval_df    = prepare_eval_dataframe(eval_df)
    X_eval     = eval_df["text"].astype(str)
    y_true_raw = eval_df["label"].astype(str)
    top_k      = 3

    print(f"\n[INFO] Evaluasi pada split: {split_name} ({len(eval_df)} baris)")

    print("\n[1/3] Evaluasi BiLSTM...")
    dl_result   = run_bilstm(X_eval.copy(), y_true_raw.copy(), top_k)

    print("\n[2/3] Evaluasi TF-IDF + Logistic Regression...")
    base_result = run_baseline(X_eval.copy(), y_true_raw.copy(), top_k)

    print("\n[3/3] Evaluasi BERT...")
    bert_result = run_bert(X_eval.copy(), y_true_raw.copy(), top_k)

    top_k_col  = f"top_{top_k}_accuracy"
    comparison = []

    if dl_result:
        comparison.append({
            "model":    "BiLSTM Deep Learning",
            "eval_split": split_name,
            "target":   "label (job_category)",
            "n_eval":   dl_result["n_eval"],
            "accuracy": dl_result["accuracy"],
            top_k_col:  dl_result[top_k_col],
        })

    if base_result:
        comparison.append({
            "model":    "TF-IDF + Logistic Regression",
            "eval_split": split_name,
            "target":   "label (job_category)",
            "n_eval":   base_result["n_eval"],
            "accuracy": base_result["accuracy"],
            top_k_col:  base_result[top_k_col],
        })

    if bert_result:
        comparison.append({
            "model":    "BERT (bert-base-uncased)",
            "eval_split": split_name,
            "target":   "label (job_category)",
            "n_eval":   bert_result["n_eval"],
            "accuracy": bert_result["accuracy"],
            top_k_col:  bert_result[top_k_col],
        })

    if not comparison:
        print("[ERROR] Tidak ada model yang berhasil dievaluasi.")
        return

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8")

    details = {
        "comparison_created": True,
        "eval_split_used":    split_name,
        "target":             "label (job_category)",
        "input_column":       "text",
        "comparison":         comparison,
    }
    if dl_result:
        details["bilstm_classification_report"]   = dl_result["cls_report"]
    if base_result:
        details["baseline_classification_report"] = base_result["cls_report"]
    if bert_result:
        details["bert_classification_report"]     = bert_result["cls_report"]

    with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("MODEL COMPARISON")
    print("Target: label (job_category)")
    print("==============================")
    print(comparison_df.to_string(index=False))

    best_idx   = comparison_df["accuracy"].idxmax()
    best_model = comparison_df.loc[best_idx, "model"]
    best_acc   = comparison_df.loc[best_idx, "accuracy"]
    print(f"\n Model terbaik (accuracy): {best_model} ({best_acc:.4f})")

    print(f"\n[INFO] CSV  disimpan ke: {COMPARISON_CSV}")
    print(f"[INFO] JSON disimpan ke: {COMPARISON_JSON}")


if __name__ == "__main__":
    main()
