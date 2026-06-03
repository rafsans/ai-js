

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from project_utils import load_ready_data, split_train_val_test

INPUT_FILE      = "data/ds_jobs_ready.csv"
TRAIN_SPLIT_FILE = "data/train_split.csv"
VAL_SPLIT_FILE   = "data/val_split.csv"
TEST_SPLIT_FILE  = "data/test_split.csv"
MODEL_FILE       = "models/jobcategory_tfidf_logreg.joblib"
RESULTS_DIR      = "results"
RESULTS_FILE     = "results/baseline_tfidf_logreg_results.json"
PREDICTION_FILE  = "results/baseline_test_predictions.csv"


def prepare_dataframe(df):
    if "text" not in df.columns:
        raise ValueError("Kolom 'text' tidak ditemukan.")
    if "label" not in df.columns:
        raise ValueError("Kolom 'label' tidak ditemukan.")
    df = df.copy()
    df["text"]  = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    return df[df["label"] != ""].reset_index(drop=True)


def top_k_accuracy_from_pipeline(model, X, y_true, k=3):
    if not hasattr(model, "predict_proba") or len(X) == 0:
        return None
    probs   = model.predict_proba(X)
    classes = np.array(model.classes_)
    k       = min(k, len(classes))
    topk_idx    = np.argsort(-probs, axis=1)[:, :k]
    topk_labels = classes[topk_idx].astype(str)
    y_true_arr  = np.array(y_true).astype(str)
    return float(np.mean([true in row for true, row in zip(y_true_arr, topk_labels)]))


def add_top_k_predictions(pred_df, model, X, k=3):
    if not hasattr(model, "predict_proba") or len(X) == 0:
        return pred_df
    probs   = model.predict_proba(X)
    classes = np.array(model.classes_)
    k       = min(k, len(classes))
    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    for rank in range(k):
        idx = topk_idx[:, rank]
        pred_df[f"top_{rank+1}_category"]    = classes[idx]
        pred_df[f"top_{rank+1}_probability"] = probs[np.arange(len(probs)), idx]
    return pred_df


def load_or_create_splits():
    if (os.path.exists(TRAIN_SPLIT_FILE) and
            os.path.exists(VAL_SPLIT_FILE) and
            os.path.exists(TEST_SPLIT_FILE)):
        train_df = prepare_dataframe(pd.read_csv(TRAIN_SPLIT_FILE))
        val_df   = prepare_dataframe(pd.read_csv(VAL_SPLIT_FILE))
        test_df  = prepare_dataframe(pd.read_csv(TEST_SPLIT_FILE))
        print("[INFO] Menggunakan split yang sudah ada.")
        return train_df, val_df, test_df

    df = prepare_dataframe(load_ready_data(INPUT_FILE))
    train_df, val_df, test_df = split_train_val_test(
        df, test_size=0.10, val_size=0.10, random_state=42
    )
    train_df.to_csv(TRAIN_SPLIT_FILE, index=False, encoding="utf-8")
    val_df.to_csv(VAL_SPLIT_FILE, index=False, encoding="utf-8")
    test_df.to_csv(TEST_SPLIT_FILE, index=False, encoding="utf-8")
    return train_df, val_df, test_df


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_df, val_df, test_df = load_or_create_splits()
    print("[INFO] Split shape:")
    print("  train:", train_df.shape)
    print("  val  :", val_df.shape)
    print("  test :", test_df.shape)

    X_train = train_df["text"].fillna("").astype(str)
    y_train = train_df["label"].fillna("").astype(str)
    X_val   = val_df["text"].fillna("").astype(str)
    y_val   = val_df["label"].fillna("").astype(str)
    X_test  = test_df["text"].fillna("").astype(str)
    y_test  = test_df["label"].fillna("").astype(str)

    if y_train.nunique() < 2:
        raise ValueError("Minimal 2 kelas label pada train set untuk baseline.")

    baseline_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    print("[INFO] Training baseline TF-IDF + Logistic Regression...")
    baseline_model.fit(X_train, y_train)

    if len(X_val):
        y_val_pred = baseline_model.predict(X_val)
        val_acc    = accuracy_score(y_val, y_val_pred)
        val_top3   = top_k_accuracy_from_pipeline(baseline_model, X_val, y_val, k=3)
        val_report = classification_report(y_val, y_val_pred, zero_division=0)
    else:
        y_val_pred, val_acc, val_top3, val_report = [], None, None, "[WARNING] Validation set kosong."

    if len(X_test):
        y_test_pred = baseline_model.predict(X_test)
        test_acc    = accuracy_score(y_test, y_test_pred)
        test_top3   = top_k_accuracy_from_pipeline(baseline_model, X_test, y_test, k=3)
        test_report = classification_report(y_test, y_test_pred, zero_division=0)
    else:
        y_test_pred, test_acc, test_top3, test_report = [], None, None, "[WARNING] Test set kosong."

    print("\n======================================")
    print("BASELINE TF-IDF + LOGISTIC REGRESSION")
    print("Target: label (job_category)")
    print("======================================")
    print("Validation Accuracy   :", val_acc)
    print("Validation Top-3 Acc  :", val_top3)
    print(val_report)
    print("Test Accuracy         :", test_acc)
    print("Test Top-3 Accuracy   :", test_top3)
    print(test_report)

    joblib.dump(baseline_model, MODEL_FILE)

    if len(X_test):
        pred_df = test_df[["text", "label"]].copy()
        pred_df["eval_split"]       = "test"
        pred_df["y_true"]           = y_test.values
        pred_df["y_pred"]           = y_test_pred
        pred_df["pred_probability"] = baseline_model.predict_proba(X_test).max(axis=1)
        pred_df = add_top_k_predictions(pred_df, baseline_model, X_test, k=3)
        pred_df.to_csv(PREDICTION_FILE, index=False, encoding="utf-8")
    elif len(X_val):
        pred_df = val_df[["text", "label"]].copy()
        pred_df["eval_split"]       = "validation"
        pred_df["y_true"]           = y_val.values
        pred_df["y_pred"]           = y_val_pred
        pred_df["pred_probability"] = baseline_model.predict_proba(X_val).max(axis=1)
        pred_df = add_top_k_predictions(pred_df, baseline_model, X_val, k=3)
        pred_df.to_csv(PREDICTION_FILE, index=False, encoding="utf-8")

    results = {
        "model": "TF-IDF + LogisticRegression (class_weight=balanced)",
        "model_file": MODEL_FILE,
        "target": "label (job_category)",
        "input_column": "text",
        "validation_accuracy":  None if val_acc  is None else float(val_acc),
        "validation_top3_accuracy": None if val_top3  is None else float(val_top3),
        "test_accuracy":     None if test_acc is None else float(test_acc),
        "test_top3_accuracy": None if test_top3 is None else float(test_top3),
        "split_shapes": {
            "train": list(train_df.shape),
            "validation": list(val_df.shape),
            "test": list(test_df.shape),
        },
        "train_label_distribution": {
            str(k): int(v) for k, v in y_train.value_counts().items()
        },
        "validation_classification_report": val_report,
        "test_classification_report": test_report,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Baseline model disimpan ke : {MODEL_FILE}")
    print(f"[INFO] Hasil disimpan ke          : {RESULTS_FILE}")


if __name__ == "__main__":
    main()
