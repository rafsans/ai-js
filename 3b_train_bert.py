

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# HuggingFace
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import torch
from torch.utils.data import Dataset

from project_utils import load_ready_data, split_train_val_test, top_k_accuracy_from_probs

# ===========================================================================
# Paths
# ===========================================================================
INPUT_FILE        = "data/ds_jobs_ready.csv"
TRAIN_SPLIT_FILE  = "data/train_split.csv"
VAL_SPLIT_FILE    = "data/val_split.csv"
TEST_SPLIT_FILE   = "data/test_split.csv"

BERT_MODEL_DIR    = "models/bert_jobcategory"
LABEL_ENCODER_BERT = "models/bert_label_encoder.pkl"
BERT_CONFIG_FILE  = "models/bert_train_config.json"

RESULTS_DIR       = "results"
BERT_RESULTS_FILE = "results/bert_results.json"

PRETRAINED_MODEL  = "bert-base-uncased"   # ganti ke "indobenchmark/indobert-base-p1" untuk Bahasa Indonesia
MAX_LENGTH        = 512
BATCH_SIZE        = 16
EPOCHS            = 5
LEARNING_RATE     = 2e-5


# ===========================================================================
# Dataset class
# ===========================================================================
class ResumeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ===========================================================================
# Helpers
# ===========================================================================
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"]  = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    return df[df["label"] != ""].reset_index(drop=True)


def load_or_create_splits():
    """Pakai split CSV yang sudah ada; kalau belum ada, buat baru."""
    if (os.path.exists(TRAIN_SPLIT_FILE) and
            os.path.exists(VAL_SPLIT_FILE) and
            os.path.exists(TEST_SPLIT_FILE)):
        train_df = prepare_dataframe(pd.read_csv(TRAIN_SPLIT_FILE))
        val_df   = prepare_dataframe(pd.read_csv(VAL_SPLIT_FILE))
        test_df  = prepare_dataframe(pd.read_csv(TEST_SPLIT_FILE))
        print("[INFO] Menggunakan split yang sudah ada (dari BiLSTM training).")
        return train_df, val_df, test_df

    df = prepare_dataframe(load_ready_data(INPUT_FILE))
    train_df, val_df, test_df = split_train_val_test(
        df, test_size=0.10, val_size=0.10, random_state=42
    )
    train_df.to_csv(TRAIN_SPLIT_FILE, index=False, encoding="utf-8")
    val_df.to_csv(VAL_SPLIT_FILE,   index=False, encoding="utf-8")
    test_df.to_csv(TEST_SPLIT_FILE,  index=False, encoding="utf-8")
    return train_df, val_df, test_df


def compute_metrics_fn(label_encoder):
    """Closure supaya compute_metrics bisa akses label_encoder."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc   = float(accuracy_score(labels, preds))

        # Top-3 accuracy
        probs  = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
        top3   = top_k_accuracy_from_probs(probs, labels, k=min(3, logits.shape[1]))
        return {"accuracy": acc, "top_3_accuracy": float(top3)}
    return compute_metrics


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    train_df, val_df, test_df = load_or_create_splits()

    print("[INFO] Split shape:")
    print("  train:", train_df.shape)
    print("  val  :", val_df.shape)
    print("  test :", test_df.shape)

    X_train = train_df["text"].tolist()
    y_train_raw = train_df["label"].tolist()
    X_val   = val_df["text"].tolist()
    y_val_raw   = val_df["label"].tolist()
    X_test  = test_df["text"].tolist()
    y_test_raw  = test_df["label"].tolist()

    # ── 2. Label encoding ─────────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    num_classes = len(label_encoder.classes_)

    if num_classes < 2:
        raise ValueError("Minimal butuh 2 kelas label untuk training BERT.")

    # Filter label yang ada di encoder (val & test)
    def filter_known(X, y_raw):
        mask = [lbl in set(label_encoder.classes_) for lbl in y_raw]
        X_f  = [x for x, m in zip(X, mask) if m]
        y_f  = label_encoder.transform([y for y, m in zip(y_raw, mask) if m])
        removed = sum(1 for m in mask if not m)
        if removed:
            print(f"[WARNING] {removed} baris dibuang — label tidak dikenal encoder.")
        return X_f, y_f

    X_val, y_val   = filter_known(X_val,  y_val_raw)
    X_test, y_test = filter_known(X_test, y_test_raw)

    id2label = {i: lbl for i, lbl in enumerate(label_encoder.classes_)}
    label2id = {lbl: i for i, lbl in id2label.items()}

    # ── 3. Tokenizer ──────────────────────────────────────────────────────
    print(f"[INFO] Memuat tokenizer: {PRETRAINED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    def tokenize(texts):
        return tokenizer(
            texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )

    print("[INFO] Tokenisasi data...")
    train_enc = tokenize(X_train)
    val_enc   = tokenize(X_val)
    test_enc  = tokenize(X_test)

    train_dataset = ResumeDataset(train_enc, y_train)
    val_dataset   = ResumeDataset(val_enc,   y_val)
    test_dataset  = ResumeDataset(test_enc,  y_test)

    # ── 4. Model ──────────────────────────────────────────────────────────
    print(f"[INFO] Memuat model: {PRETRAINED_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ── 5. Training arguments ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=BERT_MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="top_3_accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(BERT_MODEL_DIR, "logs"),
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),   # aktif hanya jika ada GPU
        dataloader_num_workers=0,
        report_to="none",                 # matikan wandb/tensorboard default
    )

    # ── 6. Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn(label_encoder),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n[INFO] Mulai training BERT...")
    trainer.train()

    # ── 7. Simpan model & tokenizer ───────────────────────────────────────
    trainer.save_model(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)

    with open(LABEL_ENCODER_BERT, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"[INFO] Model disimpan ke: {BERT_MODEL_DIR}")

    # ── 8. Evaluasi di test set ───────────────────────────────────────────
    test_result = {}
    if len(X_test) > 0:
        print("\n[INFO] Evaluasi di test set...")
        raw_preds   = trainer.predict(test_dataset)
        logits      = raw_preds.predictions
        preds_int   = np.argmax(logits, axis=1)
        probs       = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()

        y_pred_labels = label_encoder.inverse_transform(preds_int)
        y_true_labels = label_encoder.inverse_transform(y_test)

        test_acc  = float(accuracy_score(y_true_labels, y_pred_labels))
        test_top3 = float(top_k_accuracy_from_probs(probs, y_test, k=min(3, num_classes)))
        cls_report = classification_report(y_true_labels, y_pred_labels, zero_division=0)

        test_result = {
            "test_accuracy":      test_acc,
            "test_top3_accuracy": test_top3,
        }

        print("\n==============================")
        print("BERT TEST SET EVALUATION")
        print("==============================")
        print(f"Accuracy    : {test_acc:.4f}")
        print(f"Top-3 Acc   : {test_top3:.4f}")
        print("\nClassification Report:")
        print(cls_report)
    else:
        cls_report = "[WARNING] Test set kosong."
        print("[WARNING] Test set kosong, evaluasi dilewati.")

    # ── 9. Simpan config & hasil ──────────────────────────────────────────
    bert_config = {
        "pretrained_model": PRETRAINED_MODEL,
        "max_length":       MAX_LENGTH,
        "batch_size":       BATCH_SIZE,
        "epochs":           EPOCHS,
        "learning_rate":    LEARNING_RATE,
        "num_classes":      num_classes,
        "label_classes":    label_encoder.classes_.tolist(),
        "model_dir":        BERT_MODEL_DIR,
        "label_encoder":    LABEL_ENCODER_BERT,
        "input_column":     "text",
        "target_column":    "label",
        "split_shapes": {
            "train": list(train_df.shape),
            "validation": list(val_df.shape),
            "test":  list(test_df.shape),
        },
        **test_result,
    }
    with open(BERT_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(bert_config, f, indent=2, ensure_ascii=False)

    bert_results = {
        "model":  "BERT (bert-base-uncased)",
        "target": "label (job_category)",
        "input_column": "text",
        **test_result,
        "test_classification_report": cls_report,
        "config": bert_config,
    }
    with open(BERT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(bert_results, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("TRAINING BERT SELESAI")
    print("==============================")
    print(f"[INFO] Model dir         : {BERT_MODEL_DIR}")
    print(f"[INFO] Label encoder     : {LABEL_ENCODER_BERT}")
    print(f"[INFO] Train config      : {BERT_CONFIG_FILE}")
    print(f"[INFO] Hasil evaluasi    : {BERT_RESULTS_FILE}")


if __name__ == "__main__":
    main()
