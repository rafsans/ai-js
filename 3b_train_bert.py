import json
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)

from project_utils import load_ready_data, split_train_val_test, top_k_accuracy_from_probs

# ===========================================================================
# Paths
# ===========================================================================
INPUT_FILE          = "data/ds_jobs_ready.csv"
TRAIN_SPLIT_FILE    = "data/train_split.csv"
VAL_SPLIT_FILE      = "data/val_split.csv"
TEST_SPLIT_FILE     = "data/test_split.csv"

BERT_MODEL_DIR      = "models/bert_jobcategory"
BERT_SAVEDMODEL_DIR = "models/bert_jobcategory_savedmodel"   # SavedModel format (TF native)
LABEL_ENCODER_BERT  = "models/bert_label_encoder.pkl"
BERT_CONFIG_FILE    = "models/bert_train_config.json"

RESULTS_DIR         = "results"
BERT_RESULTS_FILE   = "results/bert_results.json"
HISTORY_FILE        = "results/bert_history.csv"
TENSORBOARD_LOG_DIR = "logs/bert_fit"

PRETRAINED_MODEL    = "bert-base-uncased"
MAX_LENGTH          = 256
BATCH_SIZE          = 16
EPOCHS              = 5
LEARNING_RATE       = 2e-5


# ===========================================================================
# Custom Components (memenuhi syarat tugas)
# ===========================================================================

# ── Custom Layer ─────────────────────────────────────────────────────────────
class ClassificationHead(tf.keras.layers.Layer):
    """
    Custom classification head di atas output [CLS] token BERT.
    Menggantikan dense classifier bawaan TFBert agar bisa dikustomisasi.
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes   = num_classes
        self.dropout_rate  = dropout_rate
        self.dropout       = tf.keras.layers.Dropout(dropout_rate)
        self.dense         = tf.keras.layers.Dense(
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


# ── Custom Loss ──────────────────────────────────────────────────────────────
class LabelSmoothingCategoricalCrossentropy(tf.keras.losses.Loss):
    """
    Custom Loss: Categorical Crossentropy dengan Label Smoothing.
    Membantu model tidak terlalu overconfident → generalisasi lebih baik.
    """
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


# ── Custom Callback ───────────────────────────────────────────────────────────
class Top3AccuracyCallback(tf.keras.callbacks.Callback):
    """
    Custom Callback: Hitung Top-3 Accuracy di setiap akhir epoch
    menggunakan data validasi dan cetak ke log training.
    """
    def __init__(self, val_dataset, val_steps: int, k: int = 3):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_steps   = val_steps
        self.k           = k

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        all_probs, all_labels = [], []

        for batch in self.val_dataset.take(self.val_steps):
            inputs, y_true = batch
            preds = self.model(inputs, training=False)

            # TFBert returns TFSequenceClassifierOutput — ambil logits
            if hasattr(preds, "logits"):
                probs = tf.nn.softmax(preds.logits, axis=-1).numpy()
            else:
                probs = preds.numpy()

            all_probs.append(probs)
            all_labels.append(np.argmax(y_true.numpy(), axis=-1))

        if not all_probs:
            return

        all_probs  = np.concatenate(all_probs,  axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        top3       = top_k_accuracy_from_probs(all_probs, all_labels, k=self.k)

        logs[f"val_top_{self.k}_accuracy"] = top3
        print(f" — val_top_{self.k}_accuracy: {top3:.4f}")


# ── Custom Model (Model Subclassing) ──────────────────────────────────────────
class TFBertClassifier(tf.keras.Model):
    """
    Model Subclassing: BERT + ClassificationHead kustom.
    Membangun model menggunakan tf.keras.Model subclassing
    sebagaimana disyaratkan dalam tugas AI.
    """
    def __init__(self, bert_encoder, num_classes: int, dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.bert_encoder = bert_encoder
        self.cls_head     = ClassificationHead(num_classes, dropout_rate, name="cls_head")

    def call(self, inputs, training=False):
        """
        Forward pass:
          1. Jalankan BERT encoder → ambil [CLS] token (pooler_output)
          2. Masukkan ke ClassificationHead kustom
        """
        bert_output = self.bert_encoder(inputs, training=training)
        cls_token   = bert_output.pooler_output          # shape: (batch, hidden_size)
        return self.cls_head(cls_token, training=training)

    def get_config(self):
        return super().get_config()


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
        print("[INFO] Menggunakan split yang sudah ada.")
        return train_df, val_df, test_df

    df = prepare_dataframe(load_ready_data(INPUT_FILE))
    train_df, val_df, test_df = split_train_val_test(
        df, test_size=0.10, val_size=0.10, random_state=42
    )
    train_df.to_csv(TRAIN_SPLIT_FILE, index=False, encoding="utf-8")
    val_df.to_csv(VAL_SPLIT_FILE,     index=False, encoding="utf-8")
    test_df.to_csv(TEST_SPLIT_FILE,   index=False, encoding="utf-8")
    return train_df, val_df, test_df


def tokenize_to_tf_dataset(
    texts: list,
    labels: np.ndarray,
    tokenizer,
    num_classes: int,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """
    Tokenisasi teks → tf.data.Dataset siap pakai untuk training/evaluasi.
    Output per batch: ({"input_ids", "attention_mask", "token_type_ids"}, one_hot_labels)
    """
    enc = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    input_ids      = enc["input_ids"].astype(np.int32)
    attention_mask = enc["attention_mask"].astype(np.int32)
    token_type_ids = enc.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int32)
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        one_hot_labels,
    ))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts), seed=42)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs("models",          exist_ok=True)
    os.makedirs(RESULTS_DIR,       exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    # ── 1. Load & split data ──────────────────────────────────────────────
    train_df, val_df, test_df = load_or_create_splits()

    print("[INFO] Split shape:")
    print("  train:", train_df.shape)
    print("  val  :", val_df.shape)
    print("  test :", test_df.shape)

    # ── 2. Label encoding ────────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"].tolist())
    num_classes = len(label_encoder.classes_)
    print(f"[INFO] Jumlah kelas: {num_classes}")

    if num_classes < 2:
        raise ValueError("Minimal butuh 2 kelas label untuk training BERT.")

    def safe_transform(y_raw):
        known = set(label_encoder.classes_)
        mask  = [lbl in known for lbl in y_raw]
        y_f   = label_encoder.transform([y for y, m in zip(y_raw, mask) if m])
        idx_f = [i for i, m in enumerate(mask) if m]
        removed = sum(1 for m in mask if not m)
        if removed:
            print(f"[WARNING] {removed} baris dibuang — label tidak dikenal encoder.")
        return idx_f, y_f

    val_idx,  y_val  = safe_transform(val_df["label"].tolist())
    test_idx, y_test = safe_transform(test_df["label"].tolist())

    val_df_f  = val_df.iloc[val_idx].reset_index(drop=True)
    test_df_f = test_df.iloc[test_idx].reset_index(drop=True)

    # ── 3. Tokenizer ─────────────────────────────────────────────────────
    print(f"[INFO] Memuat tokenizer: {PRETRAINED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    print("[INFO] Membangun tf.data.Dataset...")
    train_ds = tokenize_to_tf_dataset(
        train_df["text"].tolist(), y_train, tokenizer, num_classes, BATCH_SIZE, shuffle=True
    )
    val_ds = tokenize_to_tf_dataset(
        val_df_f["text"].tolist(), y_val, tokenizer, num_classes, BATCH_SIZE
    )
    test_ds = tokenize_to_tf_dataset(
        test_df_f["text"].tolist(), y_test, tokenizer, num_classes, BATCH_SIZE
    )

    val_steps  = len(val_df_f)  // BATCH_SIZE or 1
    test_steps = len(test_df_f) // BATCH_SIZE or 1

    # ── 4. Build Model (Model Subclassing) ────────────────────────────────
    print(f"[INFO] Memuat BERT encoder: {PRETRAINED_MODEL}")
    bert_encoder = TFAutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    ).layers[0]   # ambil layer TFBertMainLayer saja (tanpa classifier bawaannya)

    model = TFBertClassifier(
        bert_encoder=bert_encoder,
        num_classes=num_classes,
        dropout_rate=0.3,
        name="tf_bert_classifier",
    )

    # ── 5. Compile ────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=len(train_df) // BATCH_SIZE * EPOCHS,
        )
    )

    model.compile(
        optimizer=optimizer,
        loss=LabelSmoothingCategoricalCrossentropy(smoothing=0.1, name="label_smooth_ce"),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top_3_accuracy"),
        ],
    )

    # Build model dengan dummy input agar summary bisa ditampilkan
    dummy = {
        "input_ids":      tf.zeros((1, MAX_LENGTH), dtype=tf.int32),
        "attention_mask": tf.zeros((1, MAX_LENGTH), dtype=tf.int32),
        "token_type_ids": tf.zeros((1, MAX_LENGTH), dtype=tf.int32),
    }
    _ = model(dummy, training=False)
    model.summary()

    # ── 6. Callbacks ──────────────────────────────────────────────────────
    callbacks = [
        Top3AccuracyCallback(val_ds, val_steps, k=min(3, num_classes)),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_top_3_accuracy", patience=2,
            restore_best_weights=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_top_3_accuracy", factor=0.5,
            patience=1, min_lr=1e-6, mode="max", verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(BERT_MODEL_DIR, "best_weights"),
            monitor="val_top_3_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max", verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=TENSORBOARD_LOG_DIR,
            histogram_freq=1,
            update_freq="epoch",
        ),
    ]

    # ── 7. Training ───────────────────────────────────────────────────────
    print("\n[INFO] Mulai training TF-BERT...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    pd.DataFrame(history.history).to_csv(HISTORY_FILE, index=False, encoding="utf-8")

    # ── 8. Simpan model ───────────────────────────────────────────────────
    # a) SavedModel format (TensorFlow native — untuk produksi)
    model.save(BERT_SAVEDMODEL_DIR)
    print(f"[INFO] Model SavedModel disimpan ke: {BERT_SAVEDMODEL_DIR}")

    # b) Weights + tokenizer (untuk reload TFBertClassifier)
    os.makedirs(BERT_MODEL_DIR, exist_ok=True)
    model.save_weights(os.path.join(BERT_MODEL_DIR, "tf_bert_weights"))
    tokenizer.save_pretrained(BERT_MODEL_DIR)
    print(f"[INFO] Tokenizer & weights disimpan ke: {BERT_MODEL_DIR}")

    # c) Label encoder
    with open(LABEL_ENCODER_BERT, "wb") as f:
        pickle.dump(label_encoder, f)

    # ── 9. Evaluasi test set ──────────────────────────────────────────────
    test_result = {}
    cls_report  = ""

    if len(test_df_f) > 0:
        print("\n[INFO] Evaluasi di test set...")
        all_probs, all_true = [], []

        for batch_inputs, batch_labels in test_ds:
            preds = model(batch_inputs, training=False)
            if hasattr(preds, "logits"):
                probs = tf.nn.softmax(preds.logits, axis=-1).numpy()
            else:
                probs = preds.numpy()
            all_probs.append(probs)
            all_true.append(np.argmax(batch_labels.numpy(), axis=-1))

        all_probs  = np.concatenate(all_probs,  axis=0)
        all_true   = np.concatenate(all_true,   axis=0)
        preds_int  = np.argmax(all_probs, axis=1)

        y_pred_labels = label_encoder.inverse_transform(preds_int)
        y_true_labels = label_encoder.inverse_transform(all_true)

        test_acc  = float(accuracy_score(y_true_labels, y_pred_labels))
        test_top3 = float(top_k_accuracy_from_probs(all_probs, all_true, k=min(3, num_classes)))
        cls_report = classification_report(y_true_labels, y_pred_labels, zero_division=0)

        test_result = {
            "test_accuracy":      test_acc,
            "test_top3_accuracy": test_top3,
        }

        print("\n==============================")
        print("TF-BERT TEST SET EVALUATION")
        print("==============================")
        print(f"Accuracy    : {test_acc:.4f}")
        print(f"Top-3 Acc   : {test_top3:.4f}")
        print("\nClassification Report:")
        print(cls_report)
    else:
        print("[WARNING] Test set kosong, evaluasi dilewati.")

    # ── 10. Simpan config & hasil ─────────────────────────────────────────
    bert_config = {
        "framework":        "TensorFlow (tf.keras)",
        "model_type":       "TFBertClassifier (Model Subclassing)",
        "pretrained_model": PRETRAINED_MODEL,
        "max_length":       MAX_LENGTH,
        "batch_size":       BATCH_SIZE,
        "epochs":           EPOCHS,
        "learning_rate":    LEARNING_RATE,
        "num_classes":      num_classes,
        "label_classes":    label_encoder.classes_.tolist(),
        "custom_components": {
            "custom_layer":    "ClassificationHead",
            "custom_loss":     "LabelSmoothingCategoricalCrossentropy",
            "custom_callback": "Top3AccuracyCallback",
        },
        "saved_formats": {
            "savedmodel": BERT_SAVEDMODEL_DIR,
            "weights":    os.path.join(BERT_MODEL_DIR, "tf_bert_weights"),
            "tokenizer":  BERT_MODEL_DIR,
        },
        "split_shapes": {
            "train":      list(train_df.shape),
            "validation": list(val_df_f.shape),
            "test":       list(test_df_f.shape),
        },
        **test_result,
    }

    with open(BERT_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(bert_config, f, indent=2, ensure_ascii=False)

    bert_results = {
        "model":  "TF-BERT (bert-base-uncased) — TFBertClassifier",
        "target": "label (job_category)",
        **test_result,
        "test_classification_report": cls_report,
        "config": bert_config,
    }
    with open(BERT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(bert_results, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("TRAINING TF-BERT SELESAI")
    print("==============================")
    print(f"[INFO] SavedModel dir    : {BERT_SAVEDMODEL_DIR}")
    print(f"[INFO] Weights dir       : {BERT_MODEL_DIR}")
    print(f"[INFO] Label encoder     : {LABEL_ENCODER_BERT}")
    print(f"[INFO] Train config      : {BERT_CONFIG_FILE}")
    print(f"[INFO] Hasil evaluasi    : {BERT_RESULTS_FILE}")
    print(f"[INFO] TensorBoard logs  : {TENSORBOARD_LOG_DIR}")
    print(f"\n  tensorboard --logdir {TENSORBOARD_LOG_DIR}")


if __name__ == "__main__":
    main()
BATCH_SIZE