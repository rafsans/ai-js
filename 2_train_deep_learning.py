"""
2_train_deep_learning.py
========================
Training model BiLSTM untuk Job Category Classifier.
Dataset: data/ds_jobs_ready.csv (kolom: text, label)
"""

import json
import os
import pickle
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from project_utils import load_ready_data, split_train_val_test, top_k_accuracy_from_probs

INPUT_FILE        = "data/ds_jobs_ready.csv"
MODEL_FILE        = "models/model_jobcategory_dl.keras"
TOKENIZER_FILE    = "models/tokenizer.pkl"
LABEL_ENCODER_FILE = "models/label_encoder.pkl"
TRAIN_CONFIG_FILE = "models/train_config.json"
HISTORY_FILE      = "models/deep_learning_history.csv"
TRAIN_SPLIT_FILE  = "data/train_split.csv"
VAL_SPLIT_FILE    = "data/val_split.csv"
TEST_SPLIT_FILE   = "data/test_split.csv"



class TargetCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_accuracy", target=0.85, min_epoch=8):
        super().__init__()
        self.monitor = monitor
        self.target = target
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        score = logs.get(self.monitor)

        # Jangan stop terlalu cepat
        if epoch + 1 < self.min_epoch:
            return

        if score is not None and score >= self.target:
            print(f"\n[INFO] Target {self.monitor} tercapai: {score:.4f}")
            self.model.stop_training = True


def build_model(vocab_size: int, max_length: int, num_classes: int) -> Model:
    inputs = layers.Input(shape=(max_length,), name="input_layer")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=256,  # DITINGKATKAN DARI 128
        name="embedding"
    )(inputs)

    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.0),
        name="bilstm_1"
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.0), # UBAH return_sequences=True
        name="bilstm_2"
    )(x)

    x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x) # TAMBAH POOLING

    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)
    return Model(inputs=inputs, outputs=outputs)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    df = load_ready_data(INPUT_FILE)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df = df[df["label"] != ""].reset_index(drop=True)

    print("[INFO] Dataset    :", INPUT_FILE)
    print("[INFO] Jumlah data:", len(df))
    print("[INFO] Kolom input: text")
    print("[INFO] Kolom label: label")
    print("[INFO] Jumlah kategori label:", df["label"].nunique())
    print("\n[INFO] Distribusi label:")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        raise ValueError("Minimal butuh 2 kelas label untuk training deep learning.")

    train_df, val_df, test_df = split_train_val_test(
        df, text_col="text", label_col="label",
        train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, random_state=42
    )
    train_df.to_csv(TRAIN_SPLIT_FILE, index=False, encoding="utf-8")
    val_df.to_csv(VAL_SPLIT_FILE, index=False, encoding="utf-8")
    test_df.to_csv(TEST_SPLIT_FILE, index=False, encoding="utf-8")

    print("\n[INFO] Split shape:")
    print("  train:", train_df.shape)
    print("  val  :", val_df.shape)
    print("  test :", test_df.shape)

    X_train = train_df["text"].fillna("").astype(str)
    X_val   = val_df["text"].fillna("").astype(str)
    X_test  = test_df["text"].fillna("").astype(str)
    y_train_raw = train_df["label"].fillna("").astype(str)
    y_val_raw   = val_df["label"].fillna("").astype(str)
    y_test_raw  = test_df["label"].fillna("").astype(str)

    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train_raw)
    num_classes = len(label_encoder.classes_)
    if num_classes < 2:
        raise ValueError("Minimal butuh 2 kelas label pada train set.")

    y_val_int  = label_encoder.transform(y_val_raw)  if len(y_val_raw)  else np.array([], dtype=int)
    y_test_int = label_encoder.transform(y_test_raw) if len(y_test_raw) else np.array([], dtype=int)

    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
    y_val   = tf.keras.utils.to_categorical(y_val_int,   num_classes=num_classes) if len(y_val_int)  else None
    y_test  = tf.keras.utils.to_categorical(y_test_int,  num_classes=num_classes) if len(y_test_int) else None

    vocab_size = 20000
    max_length = 400
    tokenizer  = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length, padding="post", truncating="post")
    val_pad   = pad_sequences(tokenizer.texts_to_sequences(X_val),   maxlen=max_length, padding="post", truncating="post") if len(X_val)  else None
    test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=max_length, padding="post", truncating="post") if len(X_test) else None

    # class_weight digunakan karena dataset tidak seimbang
    # Versi stabil: akar ke-4 + clipping
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_int),
        y=y_train_int
    )

    class_weights_array = np.power(class_weights_array, 0.25)
    class_weights_array = np.clip(class_weights_array, 0.7, 2.0)

    class_weights = {
        int(cls): float(w)
        for cls, w in zip(np.unique(y_train_int), class_weights_array)
    }

    print("\n[INFO] Class weights:")
    for cls, weight in class_weights.items():
        print(f"  {cls} - {label_encoder.classes_[cls]}: {weight:.4f}")

    model = build_model(vocab_size=vocab_size, max_length=max_length, num_classes=num_classes)
    
    # UBAH LEARNING RATE DARI 1e-3 KE 5e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4) 
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=min(3, num_classes),
                name="top_3_accuracy"
            )
        ],
    )

    has_validation = val_pad is not None and y_val is not None and len(val_df) > 0
    # Karena target utama adalah accuracy minimal 85%,
    # model disimpan berdasarkan val_accuracy, bukan val_top_3_accuracy.
    monitor_metric = "val_accuracy" if has_validation else "accuracy"

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        ),
        TargetCallback(
            monitor=monitor_metric,
            target=0.85,
            min_epoch=8
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=6,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_FILE,
            monitor=monitor_metric,
            save_best_only=True,
            mode="max",
            verbose=1
        ),
    ]
    

    history = model.fit(
        train_pad, y_train,
        epochs=30,
        batch_size=min(64, max(1, len(train_df))),
        validation_data=(val_pad, y_val) if has_validation else None,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    pd.DataFrame(history.history).to_csv(HISTORY_FILE, index=False, encoding="utf-8")

    if not os.path.exists(MODEL_FILE):
        model.save(MODEL_FILE)

    if os.path.exists(MODEL_FILE):
        model = tf.keras.models.load_model(MODEL_FILE)

    test_result = {}
    if test_pad is not None and y_test is not None and len(test_df):
        test_metrics = model.evaluate(test_pad, y_test, verbose=0)
        test_result = dict(zip(model.metrics_names, [float(x) for x in test_metrics]))
        print("\n[INFO] Test metrics:", test_result)

    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(LABEL_ENCODER_FILE, "wb") as f:
        pickle.dump(label_encoder, f)

    with open(TRAIN_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": vocab_size,
            "max_length": max_length,
            "num_classes": num_classes,
            "input_column": "text",
            "target_column": "label",
            "model_type": "BiLSTM Job Category Classifier",
            "model_file": MODEL_FILE,
            "split_shapes": {
                "train": list(train_df.shape),
                "validation": list(val_df.shape),
                "test": list(test_df.shape),
            },
            "label_classes": label_encoder.classes_.tolist(),
            "test_metrics": test_result,
        }, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print("TRAINING DEEP LEARNING SELESAI")
    print("==============================")
    print(f"[INFO] Model disimpan ke         : {MODEL_FILE}")
    print(f"[INFO] Tokenizer disimpan ke     : {TOKENIZER_FILE}")
    print(f"[INFO] Label encoder disimpan ke : {LABEL_ENCODER_FILE}")
    print(f"[INFO] Train config disimpan ke  : {TRAIN_CONFIG_FILE}")


if __name__ == "__main__":
    main()