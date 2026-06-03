

import os
import pandas as pd

INPUT_FILE = "data/ds_jobs_ready.csv"
OUTPUT_FILE = "data/ds_jobs_ready.csv"


def main():
    os.makedirs("data", exist_ok=True)

    print(f"[INFO] Membaca dataset: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding="latin1")

    print(f"[INFO] Shape awal: {df.shape}")
    print(f"[INFO] Kolom tersedia: {df.columns.tolist()}")

    if "text" not in df.columns:
        raise ValueError(
            "Kolom 'text' tidak ditemukan. "
            "Dataset harus memiliki kolom: text, label"
        )
    if "label" not in df.columns:
        raise ValueError(
            "Kolom 'label' tidak ditemukan. "
            "Dataset harus memiliki kolom: text, label"
        )

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()

    before = len(df)
    df = df[
        (df["text"] != "") &
        (df["text"].str.lower() != "nan") &
        (df["label"] != "") &
        (df["label"].str.lower() != "nan")
    ].reset_index(drop=True)
    print(f"[INFO] Baris kosong dibuang: {before - len(df)}")

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(f"[INFO] Duplikat dibuang: {before_dedup - len(df)}")

    if df["label"].nunique() < 2:
        raise ValueError("Minimal harus ada 2 kelas label untuk klasifikasi.")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n[INFO] Dataset siap training disimpan ke: {OUTPUT_FILE}")
    print(f"[INFO] Jumlah data   : {len(df):,}")
    print(f"[INFO] Jumlah kolom  : {len(df.columns)}")
    print(f"[INFO] Jumlah label  : {df['label'].nunique()}")

    print("\n[INFO] Distribusi label:")
    print(df["label"].value_counts().to_string())

    print("\n[INFO] Contoh 5 data:")
    print(df[["text", "label"]].head().to_string())


if __name__ == "__main__":
    main()
