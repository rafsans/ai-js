"""
project_utils.py
================
Utility functions untuk text processing, cleaning, dan evaluasi model.
"""

import re
import pandas as pd
from typing import List, Tuple, Optional

# Generic terms yang tidak membantu klasifikasi BERT
GENERIC_TERMS = [
    "communication",
    "teamwork",
    "problem solving",
    "microsoft office",
    "hardworking",
    "fast learner",
    "detail oriented",
    "self motivated",
    "time management",
    "leadership",
    "interpersonal",
    "organizational",
    "multitasking",
    "creative",
    "analytical"
]

def clean_noise(text) -> str:
    """
    Light text cleaning untuk BERT transformer.
    TIDAK menghapus simbol penting untuk tech stack (+, #, ., /, -).
    
    Perubahan dari versi lama:
    - Menjaga C++, Node.js, ASP.NET, scikit-learn dll
    - Tidak overly agresif menghapus karakter
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize separators (tapi jangan hapus +, #, ., /, -)
    text = re.sub(r"[|,;:(){}\[\]\\n\\t]", " ", text)

    # KEEP important symbols untuk tech stack: + # . / -
    # Hanya hapus karakter yang benar-benar noise
    text = re.sub(r"[^a-zA-Z0-9+#./\s-]", " ", text)

    # Normalize spaces (multiple spaces jadi satu)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_generic_terms(text: str) -> str:
    """
    Hapus generic terms yang tidak membantu klasifikasi BERT.
    Optional: gunakan jika ingin mengurangi noise sebelum inference.
    """
    if not text:
        return ""
    
    words = text.split()
    filtered = [w for w in words if w not in GENERIC_TERMS]
    return " ".join(filtered)


def load_ready_data(path: str = "data/ds_jobs_ready.csv") -> pd.DataFrame:
    """
    Load cleaned dataset untuk matching system.
    Digunakan oleh matching.py untuk mengambil data job dan label.
    """
    df = pd.read_csv(path)

    # Pastikan kolom penting ada
    required_cols = ["text", "label"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ditemukan di dataset: {path}\n"
                f"Kolom yang tersedia: {list(df.columns)}"
            )

    return df


def extract_skills(text: str, skill_list: list = None) -> list:
    """
    ⚠️ NOTE: Fungsi ini hanya untuk analytics & debugging.
    JANGAN digunakan sebagai classifier utama.
    Gunakan BERT semantic classifier untuk production.
    """
    if skill_list is None:
        # Load skill list dari file jika ada
        try:
            with open("models/skills_list.txt", "r") as f:
                skill_list = [line.strip().lower() for line in f]
        except FileNotFoundError:
            return []
    
    cleaned = clean_noise(text)
    found_skills = []
    
    for skill in skill_list:
        if skill.lower() in cleaned:
            found_skills.append(skill)
    
    return found_skills


def split_train_val_test(df, text_col: str, label_col: str, 
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.15, 
                         test_ratio: float = 0.15,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset menjadi train, validation, test.
    Memastikan distribusi label seimbang.
    """
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratio harus berjumlah 1"
    
    # First split: train vs temp (val+test)
    train, temp = train_test_split(
        df, 
        test_size=(1 - train_ratio),
        stratify=df[label_col],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        temp,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp[label_col],
        random_state=random_state
    )
    
    return train, val, test


def top_k_accuracy_from_probs(y_true, y_probs, k: int = 3) -> float:
    """
    Hitung Top-K Accuracy dari probability predictions.
    
    Args:
        y_true: true labels (array-like)
        y_probs: probability predictions (2D array)
        k: top-k yang dihitung
    
    Returns:
        Top-K accuracy score
    """
    import numpy as np
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:][:, ::-1]
    
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Hitung confidence interval untuk metrik evaluasi.
    
    Args:
        scores: list of scores (misal: accuracy per fold)
        confidence: confidence level (default 0.95)
    
    Returns:
        (mean, margin_of_error)
    """
    import numpy as np
    from scipy import stats
    
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    n = len(scores)
    
    # Z-score untuk confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * (std / np.sqrt(n))
    
    return mean, margin


def preview_cleaning(text: str) -> dict:
    """
    Fungsi debugging: lihat perbedaan sebelum dan sesudah cleaning.
    """
    original = text
    cleaned = clean_noise(text)
    filtered = filter_generic_terms(cleaned)
    
    return {
        "original": original,
        "cleaned": cleaned,
        "filtered": filtered,
        "chars_removed": len(original) - len(cleaned),
        "generic_terms_removed": len(cleaned.split()) - len(filtered.split())
    }


# Testing function
if __name__ == "__main__":
    # Test cases untuk memastikan BERT mendapatkan sinyal yang tepat
    test_texts = [
        "Expert in C++ and Node.js development",
        "ASP.NET core with React.js frontend",
        "Python scikit-learn for machine learning",
        "Communication, teamwork, and problem solving skills",
        "Hello@email.com check http://example.com"
    ]
    
    print("=== TEST CLEAN_NOISE() UNTUK BERT ===\n")
    
    for text in test_texts:
        cleaned = clean_noise(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Filtered (tanpa generic): {filter_generic_terms(cleaned)}")
        print("-" * 70)
    
    # Test load_ready_data
    print("\n=== TEST LOAD_READY_DATA ===\n")
    try:
        df = load_ready_data()
        print(f"✅ Dataset loaded: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Unique labels: {df['label'].nunique()}")
    except FileNotFoundError:
        print("⚠️ File data/ds_jobs_ready.csv tidak ditemukan (normal jika belum ada)")
    except Exception as e:
        print(f"❌ Error: {e}")