"""
project_utils.py
================
Utilitas bersama untuk Job Category Classifier.
Dataset standar: kolom text (input) dan label (target).
"""

import math
import os
import re
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# I/O helpers
# ===========================================================================

def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def load_ready_data(path: str = "data/ds_jobs_ready.csv") -> pd.DataFrame:
    """Load dataset yang sudah memiliki kolom text dan label."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}\n"
            "Pastikan file ds_jobs_ready.csv ada di folder data/."
        )
    df = read_csv_safe(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing)}\n"
            "Dataset harus memiliki kolom: text, label\n"
            "Jalankan 1_preprocess_data.py jika dataset belum siap."
        )
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"]  = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[
        (df["label"] != "") & (df["label"].str.lower() != "nan")
    ].reset_index(drop=True)
    return df


# ===========================================================================
# Text cleaning
# ===========================================================================

def clean_noise(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[/|,;:(){}\[\]\n\t]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================================================================
# Safe preprocessing — melindungi tech tokens & normalisasi skill alias
# ===========================================================================

# Nama teknologi yang sering rusak akibat regex agresif
_TECH_PLACEHOLDERS: dict[str, str] = {
    "c++":        "CPLUSPLUSLANG",
    "c#":         "CSHARPLANG",
    ".net":       "DOTNETFRAMEWORK",
    "asp.net":    "ASPNETFRAMEWORK",
    "node.js":    "NODEJSRUNTIME",
    "react.js":   "REACTJSLIB",
    "vue.js":     "VUEJSLIB",
    "next.js":    "NEXTJSLIB",
    "nuxt.js":    "NUXTJSLIB",
    "express.js": "EXPRESSJSLIB",
    "nest.js":    "NESTJSLIB",
    "three.js":   "THREEJSLIB",
    "d3.js":      "D3JSLIB",
    "f#":         "FSHARPLANG",
    "graphql":    "GRAPHQLLANG",
}

# Alias skill pendek → nama lengkap
SKILL_ALIASES: dict[str, str] = {
    # Programming languages
    "js":    "javascript",
    "ts":    "typescript",
    "py":    "python",
    "rb":    "ruby",
    "rs":    "rust",
    "kt":    "kotlin",
    "go":    "golang",
    # ML / AI
    "ml":    "machine learning",
    "ai":    "artificial intelligence",
    "dl":    "deep learning",
    "nlp":   "natural language processing",
    "cv":    "computer vision",
    "rl":    "reinforcement learning",
    "llm":   "large language model",
    "genai": "generative ai",
    # Data
    "da":    "data analysis",
    "ds":    "data science",
    "de":    "data engineering",
    "etl":   "extract transform load",
    "bi":    "business intelligence",
    # DevOps / Cloud
    "k8s":   "kubernetes",
    "tf":    "terraform",
    "ci":    "continuous integration",
    "cd":    "continuous deployment",
    "aws":   "amazon web services",
    "gcp":   "google cloud platform",
    "az":    "microsoft azure",
    # Web
    "fe":    "frontend",
    "be":    "backend",
    "fs":    "fullstack",
    "api":   "application programming interface",
    "rest":  "restful api",
    # PM / Management
    "pm":    "project management",
    "po":    "product owner",
    "ux":    "user experience",
    "ui":    "user interface",
    # Database
    "db":    "database",
    "sql":   "structured query language",
    "nosql": "non relational database",
    "pg":    "postgresql",
}


def _protect_tech_tokens(text: str) -> str:
    """Ganti nama teknologi dengan placeholder sebelum cleaning agresif."""
    result = text.lower()
    for token, placeholder in _TECH_PLACEHOLDERS.items():
        result = result.replace(token, placeholder)
    return result


def _restore_tech_tokens(text: str) -> str:
    """Kembalikan placeholder ke nama teknologi aslinya."""
    result = text
    for token, placeholder in _TECH_PLACEHOLDERS.items():
        result = result.replace(placeholder, token)
    return result


def normalize_skills(text: str) -> str:
    """
    Normalisasi alias skill ke nama lengkap.
    Contoh: 'ml' → 'machine learning', 'k8s' → 'kubernetes'
    """
    tokens = text.split()
    normalized = []
    for token in tokens:
        clean_token = token.lower().strip(".,;:()")
        normalized.append(SKILL_ALIASES.get(clean_token, token))
    return " ".join(normalized)


def safe_clean(text: str) -> str:
    """
    Preprocessing aman untuk teks CV / job description:
    1. Protect tech tokens (c++, c#, .net, node.js, dll.)
    2. Jalankan clean_noise
    3. Restore tech tokens
    4. Normalisasi skill alias
    """
    text = _protect_tech_tokens(str(text))
    text = clean_noise(text)
    text = _restore_tech_tokens(text)
    text = normalize_skills(text)
    return text


# ===========================================================================
# Skill extraction
# ===========================================================================

SKILL_TAXONOMY = {
    "Python": [r"\bpython\b"],
    "R": [r"\br programming\b", r"\br language\b", r"\bggplot\b", r"\btidyverse\b", r"\brstudio\b"],
    "SQL": [r"\bsql\b", r"\bmysql\b", r"\bpostgresql\b", r"\bsql server\b"],
    "Excel": [r"\bexcel\b", r"\bspreadsheet\b"],
    "Machine Learning": [r"\bmachine learning\b", r"\bml\b", r"\bmodeling\b", r"\bmodelling\b"],
    "Deep Learning": [r"\bdeep learning\b", r"\bneural network\b"],
    "NLP": [r"\bnlp\b", r"\bnatural language processing\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Scikit-learn": [r"\bscikit[- ]learn\b", r"\bsklearn\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Data Analysis": [r"\bdata analysis\b", r"\banalytical\b", r"\banalyze\b", r"\banalyse\b", r"\banalytics\b"],
    "Data Visualization": [r"\bdata visualization\b", r"\btableau\b", r"\bpower bi\b", r"\bdashboard\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical\b"],
    "Big Data": [r"\bbig data\b", r"\bspark\b", r"\bhadoop\b"],
    "Cloud": [r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bcloud\b"],
    "Docker": [r"\bdocker\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "API": [r"\bapi\b", r"\brest\b", r"\bfastapi\b", r"\bflask\b", r"\bdjango\b"],
    "Marketing": [r"\bmarketing\b", r"\bbrand\b", r"\bcampaign\b"],
    "Digital Marketing": [r"\bdigital marketing\b", r"\bonline presence\b"],
    "SEO": [r"\bseo\b", r"\bsearch engine optimization\b"],
    "SEM": [r"\bsem\b", r"\bsearch engine marketing\b"],
    "Social Media": [r"\bsocial media\b", r"\binstagram\b", r"\btiktok\b", r"\bfacebook\b", r"\blinkedin\b"],
    "Sales": [r"\bsales\b", r"\bcommission\b", r"\bclient\b"],
    "Customer Service": [r"\bcustomer[- ]focused\b", r"\bcustomer service\b", r"\bclients\b"],
    "Leadership": [r"\bleadership\b", r"\blead a team\b", r"\bteam leader\b"],
    "Teamwork": [r"\bteam player\b", r"\bcollaboration\b", r"\bcollaborate\b"],
    "Communication": [r"\bcommunication skills\b", r"\bwritten and verbal\b"],
    "Project Management": [r"\bproject status\b", r"\bmanage\b", r"\bplan and oversee\b"],
    "Risk Management": [r"\brisk management\b", r"\bmodel risk\b", r"\brisk\b"],
    "Auditing": [r"\baudit\b", r"\bauditor\b", r"\bauditing\b"],
    "Compliance": [r"\bcompliance\b", r"\bregulatory\b"],
    "Documentation": [r"\bdocumentation\b", r"\breports\b"],
    "Problem Solving": [r"\bproblem[- ]solving\b", r"\broot causes\b"],
}


def extract_skills(text) -> list:
    text = "" if pd.isna(text) else str(text).lower()
    return [
        skill for skill, patterns in SKILL_TAXONOMY.items()
        if any(re.search(p, text, re.I) for p in patterns)
    ]


# ===========================================================================
# Train/val/test split helpers
# ===========================================================================

def can_use_stratify(y, test_size: float) -> bool:
    counts = Counter(y)
    if len(counts) < 2 or min(counts.values()) < 2:
        return False
    n_test  = math.ceil(len(y) * test_size)
    n_train = len(y) - n_test
    return n_test >= len(counts) and n_train >= len(counts)


def _class_preserving_holdout_split(
    df: pd.DataFrame,
    holdout_size: float,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) == 0 or holdout_size <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy().reset_index(drop=True)

    train_parts, holdout_parts = [], []
    for _, group in df.groupby("label", sort=False):
        group = group.sample(frac=1.0, random_state=random_state)
        n = len(group)
        if n < 2:
            train_parts.append(group)
            continue
        n_holdout = max(1, min(int(round(n * holdout_size)), n - 1))
        holdout_parts.append(group.iloc[:n_holdout])
        train_parts.append(group.iloc[n_holdout:])

    train_df   = pd.concat(train_parts,   axis=0) if train_parts   else df.iloc[0:0].copy()
    holdout_df = pd.concat(holdout_parts, axis=0) if holdout_parts else df.iloc[0:0].copy()
    return (
        train_df.sample(frac=1.0,   random_state=random_state).reset_index(drop=True),
        holdout_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True),
    )


def _move_unseen_labels_to_train(
    train_df: pd.DataFrame,
    eval_df:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(eval_df) == 0:
        return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)
    unseen = ~eval_df["label"].astype(str).isin(set(train_df["label"].astype(str)))
    if unseen.any():
        moved    = eval_df[unseen].copy()
        eval_df  = eval_df[~unseen].copy()
        train_df = pd.concat([train_df, moved], axis=0)
        print(f"[INFO] {len(moved)} baris eval/test dipindah ke train karena label belum muncul di train.")
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = 0.10,
    val_size:  float = 0.10,
    random_state: int = 42,
    prefer_existing_split: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy().reset_index(drop=True)
    if "label" not in df.columns:
        raise ValueError("Kolom 'label' tidak ditemukan.")
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["label"] != "") & (df["label"].str.lower() != "nan")].reset_index(drop=True)

    if prefer_existing_split and "split" in df.columns:
        split_col = df["split"].fillna("").astype(str).str.lower().str.strip()
        if split_col.isin(["train", "test"]).any():
            train_base = df[split_col.eq("train")].copy()
            test_df    = df[split_col.eq("test")].copy()
            other_df   = df[~split_col.isin(["train", "test"])].copy()
            if len(other_df):
                train_base = pd.concat([train_base, other_df], axis=0)
            if len(train_base) == 0:
                train_base, test_df = df.copy(), df.iloc[0:0].copy()
            train_base, test_df = _move_unseen_labels_to_train(train_base, test_df)
            train_df, val_df    = _class_preserving_holdout_split(train_base, val_size, random_state)
            train_df, val_df    = _move_unseen_labels_to_train(train_df, val_df)
            return train_df, val_df, test_df.reset_index(drop=True)

    train_val_df, test_df = _class_preserving_holdout_split(df, test_size, random_state)
    train_df, val_df      = _class_preserving_holdout_split(train_val_df, val_size, random_state)
    train_df, val_df      = _move_unseen_labels_to_train(train_df, val_df)
    train_df, test_df     = _move_unseen_labels_to_train(train_df, test_df)
    return train_df, val_df, test_df


# ===========================================================================
# Metrics
# ===========================================================================

def top_k_accuracy_from_probs(probs, y_true_int, k: int = 3) -> float:
    if len(y_true_int) == 0:
        return 0.0
    k    = min(k, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([true in row for true, row in zip(y_true_int, topk)]))