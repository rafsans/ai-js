
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from matching import rank_jobs_by_resume_text
from project_utils import clean_noise, load_ready_data

INPUT_FILE             = "data/ds_jobs_ready.csv"
SAMPLE_RESUME_FILE     = "data/sample_resume.txt"
RESULTS_DIR            = "results"
RANKING_OUTPUT         = "results/ranked_job_matches.csv"
REGRESSION_MODEL_FILE  = "models/cosine_regression_model.joblib"
REGRESSION_RESULTS_FILE = "results/cosine_regression_results.json"


def get_sample_resume_text() -> str:
    if os.path.exists(SAMPLE_RESUME_FILE):
        with open(SAMPLE_RESUME_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return (
        "Data analyst with Python, SQL, Excel, statistics, dashboard, data visualization, "
        "machine learning, pandas, numpy, and business reporting experience."
    )


def compute_pairwise_cosine_scores(df: pd.DataFrame, resume_col: str, job_col: str = "text") -> np.ndarray:
    resume_texts = df[resume_col].fillna("").astype(str).apply(clean_noise).tolist()
    job_texts    = df[job_col].fillna("").astype(str).apply(clean_noise).tolist()

    scores = []
    for resume_text, job_text in zip(resume_texts, job_texts):
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([resume_text, job_text])
        score  = cosine_similarity(matrix[0], matrix[1])[0, 0]
        scores.append(float(score))
    return np.array(scores)


def train_regression_if_resume_column_exists(df: pd.DataFrame):
    resume_col = None
    for candidate in ["resume_text", "resume_clean_text", "cleaned_resume", "resume"]:
        if candidate in df.columns:
            resume_col = candidate
            break

    if resume_col is None:
        return {
            "regression_trained": False,
            "reason": (
                "Dataset tidak memiliki kolom resume_text/resume_clean_text, "
                "sehingga model regresi supervised belum bisa dilatih."
            ),
            "available_solution": (
                "Cosine similarity ranking tetap dibuat dari sample resume terhadap seluruh kolom text."
            ),
        }

    df = df.dropna(subset=[resume_col, "text"]).copy()
    df["cosine_similarity_score"] = compute_pairwise_cosine_scores(df, resume_col=resume_col, job_col="text")
    df["combined_text"] = (
        df[resume_col].fillna("").astype(str).apply(clean_noise)
        + " [SEP] "
        + df["text"].fillna("").astype(str).apply(clean_noise)
    )

    X = df["combined_text"]
    y = df["cosine_similarity_score"]

    if len(df) < 5:
        return {
            "regression_trained": False,
            "reason": "Data pasangan resume-job terlalu sedikit untuk regresi.",
            "rows_with_resume_pair": int(len(df)),
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words="english")),
        ("regressor", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0, 1)

    metrics = {
        "regression_trained": True,
        "resume_column": resume_col,
        "target": "cosine_similarity_score",
        "rows_used": int(len(df)),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "mse":  float(mean_squared_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "r2":   float(r2_score(y_test, y_pred)),
    }
    joblib.dump(model, REGRESSION_MODEL_FILE)
    return metrics


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_ready_data(INPUT_FILE)

    sample_resume = get_sample_resume_text()
    ranked = rank_jobs_by_resume_text(sample_resume, jobs_df=df, top_n=20)
    ranked.to_csv(RANKING_OUTPUT, index=False, encoding="utf-8")

    regression_status = train_regression_if_resume_column_exists(df)

    results = {
        "cosine_similarity_ranking_created": True,
        "ranking_output": RANKING_OUTPUT,
        "score_range": "0-1",
        "dataset_mapping": {
            "job_text": "text (kolom dataset)",
            "job_label": "label (job_category)",
        },
        "regression_status": regression_status,
    }

    with open(REGRESSION_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=== COSINE SIMILARITY / REGRESSION MODULE ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[INFO] Ranking job match disimpan ke: {RANKING_OUTPUT}")
    print(f"[INFO] Summary disimpan ke          : {REGRESSION_RESULTS_FILE}")


if __name__ == "__main__":
    main()
