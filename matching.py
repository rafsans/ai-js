"""
matching.py
===========
Modul resume-to-job matching menggunakan cosine similarity TF-IDF.
Dataset standar: kolom text (teks pekerjaan) dan label (kategori pekerjaan).
Kini menggunakan dataset 'jobs_recommendation.csv' untuk mempertahankan metadata 
(job_title, job_type, gaji_perbulan, job_skill).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from project_utils import clean_noise, load_ready_data

# [PERUBAHAN ARSITEKTUR] Mengarah ke dataset khusus rekomendasi
READY_DATA_FILE = "data/jobs_recommendation.csv"

# [PERUBAHAN ARSITEKTUR] Membuka gerbang untuk metadata pekerjaan yang kaya
OUTPUT_COLUMNS = [
    "job_title", 
    "job_type", 
    "gaji_perbulan", 
    "job_skill", 
    "label", 
    "text", 
    "cosine_similarity_score", 
    "match_percentage"
]


def load_jobs(path: str = READY_DATA_FILE) -> pd.DataFrame:
    return load_ready_data(path)


def _normalize_text(value: str) -> str:
    return str(value).strip().lower()


def _get_job_text_column(jobs_df: pd.DataFrame) -> str:
    """Cari kolom teks pekerjaan. Dataset baru pasti punya 'text'."""
    for col in ["text", "clean_text", "job_description"]:
        if col in jobs_df.columns:
            return col
    raise ValueError(
        "Tidak ada kolom teks pekerjaan. "
        "Pastikan dataset memiliki kolom 'text'."
    )


def _get_category_column(jobs_df: pd.DataFrame) -> str | None:
    """Cari kolom kategori. Dataset baru pasti punya 'label'."""
    for col in ["label", "job_category", "category"]:
        if col in jobs_df.columns:
            return col
    return None


def _build_tfidf_scores(resume_clean: str, job_texts: list[str]) -> list[float]:
    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), stop_words="english"
    )
    matrix     = vectorizer.fit_transform([resume_clean] + job_texts)
    resume_vec = matrix[0]
    job_vecs   = matrix[1:]
    return cosine_similarity(resume_vec, job_vecs).flatten()


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in OUTPUT_COLUMNS if col in df.columns]
    return df[available]


def rank_jobs_by_resume_text(
    resume_text: str,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Ranking seluruh pekerjaan berdasarkan cosine similarity dengan resume."""
    jobs_df = jobs_df if jobs_df is not None else load_jobs()

    resume_clean = clean_noise(resume_text)
    text_col     = _get_job_text_column(jobs_df)
    job_texts    = jobs_df[text_col].fillna("").astype(str).tolist()

    scores = _build_tfidf_scores(resume_clean, job_texts)

    result = jobs_df.copy()
    result["cosine_similarity_score"] = scores
    result["match_percentage"]        = (result["cosine_similarity_score"] * 100).round(2)
    result = result.sort_values("cosine_similarity_score", ascending=False)\
                   .head(top_n).reset_index(drop=True)

    # Tambahkan ranking
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def rank_jobs_by_category(
    resume_text: str,
    predicted_category: str,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Ranking pekerjaan dalam kategori tertentu berdasarkan cosine similarity."""
    jobs_df = jobs_df if jobs_df is not None else load_jobs()

    category_col = _get_category_column(jobs_df)
    if category_col is None:
        print("[WARNING] Kolom kategori tidak ditemukan. Fallback ke pencarian global.")
        return rank_jobs_by_resume_text(resume_text=resume_text, jobs_df=jobs_df, top_n=top_n)

    predicted_clean = _normalize_text(predicted_category)
    filtered = jobs_df[
        jobs_df[category_col].fillna("").astype(str).apply(_normalize_text) == predicted_clean
    ].copy()

    if filtered.empty:
        print(f"[WARNING] Tidak ada data untuk kategori: {predicted_category}. Fallback ke pencarian global.")
        return rank_jobs_by_resume_text(resume_text=resume_text, jobs_df=jobs_df, top_n=top_n)

    resume_clean = clean_noise(resume_text)
    text_col     = _get_job_text_column(filtered)
    job_texts    = filtered[text_col].fillna("").astype(str).tolist()

    scores = _build_tfidf_scores(resume_clean, job_texts)

    result = filtered.copy()
    result["cosine_similarity_score"] = scores
    result["match_percentage"]        = (result["cosine_similarity_score"] * 100).round(2)
    result = result.sort_values("cosine_similarity_score", ascending=False)\
                   .head(top_n).reset_index(drop=True)

    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def rank_jobs_from_file(
    file_path: str,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    from predict_top3_dl import extract_text_from_file
    text = extract_text_from_file(file_path)
    return rank_jobs_by_resume_text(resume_text=text, jobs_df=jobs_df, top_n=top_n)


def rank_jobs_from_file_by_category(
    file_path: str,
    predicted_category: str,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    from predict_top3_dl import extract_text_from_file
    text = extract_text_from_file(file_path)
    return rank_jobs_by_category(
        resume_text=text,
        predicted_category=predicted_category,
        jobs_df=jobs_df,
        top_n=top_n,
    )