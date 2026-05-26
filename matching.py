import functools
import hashlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from project_utils import safe_clean, load_ready_data
from logger import get_logger

log = get_logger("matching")

# Cache safe_clean agar teks yang sama tidak diproses ulang
_cached_safe_clean = functools.lru_cache(maxsize=10000)(safe_clean)

READY_DATA_FILE = "data/ds_jobs_ready.csv"

OUTPUT_COLUMNS = ["label", "text", "cosine_similarity_score", "match_percentage", "job_title", "job_type", "gaji_perbulan", "salary"]

SIMILARITY_THRESHOLD = 0.03  # Ambang batas untuk "matched" vs "uncertain"

# Cache vectorizer yang sudah di-fit agar tidak rebuild setiap request
_vectorizer_cache: dict = {}

def _get_vectorizer(job_texts: list[str]) -> tuple[TfidfVectorizer, any]:
    """
    Kembalikan vectorizer yang sudah di-fit untuk kumpulan job_texts tertentu.
    Jika job_texts sama seperti sebelumnya, gunakan cache — tidak fit ulang.
    """
    cache_key = hashlib.md5(" |||| ".join(job_texts).encode()).hexdigest()
    if cache_key not in _vectorizer_cache:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
        )
        job_matrix = vectorizer.fit_transform(job_texts)
        _vectorizer_cache[cache_key] = (vectorizer, job_matrix)
    return _vectorizer_cache[cache_key]


def load_jobs(path: str = READY_DATA_FILE) -> pd.DataFrame:
    return load_ready_data(path)


def _normalize_text(value: str) -> str:
    return str(value).strip().lower()


def _get_job_text_column(jobs_df: pd.DataFrame) -> str:
    for col in ["text", "clean_text", "job_description"]:
        if col in jobs_df.columns:
            return col
    raise ValueError(
        "Tidak ada kolom teks pekerjaan. "
        "Pastikan dataset memiliki kolom 'text'."
    )


def _get_category_column(jobs_df: pd.DataFrame) -> str | None:
    for col in ["label", "job_category", "category"]:
        if col in jobs_df.columns:
            return col
    return None


def _build_tfidf_scores(resume_clean: str, job_texts: list[str]) -> list[float]:
    """
    Hitung cosine similarity antara resume dan setiap job description.
    Vectorizer di-cache berdasarkan kumpulan job_texts — tidak di-fit ulang
    selama dataset tidak berubah.
    """
    vectorizer, job_matrix = _get_vectorizer(job_texts)
    resume_vec = vectorizer.transform([resume_clean])
    return cosine_similarity(resume_vec, job_matrix).flatten()


def _flag_low_confidence(df: pd.DataFrame, threshold: float = SIMILARITY_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    df["match_status"] = df["cosine_similarity_score"].apply(
        lambda s: "matched" if s >= threshold else "uncertain"
    )
    return df


def rank_jobs_by_resume_text(
    resume_text: str,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 10,
    include_uncertain: bool = False,
) -> pd.DataFrame:
    """Ranking seluruh pekerjaan berdasarkan cosine similarity dengan resume."""
    jobs_df = jobs_df if jobs_df is not None else load_jobs()

    resume_clean = _cached_safe_clean(resume_text)
    text_col     = _get_job_text_column(jobs_df)
    job_texts    = jobs_df[text_col].fillna("").astype(str).apply(_cached_safe_clean).tolist()

    scores = _build_tfidf_scores(resume_clean, job_texts)

    result = jobs_df.copy()
    result["cosine_similarity_score"] = scores
    result["match_percentage"]        = (result["cosine_similarity_score"] * 100).round(2)
    result = _flag_low_confidence(result)

    if not include_uncertain:
        result = result[result["match_status"] == "matched"]

    result = (
        result.sort_values("cosine_similarity_score", ascending=False)
              .head(top_n)
              .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    
    # PERBAIKAN 1: Filter kolom sesuai OUTPUT_COLUMNS yang tersedia
    available_cols = [col for col in OUTPUT_COLUMNS if col in result.columns]
    return result[
        ["rank"] + available_cols
    ]


def rank_jobs_by_category(
    resume_text: str,
    predicted_category: str,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 10,
    include_uncertain: bool = False,
) -> pd.DataFrame:
    """Ranking pekerjaan dalam kategori tertentu berdasarkan cosine similarity."""
    jobs_df = jobs_df if jobs_df is not None else load_jobs()

    category_col = _get_category_column(jobs_df)
    if category_col is None:
        log.warning("Kolom kategori tidak ditemukan. Fallback ke pencarian global.")
        return rank_jobs_by_resume_text(
            resume_text=resume_text,
            jobs_df=jobs_df,
            top_n=top_n,
            include_uncertain=include_uncertain,
        )

    predicted_clean = _normalize_text(predicted_category)
    filtered = jobs_df[
        jobs_df[category_col].fillna("").astype(str).apply(_normalize_text) == predicted_clean
    ].copy()

    if filtered.empty:
        log.warning(f"Tidak ada data untuk kategori: {predicted_category}. Fallback ke pencarian global.")
        return rank_jobs_by_resume_text(
            resume_text=resume_text,
            jobs_df=jobs_df,
            top_n=top_n,
            include_uncertain=include_uncertain,
        )

    resume_clean = _cached_safe_clean(resume_text)
    text_col     = _get_job_text_column(filtered)
    job_texts    = filtered[text_col].fillna("").astype(str).apply(_cached_safe_clean).tolist()

    scores = _build_tfidf_scores(resume_clean, job_texts)

    result = filtered.copy()
    result["cosine_similarity_score"] = scores
    result["match_percentage"]        = (result["cosine_similarity_score"] * 100).round(2)
    result = _flag_low_confidence(result)

    if not include_uncertain:
        result = result[result["match_status"] == "matched"]

    result = (
        result.sort_values("cosine_similarity_score", ascending=False)
              .head(top_n)
              .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    
    # PERBAIKAN 2: Filter kolom sesuai OUTPUT_COLUMNS yang tersedia
    available_cols = [col for col in OUTPUT_COLUMNS if col in result.columns]
    return result[
        ["rank"] + available_cols
    ]


def rank_jobs_from_file(
    file_path: str,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 10,
    include_uncertain: bool = False,
) -> pd.DataFrame:
    from predict_top3_dl import extract_text_from_file
    text = extract_text_from_file(file_path)
    return rank_jobs_by_resume_text(
        resume_text=text,
        jobs_df=jobs_df,
        top_n=top_n,
        include_uncertain=include_uncertain,
    )


def rank_jobs_from_file_by_category(
    file_path: str,
    predicted_category: str,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 10,
    include_uncertain: bool = False,
) -> pd.DataFrame:
    from predict_top3_dl import extract_text_from_file
    text = extract_text_from_file(file_path)
    return rank_jobs_by_category(
        resume_text=text,
        predicted_category=predicted_category,
        jobs_df=jobs_df,
        top_n=top_n,
        include_uncertain=include_uncertain,
    )
