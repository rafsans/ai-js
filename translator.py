import os
from pathlib import Path
from dotenv import load_dotenv

from logger import get_logger
log = get_logger("translator")

_ENV_PATHS = [Path(".env")]

def _load_env() -> None:
    for path in _ENV_PATHS:
        if path.exists():
            load_dotenv(path, override=True)
            return

_load_env()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Baca GEMINI_API_KEY dari environment. Raise jika kosong."""
    key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY belum dikonfigurasi.\n"
            "Buat file .env lalu isi: GEMINI_API_KEY=your_api_key_here\n"
            "Dapatkan key di: https://aistudio.google.com"
        )
    return key


def _get_client():
    """Buat Gemini client dengan key terbaru dari environment."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "Library google-genai belum terinstall.\n"
            "Jalankan: pip install google-genai"
        )
    return genai.Client(api_key=_get_api_key())


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

_MAX_CHARS         = 12000
_TRANSLATE_TIMEOUT = 15
_ANALYZE_TIMEOUT   = 20


def _truncate_at_sentence(text: str, max_chars: int = _MAX_CHARS) -> str:
    """Potong teks di batas kalimat terakhir yang muat dalam max_chars."""
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    last  = max(chunk.rfind(". "), chunk.rfind(".\n"), chunk.rfind("! "), chunk.rfind("? "))
    return chunk[:last + 1].strip() if last != -1 else chunk.strip()


def translate_to_english(text: str, timeout: int = _TRANSLATE_TIMEOUT) -> str:
    """
    Menerjemahkan teks resume dari bahasa Indonesia ke bahasa Inggris
    menggunakan Gemini API.
    """
    if not text or not text.strip():
        return text

    text = _truncate_at_sentence(text)

    try:
        from google.genai import types

        client = _get_client()
        prompt = (
            "Translate ALL text to professional English.\n"
            "Do not keep Indonesian words.\n"
            "Convert the entire sentence fully into English.\n"
            "Keep only technical terms, programming languages, and software names unchanged.\n"
            "Output ONLY the translated English text without explanation.\n\n"
            f"{text}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(timeout=timeout * 1000),
            ),
        )
        return (response.text or "").strip() or text

    except EnvironmentError as e:
        log.warning(f"Translator: {e}")
        return text
    except Exception as e:
        log.warning(f"Gemini API Translation failed: {e}. Menggunakan teks asli.")
        return text


def analyze_resume(text: str, timeout: int = _ANALYZE_TIMEOUT) -> str:
    """
    Memberikan umpan balik (feedback) perbaikan CV menggunakan peran
    HR Expert via Gemini API.
    """
    if not text or not text.strip():
        return "Tidak ada teks yang dapat dianalisis."

    text = _truncate_at_sentence(text)

    try:
        from google.genai import types

        client = _get_client()
        prompt = (
            "Anda adalah seorang Senior HRD dan Konsultan Karir Profesional. "
            "Tugas Anda adalah membaca teks CV/Resume berikut lalu memberikan umpan balik (feedback) "
            "yang konstruktif dalam bahasa Indonesia. Tolong berikan analisis singkat yang mencakup:\n"
            "1. Evaluasi ringkas mengenai struktur dan keterbacaan CV.\n"
            "2. Saran perbaikan kalimat agar terlihat lebih profesional dan 'menjual'.\n"
            "3. Kelebihan utama yang menonjol dari CV ini.\n"
            "4. Hal penting apa yang masih kurang/perlu ditambahkan.\n\n"
            "PENTING: Pastikan total keseluruhan respon Anda tepat terdiri dari 6 hingga 8 kalimat saja. "
            "Gunakan format Markdown yang rapi.\n\n"
            f"Teks CV:\n{text}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(timeout=timeout * 1000),
            ),
        )
        return (response.text or "").strip()

    except EnvironmentError as e:
        log.warning(f"Analyzer: {e}")
        return f"Analisis CV tidak tersedia: {e}"
    except Exception as e:
        log.error(f"Analyze error: {e}")
        return f"ERROR ANALYZE: {str(e)}"
