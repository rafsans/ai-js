import os
from google import genai

# Setup API Key milik Anda (Termasuk dalam Free Tier / Jalur Gratis)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyD7rS26HCUD1of7DpH3a9au1loMMyLa4lE")

def translate_to_english(text: str) -> str:
    """
    Menerjemahkan teks resume dari bahasa Indonesia ke bahasa Inggris menggunakan Gemini API.
    Memanfaatkan jalur Free Tier berkecepatan tinggi dengan SDK resmi terbaru (google-genai).
    """
    if not text or not text.strip():
        return text
    
    try:
        # Menginisialisasi koneksi klien dengan kunci API Anda
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = (
            "Translate the following resume or job description text to English. "
            "Only output the translated text without any conversational filler or explanation. "
            "If it's already in English, just return the exact same text or fix minor typos:\n\n"
            f"{text}"
        )
        
        # Menggunakan model cerdas yang gratis di tier pengembang
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[WARNING] Gemini API Translation failed: {e}. Falling back to original text.")
        return text

def analyze_resume(text: str) -> str:
    """
    Memberikan umpan balik (feedback) perbaikan CV menggunakan peran HR Expert via Gemini API.
    """
    if not text or not text.strip():
        return "Tidak ada teks yang dapat dianalisis."
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
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
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[WARNING] Gemini API Analysis failed: {e}")
        return "Analisis CV tidak tersedia saat ini. Silakan coba lagi nanti."
