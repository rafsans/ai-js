# Job Matching Platform - AI Service

Ini adalah Microservice AI (Kecerdasan Buatan) khusus untuk **Job Matching Platform**. Tugas utama servis ini adalah menerima dokumen resume/CV (PDF), membaca teks di dalamnya, dan memberikan rekomendasi kategori pekerjaan terbaik menggunakan Natural Language Processing (NLP).

## Tech Stack
- **Python** (Bahasa pemrograman utama)
- **Flask** (Framework *micro-web* yang ringan)
- **Transformers (HuggingFace)** (Menggunakan arsitektur model BERT untuk pemrosesan teks tingkat lanjut)
- **PyPDF2** (Untuk mengekstrak tulisan dari file PDF pelamar)
- **Scikit-Learn & PyTorch** (Untuk kalkulasi dan *machine learning*)

## Cara Menjalankan Secara Lokal

Pastikan kamu sudah menginstal **Python** (disarankan versi 3.9 ke atas).

1. **Buka terminal** di dalam folder ini.
2. **Buat Virtual Environment (Sangat Direkomendasikan)**:
   Ini berguna agar *library* proyek ini tidak bentrok dengan Python di sistem operasi kamu.
   ```bash
   python -m venv venv
   ```
3. **Aktifkan Virtual Environment**:
   - Jika kamu pakai **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - Jika kamu pakai **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```
   *(Kamu akan melihat tanda `(venv)` di sebelah kiri terminalmu jika berhasil).*
4. **Install Library yang Dibutuhkan**:
   ```bash
   pip install flask pypdf2 transformers torch scikit-learn numpy pandas
   ```
5. **Nyalakan Server AI**:
   ```bash
   python app.py
   ```
Servis AI akan menyala di `http://127.0.0.1:5000`. Sekarang Backend sudah bisa mengirimkan *resume* ke sini untuk diproses!