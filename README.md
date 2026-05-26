# Job Category Classifier API

Sistem klasifikasi kategori pekerjaan berbasis BERT dan matching resume-to-job menggunakan cosine similarity TF-IDF, dilengkapi REST API Flask dengan fitur auto-translate Bahasa Indonesia → Inggris via Gemini API.

---

## Arsitektur Sistem

| Komponen | Keterangan |
|---|---|
| Model klasifikasi | BERT (`bert-base-uncased`) fine-tuned |
| Matching | Cosine Similarity TF-IDF (bigram, cached) |
| Auto-translate | Gemini 2.5 Flash API |
| Analisis CV | Gemini 2.5 Flash API (role: Senior HRD) |
| Framework API | Flask + flask-limiter |
| Ekstraksi file | PyMuPDF (PDF), python-docx (DOCX) |

---

## Struktur Folder

```
Model_Bert_v2/
├── data/
│   ├── ds_jobs_ready.csv          ← dataset utama (text, label, job_title, job_type, salary)
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   └── uploads/                   ← file sementara saat proses (auto-hapus)
├── models/
│   ├── bert_jobcategory/          ← folder model BERT hasil training
│   └── bert_label_encoder.pkl     ← LabelEncoder scikit-learn
├── logs/
│   └── app/
│       └── app.log                ← log rotasi (maks 5MB × 3 backup)
├── app.py                         ← entry point API Flask
├── predict_top3_dl.py             ← inferensi BERT + auto-translate
├── matching.py                    ← job matching cosine similarity
├── translator.py                  ← Gemini translate & analisis CV
├── extractors.py                  ← ekstrak teks dari PDF/DOCX/TXT
├── logger.py                      ← konfigurasi logging
├── project_utils.py               ← preprocessing & utilitas dataset
├── 1_preprocess_data.py
├── 2_train_deep_learning.py
├── 3b_train_bert.py               ← training BERT (model utama)
├── 3_evaluate_deep_learning.py
├── 4_compare_models.py
├── 5_train_baseline_tfidf.py
├── 6_cosine_similarity_regression.py
├── requirements.txt
└── .env                           ← konfigurasi API key (jangan di-commit)
```

---

## Instalasi

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd Model_Bert_v2
pip install -r requirements.txt
```

### 2. Konfigurasi environment

Buat file `.env` di root project:

```env
GEMINI_API_KEY=your_api_key_here
FLASK_DEBUG=false
PORT=5000
REDIS_URL=redis://localhost:6379   # opsional, untuk rate limiter multi-worker
```

> Dapatkan `GEMINI_API_KEY` di: https://aistudio.google.com

### 3. Siapkan dataset

Letakkan file dataset di `data/ds_jobs_ready.csv`. Kolom wajib:

| Kolom | Keterangan |
|---|---|
| `text` | Teks deskripsi pekerjaan (sudah di-preprocess) |
| `label` | Kategori pekerjaan |
| `job_title` | Judul pekerjaan |
| `job_type` | Tipe pekerjaan (Full-time, Part-time, dll) |
| `salary` / `gaji_perbulan` | Gaji (opsional, ditampilkan di response jika ada) |

### 4. Training model BERT

```bash
python 3b_train_bert.py
```

Model tersimpan di `models/bert_jobcategory/` dan `models/bert_label_encoder.pkl`.

### 5. Jalankan API

```bash
python app.py
```

Server berjalan di `http://localhost:5000` (default).

---

## API Endpoints

### GET `/`

Info status API.

```bash
curl http://localhost:5000/
```

```json
{
  "service": "Job Category Classifier API",
  "model": "BERT (bert-base-uncased)",
  "status": "active",
  "model_loaded": true,
  "dataset_loaded": true
}
```

---

### POST `/predict-text`

Prediksi Top-3 kategori pekerjaan dari teks JSON. Mendukung Bahasa Indonesia (auto-translate).

Tidak ada rate limit.

**Request:**

```bash
curl -X POST http://localhost:5000/predict-text \
  -H "Content-Type: application/json" \
  -d '{"text": "python sql machine learning data analysis", "top_n": 5}'
```

| Parameter | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `text` | string | ✅ | Teks CV atau deskripsi keahlian |
| `top_n` | integer | ❌ | Jumlah rekomendasi pekerjaan (default: 5, maks: 50) |

**Response:**

```json
{
  "original_text": "python sql machine learning data analysis",
  "top3_predictions": [
    {"category": "information_technology", "confidence": 0.91},
    {"category": "administration",         "confidence": 0.05},
    {"category": "digital_marketing",      "confidence": 0.02}
  ],
  "predicted_category": "information_technology",
  "job_recommendations": [
    {
      "rank": 1,
      "label": "information_technology",
      "job_title": "Data Analyst",
      "job_type": "Full-time",
      "salary": "5000000",
      "cosine_similarity_score": 0.82,
      "match_percentage": 82.0
    }
  ]
}
```

---

### POST `/predict`

Prediksi Top-3 kategori pekerjaan dari file upload. Mendukung PDF, DOCX, TXT.

**Rate limit:** 5 request/menit, 20 request/jam.

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@resume.pdf" \
  -F "top_n=5"
```

| Parameter | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `file` | file | ✅ | File PDF, DOCX, atau TXT (maks 10MB) |
| `top_n` | integer | ❌ | Jumlah rekomendasi pekerjaan (default: 5, maks: 50) |

**Response:** sama dengan `/predict-text`, ditambah field `filename` dan `confidence_level` (High / Medium / Low).

> PDF berbasis gambar/scan tidak didukung — hanya PDF dengan teks yang dapat diseleksi.

---

### POST `/match-jobs`

Ranking seluruh pekerjaan berdasarkan cosine similarity dengan teks resume, tanpa klasifikasi BERT.

Tidak ada rate limit.

```bash
curl -X POST http://localhost:5000/match-jobs \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "python sql data analyst machine learning", "top_n": 10}'
```

| Parameter | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `resume_text` | string | ✅ | Teks resume atau keahlian |
| `top_n` | integer | ❌ | Jumlah hasil (default: 10, maks: 50) |

---

### POST `/analyze-cv`

Analisis CV dan feedback perbaikan menggunakan Gemini API (role: Senior HRD). Mendukung teks JSON atau file upload.

**Rate limit:** 5 request/menit, 20 request/jam.

```bash
# Dari teks
curl -X POST http://localhost:5000/analyze-cv \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya lulusan S1 Informatika dengan pengalaman 2 tahun..."}'

# Dari file
curl -X POST http://localhost:5000/analyze-cv \
  -F "file=@resume.pdf"
```

**Response:**

```json
{
  "cv_feedback": "## Evaluasi CV\n\nStruktur CV cukup baik...",
  "source": "file"
}
```

---

### POST `/translate`

Terjemahkan teks ke Bahasa Inggris menggunakan Gemini API.

**Rate limit:** 10 request/menit, 50 request/jam.

```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya memiliki pengalaman di bidang pengembangan web"}'
```

---

## Konfigurasi Rate Limit

| Endpoint | Limit/Menit | Limit/Jam |
|---|---|---|
| `POST /predict` | 5 | 20 |
| `POST /analyze-cv` | 5 | 20 |
| `POST /translate` | 10 | 50 |
| `POST /match-jobs` | 10 | 50 |
| `POST /predict-text` | — | — |

Rate limiter menggunakan **Redis** jika `REDIS_URL` dikonfigurasi. Tanpa Redis, fallback ke in-memory (tidak persisten antar restart, tidak efektif di multi-worker).

---

## Error Codes

| HTTP Code | Penyebab |
|---|---|
| `400` | Input kosong atau format tidak valid |
| `413` | File melebihi batas ukuran (10MB) |
| `429` | Rate limit terlampaui |
| `500` | Model belum dilatih atau error internal |
| `503` | `GEMINI_API_KEY` belum dikonfigurasi |

---

## Logging

Log ditulis ke terminal dan file `logs/app/app.log` secara bersamaan.

**Format:**
```
2026-05-24 14:16:28 | INFO    | app        | File diterima: resume.pdf (214.3 KB)
2026-05-24 14:16:30 | INFO    | app        | Prediksi selesai: resume.pdf → administration
2026-05-24 14:18:03 | ERROR   | translator | Analyze error: quota exceeded
```

Rotasi otomatis: maksimal 5MB per file, 3 file backup (`app.log.1`, `app.log.2`, `app.log.3`).

---

## Catatan Teknis

- Model BERT wajib dilatih terlebih dahulu via `3b_train_bert.py` sebelum API bisa melayani request prediksi.
- API tidak crash jika model atau dataset belum tersedia — endpoint yang membutuhkannya mengembalikan HTTP 500 dengan pesan yang jelas.
- TF-IDF vectorizer di-cache per kumpulan job description sehingga request kedua dan seterusnya jauh lebih cepat.
- File upload dihapus otomatis setelah diproses, baik sukses maupun error.
- Teks di-truncate di batas kalimat terakhir (maks 12.000 karakter) sebelum dikirim ke Gemini.