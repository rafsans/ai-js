# Job Category Classifier + Resume Matching

Project ini adalah sistem klasifikasi kategori pekerjaan berbasis teks dan matching resume-to-job menggunakan machine learning.

## Deskripsi

| Komponen | Keterangan |
|---|---|
| Input model | Teks pekerjaan atau resume (kolom `text`) |
| Target klasifikasi | Kategori pekerjaan (kolom `label`) |
| Algoritma utama | BiLSTM Deep Learning |
| Baseline | TF-IDF + Logistic Regression |
| Matching | Cosine Similarity TF-IDF |

## Format Dataset

Dataset training **wajib** memiliki dua kolom berikut:

```
text,label
```

| Kolom | Keterangan |
|---|---|
| `text` | Input model: teks pekerjaan atau resume |
| `label` | Target: kategori pekerjaan (job category) |

Contoh isi dataset:

```csv
text,label
"software engineer python sql database api",Information Technology
"mechanical design engineer autocad manufacturing",Engineering
"procurement executive supplier vendor purchasing",Operations
```

**File dataset:** `data/ds_jobs_ready.csv`

Dataset dengan 14 kategori pekerjaan yang didukung:

- Engineering
- Information Technology
- Sales
- Healthcare
- Finance
- Operations
- Marketing
- Human Resources
- Administration
- Education
- Customer Service
- Legal
- Creative Design
- Hospitality

## Struktur Folder

```
resume_classifier/
├── data/
│   ├── ds_jobs_ready.csv          ← dataset utama (text, label)
│   ├── train_split.csv            ← data train
│   ├── val_split.csv              ← data validation
│   ├── test_split.csv             ← data test
│   └── uploads/                   ← file resume yang diupload via API
├── models/
│   ├── model_jobcategory_dl.keras ← model BiLSTM
│   ├── tokenizer.pkl              ← tokenizer Keras
│   ├── label_encoder.pkl          ← LabelEncoder scikit-learn
│   ├── train_config.json          ← konfigurasi training
│   └── jobcategory_tfidf_logreg.joblib  ← model baseline
├── results/
│   ├── deep_learning_report.txt
│   ├── baseline_tfidf_logreg_results.json
│   ├── model_comparison.csv
│   └── ranked_job_matches.csv
├── 1_preprocess_data.py
├── 2_train_deep_learning.py
├── 3_evaluate_deep_learning.py
├── 4_compare_models.py
├── 5_train_baseline_tfidf.py
├── 6_cosine_similarity_regression.py
├── app.py
├── predict_top3_dl.py
├── matching.py
├── project_utils.py
├── extractors.py
└── requirements.txt
```

## Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Siapkan Dataset

Letakkan file dataset di:

```bash
data/ds_jobs_ready.csv
```

Dataset harus memiliki kolom `text` dan `label`.

### 3. Jalankan Pipeline Secara Urutan

```bash
# Step 1: Preprocessing & validasi dataset
python 1_preprocess_data.py

# Step 2: Training model deep learning BiLSTM
python 2_train_deep_learning.py

# Step 3: Evaluasi model deep learning
python 3_evaluate_deep_learning.py

# Step 4: Training model baseline TF-IDF
python 5_train_baseline_tfidf.py

# Step 5: Bandingkan DL vs Baseline
python 4_compare_models.py

# Step 6: Cosine similarity ranking (opsional)
python 6_cosine_similarity_regression.py

# Step 7: Jalankan API Flask
python app.py
```

## API Endpoints

### GET /

Info API.

```bash
curl http://127.0.0.1:5000/
```

---

### POST /predict-text

Prediksi Top-3 job category dari teks JSON.

**Request:**

```bash
curl -X POST http://127.0.0.1:5000/predict-text \
  -H "Content-Type: application/json" \
  -d '{"text": "python sql machine learning data analysis dashboard statistics"}'
```

**Response:**

```json
{
  "top3_predictions": [
    {"category": "Information Technology", "confidence": 0.85},
    {"category": "Engineering",            "confidence": 0.10},
    {"category": "Finance",                "confidence": 0.05}
  ],
  "predicted_category": "Information Technology",
  "job_recommendations": [
    {
      "rank": 1,
      "label": "Information Technology",
      "text": "data analyst python sql...",
      "cosine_similarity_score": 0.82,
      "match_percentage": 82.0
    }
  ]
}
```

---

### POST /predict

Prediksi Top-3 job category dari file upload (PDF, DOCX, atau TXT).

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@resume.pdf"
```

---

### POST /match-jobs

Ranking pekerjaan berdasarkan cosine similarity resume.

**Request:**

```bash
curl -X POST http://127.0.0.1:5000/match-jobs \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "python sql machine learning data analyst", "top_n": 5}'
```

**Response:**

```json
{
  "total_matches": 5,
  "matches": [
    {
      "rank": 1,
      "label": "Information Technology",
      "text": "data analyst python sql ...",
      "cosine_similarity_score": 0.79,
      "match_percentage": 79.0
    }
  ]
}
```

## Catatan Teknis

- Dataset tidak seimbang → menggunakan `class_weight='balanced'` (bukan SMOTE).
- Model BiLSTM menggunakan EarlyStopping dan ModelCheckpoint.
- Nama model konsisten: `model_jobcategory_dl.keras`, `jobcategory_tfidf_logreg.joblib`.
- API tidak crash jika model belum dilatih — memberikan pesan error yang jelas.
- Matching menggunakan kolom `text` dan `label` dari dataset.
