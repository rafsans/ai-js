import os
from predict_top3_dl import predict_from_file

def jalankan_test():
    print("=== PROGRAM PENGUJIAN CV/RESUME (BERT + GEMINI API) ===")
    
    path_cv = os.path.join("data", "uploads", "CV_test.pdf")  # path relatif dari root project
    
    if not os.path.exists(path_cv):
        print(f"\n[INFO] Silakan ubah variabel 'path_cv' di dalam file ini ke lokasi CV asli Anda.")
        print(f"Path saat ini ({path_cv}) belum tersedia.")
        return

    print(f"\nMemproses berkas: {path_cv}...")
    print("Menerjemahkan ke bahasa Inggris menggunakan Gemini API & memprediksi kategori...")
    
    try:
        hasil = predict_from_file(path_cv)
        
        print("\n" + "="*50)
        print("BERKAS             :", hasil.get("filename"))
        print("-" * 50)
        print("UMPAN BALIK HR (GEMINI AI):")
        print(hasil.get("cv_feedback", "Tidak ada umpan balik."))
        print("-" * 50)
        print("TINGKAT KEYAKINAN  :", hasil.get("confidence_level"))
        print("TOP-3 PREDIKSI KATEGORI:")
        
        for i, rank in enumerate(hasil.get("top3_predictions", []), 1):
            print(f"   {i}. {rank['category']} -> Confidence: {rank['confidence']:.4f}")
            
        print("="*50)
        
    except Exception as e:
        print(f"\nTerjadi kesalahan saat memproses: {e}")

if __name__ == "__main__":
    jalankan_test()
