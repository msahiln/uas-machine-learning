# Laporan Proyek Machine Learning
 
## Domain Proyek
Industri hiburan anime memiliki ribuan judul dengan berbagai genre. Pengguna sering kali merasa kesulitan untuk menemukan judul baru yang sesuai dengan minat mereka di tengah banyaknya pilihan (*Information Overload*). Sistem rekomendasi menjadi solusi krusial untuk membantu pengguna menavigasi pilihan tersebut secara efisien.

**Mengapa masalah ini harus diselesaikan?**
- Meningkatkan kepuasan pengguna dalam menemukan konten yang relevan.
- Membantu platform dalam mempertahankan retensi pengguna.
- Memberikan saran yang dipersonalisasi berdasarkan data historis.

## Business Understanding

### Problem Statements
- Bagaimana memberikan rekomendasi anime yang memiliki kemiripan fitur (genre) dengan anime yang pernah ditonton pengguna?
- Bagaimana memprediksi preferensi pengguna secara akurat berdasarkan rating yang diberikan oleh pengguna lain?

### Goals
- Membangun model *Content-Based Filtering* untuk memberikan rekomendasi berdasarkan kesamaan genre.
- Membangun model *Collaborative Filtering* untuk memberikan rekomendasi yang dipersonalisasi berdasarkan pola rating.

### Solution Statement
- **Content-Based Filtering**: Menggunakan TF-IDF Vectorizer dan Cosine Similarity untuk menghitung kemiripan antar anime.
- **Collaborative Filtering**: Menggunakan Neural Network (RecommenderNet) dengan layer embedding untuk memprediksi rating anime yang belum ditonton.

## Data Understanding
Dataset yang digunakan adalah **Anime Recommendation Database 2020** dari Kaggle yang mencakup lebih dari 17.000 judul anime dan jutaan rating pengguna.

### Variabel-variabel pada dataset:
- `MAL_ID`: Identitas unik anime.
- `Name`: Judul anime.
- `Genres`: Kategori genre (misal: Action, Adventure).
- `user_id`: Identitas unik pengguna.
- `rating`: Skor penilaian dari pengguna (1-10).

## Data Preparation
1. **Data Cleaning**: Menghapus *missing value* dan data duplikat menggunakan `dropna()` dan `drop_duplicates()`.
2. **Feature Engineering**: Melakukan *encoding* pada `user_id` dan `anime_id` menjadi indeks integer agar bisa diproses model deep learning.
3. **Normalization**: Mengubah skala rating menjadi 0-1 untuk mempermudah proses training model.
4. **Data Splitting**: Membagi data menjadi 80% data latih dan 20% data validasi.

## Modeling and Result
### 1. Content-Based Filtering
Menggunakan **Cosine Similarity** untuk menghitung kemiripan antar anime berdasarkan genre yang telah diubah menjadi vektor TF-IDF.
- **Hasil**: Sistem berhasil merekomendasikan anime serupa. Contoh: Input "Naruto" menghasilkan output anime dengan genre Shounen/Action yang mirip.

### 2. Collaborative Filtering
Menggunakan **RecommenderNet** untuk memprediksi rating. Model ini mempelajari selera pengguna berdasarkan kemiripan dengan pengguna lain.
- **Kelebihan**: Dapat menemukan minat baru di luar genre yang biasa ditonton.
- **Kekurangan**: Membutuhkan banyak data rating (*Cold Start Problem*).

## Evaluation
### Content-Based Filtering
Menggunakan metrik **Precision**. 
- Rumus: $Precision = \frac{\text{Jumlah rekomendasi relevan}}{\text{Jumlah total rekomendasi}}$
- Hasil: Jika sistem memberikan 5 rekomendasi dan semuanya relevan dengan genre input, maka Precision = 100%.

### Collaborative Filtering
Menggunakan metrik **Root Mean Squared Error (RMSE)**.
- Rumus: $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
- Hasil: Model menunjukkan penurunan error selama 20 epoch, dengan nilai RMSE akhir yang cukup rendah (sekitar 0.1 - 0.2), menandakan prediksi rating yang akurat.
