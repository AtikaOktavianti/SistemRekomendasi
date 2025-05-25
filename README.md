# Sistem Rekomendasi - Movie Recommendation System

Membangun sistem rekomendasi film menggunakan pendekatan Content-Based Filtering dan Collaborative Filtering

## 1. Project Overview

### Latar Belakang
Dalam era digital saat ini, pengguna memiliki akses ke ribuan film melalui berbagai platform. Namun, banyaknya pilihan justru menciptakan tantangan dalam menemukan film yang sesuai dengan preferensi individu. Oleh karena itu, sistem rekomendasi menjadi solusi penting untuk membantu pengguna menemukan konten yang relevan.

Proyek ini bertujuan membangun sistem rekomendasi film dengan pendekatan:
- **Content-Based Filtering**, yang merekomendasikan film berdasarkan kesamaan konten.
- **Collaborative Filtering**, yang menggunakan perilaku pengguna untuk memberikan rekomendasi.

### Relevansi dan Pentingnya Proyek
Meningkatkan pengalaman pengguna dengan menyarankan film yang relevan dapat meningkatkan engagement serta kepuasan pengguna. Sistem ini banyak digunakan oleh platform seperti Netflix, Disney+, dan Amazon Prime.

---

## 2. Business Understanding

### Problem Statements
1. Bagaimana cara merekomendasikan film yang relevan bagi pengguna?
2. Bagaimana memanfaatkan data rating dan informasi konten film untuk menyusun sistem rekomendasi yang efektif?

### Goals
- Membangun dua sistem rekomendasi: content-based dan collaborative.
- Menyajikan rekomendasi film yang sesuai preferensi pengguna.
- Mengevaluasi performa sistem dengan metrik yang sesuai.

### Solution Approach

Untuk mencapai tujuan proyek yaitu membuat sistem rekomendasi film, dilakukan dua pendekatan yaitu: 
1. **Content-Based Filtering**
   - Menggunakan **TF-IDF Vectorizer** pada kolom genre.
   - Menghitung kemiripan antar film menggunakan **Cosine Similarity**.

2. **Collaborative Filtering**
   - Menggunakan algoritma **Singular Value Decomposition (SVD)** dari pustaka Surprise.
   - Berdasarkan interaksi pengguna dalam bentuk rating.

---

## 3. Data Understanding

### Dataset Overview
Dataset ini diambil dari Kaggle:
https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system

Dataset tersebut terdiri dari dua file utama:

#### a. movies.csv
- Jumlah data: 62.432 baris × 3 kolom
- Kondisi data:
  - Missing value: 0
  - Duplikat: Tidak ada
  - Outlier: Tidak relevan karena semua data kategori
- Fitur:

| Fitur | Deskripsi |
| ------ | ------ |
| MovieId | ID unik untuk setiap film |
| title | Judul film |
| genres | Genre film, dipisahkan dengan tanda | |

Tabel 1. Fitur dataset movie.csv

#### b. ratings.csv
- Jumlah data: 25.000.095 baris × 4 kolom
- Kondisi data:
  - Missing value: Ada pada sebagian kecil data
  - Duplikat: Ada dan sudah dibersihkan
  - Outlier: Tidak signifikan, rating dalam skala 0.5–5
- Fitur:

| Fitur | Deskripsi |
| ------ | ------ |
| userId | ID unik pengguna yang memberikan rating |
| movieId | ID film yang diberi rating |
| rating | Nilai rating, biasanya skala 0.5 sampai 5 |
| timestamp | Waktu rating diberikan dalam format UNIX |

Tabel 2. Fitur dataset rating.csv

---

## 4. Data Preparation
Pada tahap ini, dilakukan beberapa teknik data preparation untuk mempersiapkan dataset sebelum digunakan dalam model sistem rekomendasi. Teknik-teknik tersebut dilakukan secara berurutan sebagai berikut:

### 1. Handling Missing Value
- Dataset movies: kolom genres yang kosong diisi dengan string kosong ('') agar kompatibel dengan TF-IDF.
- Dataset ratings: baris dengan nilai kosong pada userId, movieId, atau rating dihapus karena ketiganya bersifat wajib untuk training model collaborative filtering.

Kode yang digunakan 
```
movies.isnull().sum()
ratings.isnull().sum()
movies['genres'] = movies['genres'].fillna('')
```

**Penjelasan**: Nilai kosong pada kolom genres di dataset movies diisi dengan string kosong ('') agar tidak error saat digunakan oleh TF-IDF Vectorizer. Sementara untuk dataset ratings, baris yang memiliki missing value pada kolom userId, movieId, atau rating dihapus karena kolom-kolom ini esensial dalam proses pelatihan model collaborative filtering.

### 2. Handling Duplicates
Kode yang digunakan 
```
ratings = ratings.drop_duplicates()
```

**Penjelasan**: Baris duplikat pada dataset ratings dihapus menggunakan fungsi drop_duplicates() untuk menghindari bias data dan overfitting pada collaborative filtering. Satu user tidak boleh memengaruhi bobot prediksi lebih dari sekali untuk film yang sama dengan rating yang sama.

### 3. Handling Outliers
Penanganan outlier tidak dilakukan karena fitur utama berupa rating memiliki domain terbatas dan terstandarisasi. Tidak ditemukan nilai di luar rentang 0.5–5.0

### 4. Content-Based Filtering Preparation
Kode yang digunakan 
```
movies['genres_str'] = movies['genres'].str.replace('|', ' ')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_str'])
```

**Penjelasan**:

- Mengganti karakter pemisah '|' menjadi spasi agar setiap genre dikenali sebagai token berbeda.
- Membangun TF-IDF matrix dari kolom genres_str.
- Matriks ini menjadi dasar untuk menghitung cosine similarity antar film.

### 5. Collaborative Filtering Preparation
Kode yang digunakan 
```
from surprise import Reader, Dataset
reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
```

**Penjelasan**: 

- Mengatur skala rating dinamis berdasarkan data.
- Memuat data ke dalam format Dataset Surprise untuk digunakan pada algoritma collaborative filtering (SVD).
- Melakukan validasi silang menggunakan cross_validate untuk mengukur performa model.


## 5. Modeling and Result

### Pendekatan dan Permasalahan
Model sistem rekomendasi dikembangkan dengan dua pendekatan utama, yaitu Content-Based Filtering dan Collaborative Filtering. Masing-masing pendekatan menggunakan teknik yang berbeda untuk menghasilkan rekomendasi film yang relevan bagi pengguna.

**1. Content-Based Filtering**

**definisi dan Cara Kerja**: Content-Based Filtering merekomendasikan produk berdasarkan kemiripan konten antar item. Dalam implementasi ini:
- Deskripsi produk diubah menjadi vektor fitur menggunakan teknik **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- Kemudian, dihitung **cosine similarity** antar vektor produk.
- Produk-produk yang paling mirip dengan produk yang telah disukai oleh pengguna akan direkomendasikan kembali.

**Top-N Recommendation**: 
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/cbf-10.png?raw=true)

**Penjelasan**: 

Gambar tersebut menunjukkan daftar 10 film yang direkomendasikan berdasarkan kemiripan konten dengan film "Toy Story (1995)". Rekomendasi ini dihitung menggunakan cosine similarity pada fitur TF-IDF dari deskripsi film, sehingga film-film yang muncul memiliki konten atau tema yang mirip dengan "Toy Story (1995)". Film-film tersebut diurutkan dari yang paling mirip hingga yang kurang mirip dalam 10 teratas.

**Kelebihan**:
- Tidak memerlukan data dari pengguna lain, cukup metadata dari item.
- Dapat memberikan rekomendasi untuk item yang belum pernah di-rating (cold-start item).

**Kekurangan**:
- Cenderung sempit, hanya merekomendasikan item yang serupa (kurang beragam).
- Tidak mempertimbangkan opini atau preferensi pengguna lain.

**2. Collaborative Filtering (SVD - Singular Value Decomposition)**

**Definisi dan Cara Kerja**: Collaborative Filtering merekomendasikan produk berdasarkan perilaku pengguna lain yang memiliki pola mirip. Pada notebook ini digunakan:
- Algoritma **Singular Value Decomposition (SVD)** dari pustaka `Surprise`.
- SVD mengurangi dimensi data interaksi pengguna-produk dan memperkirakan rating yang tidak diketahui.

**Top-N Recommendation**: Berikut adalah contoh hasil Top-10 rekomendasi produk untuk pengguna
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/userid1.png?raw=true)

**Penjelasan**: Pada gambar tersebut merupakan hasil dari sistem rekomendasi berbasis Collaborative Filtering menggunakan algoritma SVD. Sistem merekomendasikan 10 film terbaik untuk pengguna dengan `userId=1` berdasarkan pola rating pengguna tersebut dan kesamaan preferensi dengan pengguna lain. Film-film yang ditampilkan adalah yang diprediksi akan paling disukai oleh pengguna karena memiliki prediksi rating tertinggi dari model.

**Kelebihan**: 
- Mempertimbangkan pola rating dari banyak pengguna.
- Dapat menemukan keterkaitan yang tidak tampak langsung dalam metadata film.

**Kekurangan**:
- Membutuhkan cukup banyak data rating (masalah cold-start pada pengguna baru).
- Lebih kompleks dan memerlukan lebih banyak sumber daya komputasi.

---

## 6. Evaluation

### Pendekatan 1: Content-Based Filtering

Disini saya merekomendasikan film Toy Story (1995)

Hasil dari Top 5 dari film atau movie yang saya rekomendasikan adalah sebagai berikut :
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/top5.png?raw=true)

**Penjelasan** : Sistem rekomendasi content-based memberikan 5 film teratas yang mirip dengan Toy Story (1995). Dari 5 film yang direkomendasikan, 3 film ditemukan dalam daftar film relevan (ground truth), yaitu:
- Pagemaster, The (1994)
- Kids of the Round Table (1995)
- Space Jam (1996)
- Jumanji (1995)
- Indian in the Cupboard, The (1995)

Namun hanya 3 film yang cocok dengan daftar relevant_movies, sehingga:
Precision@5 = 3 relevan / 5 rekomendasi = 0.60

Artinya, 60% rekomendasi yang diberikan sistem terbukti relevan berdasarkan daftar acuan, menunjukkan performa sistem yang cukup baik.

Teknik Evaluasi di atas adalah dengan menggunakan precission, rumus dari teknik ini adalah :
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/rumuscbs.png?raw=true)

### Pendekatan 2: Collaborative Filtering (SVD)

#### Metrik Evaluasi yang Digunakan
Evaluasi dilakukan secara **kuantitatif** dengan dua metrik:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

Keduanya digunakan untuk mengukur seberapa dekat prediksi model terhadap rating sebenarnya yang diberikan pengguna.

#### Penjelasan Metrik

- **RMSE**:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

- **MAE**:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

RMSE lebih sensitif terhadap error besar, sedangkan MAE memberikan estimasi rata-rata kesalahan model.

#### Hasil Evaluasi
Berdasarkan hasil validasi silang:
- **RMSE: 0.7812**
- **MAE: 0.5894**

Nilai ini menunjukkan bahwa model cukup akurat dalam memprediksi rating pengguna terhadap film.

#### Kesesuaian Metrik dengan Konteks
- Relevan digunakan dalam sistem rekomendasi berbasis rating.
- Menilai performa prediksi terhadap preferensi pengguna.
- Mendukung strategi personalisasi rekomendasi.

---
## Hubungan Model dengan Business Understanding

### 1. Apakah sudah menjawab setiap problem statement?

**Problem Statement 1:** _Bagaimana cara merekomendasikan film yang relevan bagi pengguna?_  
**Terjawab:**  
Model content-based dan collaborative filtering dibangun untuk menghasilkan rekomendasi film yang relevan berdasarkan kesamaan konten dan perilaku pengguna.

**Problem Statement 2:** _Bagaimana memanfaatkan data rating dan informasi konten film untuk menyusun sistem rekomendasi yang efektif?_  
**Terjawab:**  
- **Content-Based Filtering** menggunakan TF-IDF pada data genre film.  
- **Collaborative Filtering** menggunakan data rating dengan algoritma SVD.

### 2. Apakah berhasil mencapai setiap goals yang diharapkan?

**Goal 1:** _Membangun dua sistem rekomendasi: content-based dan collaborative._  
**Tercapai.** Kedua pendekatan berhasil diimplementasikan.

**Goal 2:** _Menyajikan rekomendasi film yang sesuai preferensi pengguna._  
**Tercapai.** Rekomendasi dihasilkan untuk film dan pengguna spesifik.

**Goal 3:** _Mengevaluasi performa sistem dengan metrik yang sesuai._  
**Tercapai.**  
- **Content-Based:** Precision@5 = 0.60  
- **Collaborative Filtering:** RMSE = 0.7812, MAE = 0.5894

### 3. Apakah setiap solusi statement yang kamu rencanakan berdampak?

Ya, kedua pendekatan memberikan dampak nyata terhadap tujuan bisnis:

#### Content-Based Filtering
- **Dampak:** Dapat memberikan rekomendasi untuk film baru (cold-start item).
- **Manfaat bisnis:** Menjaga ketertarikan pengguna baru melalui rekomendasi yang relevan.

#### Collaborative Filtering
- **Dampak:** Memberikan rekomendasi personal berdasarkan preferensi pengguna lain.
- **Manfaat bisnis:** Meningkatkan pengalaman pengguna jangka panjang dan loyalitas.

## Kesimpulan

Model yang dibangun sangat relevan terhadap _business understanding_ proyek ini:
- Semua **problem statement** dijawab.
- **Tujuan bisnis** tercapai dan divalidasi secara metrik.
- **Solusi yang diimplementasikan berdampak nyata** pada peningkatan kepuasan pengguna.

---

## Referensi
1. Larasati, F. B. A., & Februariyanti, H. (2021). Sistem Rekomendasi Produk Emina Cosmetics dengan Menggunakan Metode Content-Based Filtering. Jurnal Manajemen Informatika dan Sistem Informasi, 4(1), 45–54.
https://e-journal.upr.ac.id/index.php/JTI/article/view/12543
2. Putri, M. W., & Wibowo, A. T. (2024). Content-Based Filtering pada Sistem Rekomendasi Buku Informatika. Jurnal Ilmiah SINUS (JIS), 22(2), 58–64.
https://doi.org/10.30646/sinus.v22i2.840
3. Rachmat, R. (2024). Analysis of Algorithms and Data Processing Efficiency in Movie Recommendation Systems. Jurnal Mandiri IT, 13(2), 273–279. https://ejournal.isha.or.id/index.php/Mandiri/article/download/358/388/2234
4. Setiawan, A. H., & Kurniawan, I. (2021). Penerapan Collaborative Filtering untuk Rekomendasi Produk di Platform E-Commerce. Jurnal Ilmu Komputer AMIKOM, 7(1), 71–78. https://ejournal.amikom.ac.id/index.php/jik/article/view/1234
5. Wulandari, F., & Hermawan, D. (2019). Perbandingan Collaborative Filtering dan Content-Based Filtering untuk Sistem Rekomendasi Buku. Jurnal Ilmiah Teknik Informatika Komputer (JITIK), 5(1), 23–30. https://jitik.stmikjayakarta.ac.id/index.php/jitik/article/view/1904
