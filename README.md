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

### Referensi Dataset
Dataset diambil dari Kaggle:  
[Movie Recommendation System - Kaggle](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)

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

### Deskripsi Data
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/isidata.png?raw=true)

Gambar 1. Informasi Dataset

Dataset terdiri dari dua file utama:
1. `movies.csv`: berisi informasi tentang film.
- Jumlah baris: 62,432
- Jumlah kolom: 3
  
| Fitur | Deskripsi |
| ------ | ------ |
| MovieId | ID unik untuk setiap film |
| title | Judul film |
| genres | Genre film, dipisahkan dengan tanda | |

Tabel 1. Fitur dataset movie.csv

2. `ratings.csv`: berisi rating dari pengguna terhadap film.
- Jumlah baris: 25,000,095
- Jumlah kolom: 4
  
| Fitur | Deskripsi |
| ------ | ------ |
| userId | ID unik pengguna yang memberikan rating |
| movieId | ID film yang diberi rating |
| rating | Nilai rating, biasanya skala 0.5 sampai 5 |
| timestamp | Waktu rating diberikan dalam format UNIX |

Tabel 2. Fitur dataset rating.csv

### Eksplorasi Data Analisis (EDA)
**1. Statistik dasar rating**

|  |  |
| ------ | ------ |
| count | 2.500010e+07 |
| mean | 3.533854e+00 |
| std | 1.060744e+00 |
| min | 5.000000e-01 |
| 25% | 3.000000e+00 |
| 50% | 3.500000e+00 |
| 75% | 4.000000e+00 |
| max | 5.000000e+00 |

Tabel 3 Rating statistik

Pada tabel 3 Statistik rating menunjukkan bahwa dari total 25 juta data, **rata-rata rating** film adalah **3.53** dengan **standar deviasi** sebesar **1.06**, yang menandakan variasi penilaian pengguna. Nilai rating minimum adalah **0.5** dan maksimum **5.0**. Sebagian besar rating berada di antara **3.0 (kuartil 1)** dan **4.0 (kuartil 3)**, dengan **nilai tengah (median)** di **3.5**, menunjukkan bahwa pengguna cenderung memberikan rating positif

**2. Jumalh rating per film (top 10)**

|  |  |
| ------ | ------ |
| 356 | | 81491 |
| 318 | 81482 |
| 296 | 79672 |
| 593 | 74127 |
| 2571 | 72674 |
| 260 | 68717 |
| 480 | 64144 |
| 527 | 60411 |
| 110 | 59184 |
| 2959 | 58773 |

Tabel 4 Jumlah rating per film (top 10) 

Pada tabel 4, menunjukkan **10 film teratas dengan jumlah rating terbanyak**. Film dengan `movieId 356` menerima rating paling banyak, yaitu **81.491 kali**, diikuti oleh `movieId 318` dengan **81.482 rating**, dan seterusnya. Ini menunjukkan bahwa film-film tersebut sangat populer atau sering ditonton oleh pengguna dalam dataset.

**3. Visualisasi distribusi rating**
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/distibusi%20rating.png?raw=true)

Gambar 2. Distribusi Rating

Gambar 2, menunjukkan  **distribusi rating film** dari pengguna. Terlihat bahwa sebagian besar pengguna memberikan **rating tinggi**, terutama pada **rating 4.0** yang mendominasi, disusul oleh **rating 3.0** dan **5.0**. Sebaliknya, rating rendah seperti **0.5 hingga 2.0** jauh lebih jarang diberikan. Ini menunjukkan bahwa pengguna cenderung lebih sering memberi penilaian positif terhadap film yang mereka tonton.

**4. Visualisasi jumlah rating per film (top 20)**
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/top20.png?raw=true)

Gambar 3 Jumalah rating perfilm (top 20)

Gambar 3, menampilkan **20 film dengan jumlah rating terbanyak** berdasarkan `movieId`. Film dengan `movieId` **356 dan 318** menerima rating terbanyak, masing-masing lebih dari **80.000 kali**, diikuti oleh film `296`, `593`, dan `260`. Secara umum, semua film dalam grafik ini mendapatkan lebih dari **50.000 rating**, menunjukkan bahwa film-film tersebut sangat populer atau sering ditonton oleh pengguna dalam dataset.

**5. Eksplorasi dan visualisasi genre film**
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/distribusi%20genre.png?raw=true)

Gambar 4 Distribusi Genre Film

Gambar 4, menunjukkan **distribusi jumlah film berdasarkan genre**. Genre **Drama** merupakan yang paling banyak, dengan lebih dari **25.000 film**, diikuti oleh **Comedy**, **Thriller**, dan **Romance**. Genre dengan jumlah film paling sedikit adalah **IMAX**, **Film-Noir**, dan **Musical**. Ini menunjukkan bahwa film bergenre drama dan komedi paling dominan dalam dataset, sedangkan genre-genre tertentu bersifat lebih niche atau jarang diproduksi.

## 4. Data Preparation
Pada tahap ini, dilakukan beberapa teknik data preparation untuk mempersiapkan dataset sebelum digunakan dalam model sistem rekomendasi. Teknik-teknik tersebut dilakukan secara berurutan sebagai berikut:
### 1. Pengecekan dan Penananan Missing Value
Kode yang digunakan 
```
print("Missing value pada movies:")
print(movies.isnull().sum())

print("\nMissing value pada ratings:")
print(ratings.isnull().sum())
```

**Penjelasan**: Langkah pertama adalah memeriksa apakah terdapat nilai kosong (missing values) pada kolom-kolom dataset movies dan ratings. Nilai kosong bisa mengganggu proses feature extraction dan pelatihan model.

**Alasan**: Tahapan ini penting karena model machine learning dan algoritma text processing tidak dapat menangani data yang hilang. Deteksi dini membantu dalam menentukan strategi penanganan seperti imputasi atau penghapusan.

### 2. Imputasi Missing Value pada Kolom genres
Kode yang digunakan 
```
movies['genres'] = movies['genres'].fillna('')
```

**Penjelasan**: Nilai kosong pada kolom genres diisi dengan string kosong ('') agar bisa diproses oleh TfidfVectorizer.

**Alasan**: Agar semua data dapat diproses secara seragam oleh teknik representasi teks berbasis TF-IDF, kolom teks tidak boleh mengandung NaN.

### 3. Preprocessing Teks Genre
Kode yang digunakan 
```
movies['genres_str'] = movies['genres'].str.replace('|', ' ')
```

**Penjelasan**: Tanda pemisah genre (|) diganti menjadi spasi agar TF-IDF dapat mengenali setiap genre sebagai kata yang berbeda.

**Alasan**: TF-IDF bekerja dengan asumsi bahwa teks dipisahkan oleh spasi. Jika tidak diubah, TF-IDF akan memperlakukan “Action|Adventure” sebagai satu kata.

### 4. Transformasi Teks Genre dengan TF-IDF
Kode yang digunakan 
```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_str'])
```

**Penjelasan**: Digunakan TfidfVectorizer untuk mengubah teks genre menjadi matriks fitur numerik berbasis bobot TF-IDF.

**Alasan**: Representasi numerik sangat penting untuk digunakan dalam model machine learning berbasis content-based filtering.

### 5. Menghapus Duplikasi dan Missing Value pada Ratings
Kode yang digunakan 
```
ratings = ratings.drop_duplicates()
ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
```

**Penjelasan**: Duplikat dan baris dengan nilai kosong dihapus agar tidak mengganggu proses collaborative filtering.

**Alasan**: Data yang bersih dan unik akan meningkatkan performa dan keandalan model rekomendasi.

### 6. Menentukan Skala Rating dan Mengonversi ke Format Surprise
Kode yang digunakan 
```
from surprise import Reader, Dataset
reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
```

**Penjelasan**: Skala rating ditentukan secara dinamis dari data, lalu dataset disiapkan untuk digunakan oleh library Surprise.

**Alasan**: Library Surprise membutuhkan format dan skala data yang spesifik untuk dapat menjalankan algoritma collaborative filtering.

---

## 5. Modeling and Result

### Pendekatan dan Permasalahan
Sistem rekomendasi ini bertujuan untuk membantu pengguna dalam menemukan film yang relevan dengan preferensi mereka berdasarkan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering. Masalah yang dipecahkan adalah membantu pengguna memilih film dari jumlah pilihan yang sangat besar, dengan menyajikan rekomendasi yang relevan dan dipersonalisasi.

**1. Content-Based Filtering**
- **Modelling**
Pendekatan ini memanfaatkan informasi konten dari film, khususnya kolom genre. Prosesnya meliputi:
- Ekstraksi fitur teks menggunakan TF-IDF Vectorizer.
- Perhitungan kesamaan antar film menggunakan Cosine Similarity.
- Pemberian rekomendasi berdasarkan film yang mirip dengan input pengguna.

- **Top-N Recommendation**
Sebagai contoh, jika pengguna memilih film tertentu, sistem akan merekomendasikan 10 film teratas yang paling mirip berdasarkan genre-nya

- **Kelebihan**
- Tidak memerlukan data dari pengguna lain, cukup metadata dari item.
- Dapat memberikan rekomendasi untuk item yang belum pernah di-rating (cold-start item).

- **Kekurangan**
- Cenderung sempit, hanya merekomendasikan item yang serupa (kurang beragam).
- Tidak mempertimbangkan opini atau preferensi pengguna lain.

**2. Collaborative Filtering (SVD - Singular Value Decomposition)**
- **Modelling**
Menggunakan pendekatan Matrix Factorization dengan algoritma SVD dari library Surprise. Langkah-langkah:
- Data rating pengguna diformat menggunakan Reader dan Dataset dari Surprise.
- Model dilatih menggunakan SVD.
- Evaluasi menggunakan cross-validation dengan metrik RMSE dan MAE.

- **Top-N Recommendation**
Model ini digunakan untuk memprediksi rating film yang belum ditonton pengguna, dan menyarankan 10 film teratas dengan prediksi rating tertinggi.

- **Kelebihan**
- Mempertimbangkan pola rating dari banyak pengguna.
- Dapat menemukan keterkaitan yang tidak tampak langsung dalam metadata film.

- **Kekurangan**
- Membutuhkan cukup banyak data rating (masalah cold-start pada pengguna baru).
- Lebih kompleks dan memerlukan lebih banyak sumber daya komputasi.

---

## 6. Evaluation

### Pendekatan 1: Content-Based Filtering

#### Metrik Evaluasi yang Digunakan
Evaluasi dilakukan secara **kualitatif** dengan menggunakan **Cosine Similarity** sebagai metrik internal untuk mengukur kesamaan antar item berdasarkan fitur kontennya (genre film). Karena tidak tersedia data interaksi pengguna atau relevansi eksplisit, metrik seperti *precision* atau *recall* belum digunakan.

#### Penjelasan Metrik
**Cosine Similarity** menghitung tingkat kemiripan antara dua vektor (dalam hal ini representasi TF-IDF dari genre film) dengan rumus:

\[
\text{Cosine Similarity} = \frac{A \cdot B}{\|A\|\|B\|}
\]

Nilainya berkisar antara 0 (tidak mirip) hingga 1 (sangat mirip). Film dengan nilai cosine similarity tertinggi terhadap film pilihan pengguna akan direkomendasikan.

#### Hasil Evaluasi
Sebagai contoh, ketika pengguna memilih film *Toy Story (1995)*, sistem merekomendasikan lima film dengan deskripsi yang mirip, seperti:

1. The Pagemaster (1994)
2. Kids of the Round Table (1995)
3. Space Jam (1996)
4. Jumanji (1995)
5. Indian in the Cupboard, The (1995)

Film-film tersebut memiliki unsur cerita yang mirip, seperti petualangan, fantasi, dan tema ramah anak-anak, yang secara logis relevan bagi penonton *Toy Story*. Hal ini menunjukkan bahwa model berhasil menangkap kemiripan konten secara semantik.

#### Kesesuaian Metrik dengan Konteks
- Cosine similarity sangat sesuai digunakan untuk pendekatan berbasis konten.
- Efektif digunakan saat tidak tersedia data interaksi pengguna.
- Cocok untuk mengatasi cold-start item dan user baru.

---

### Pendekatan 2: Collaborative Filtering (SVD)

#### Metrik Evaluasi yang Digunakan
Evaluasi dilakukan secara **kuantitatif** dengan dua metrik:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

Keduanya digunakan untuk mengukur seberapa dekat prediksi model terhadap rating sebenarnya yang diberikan pengguna.

#### Penjelasan Metrik

- **RMSE**:
\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
\]

- **MAE**:
\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
\]

RMSE lebih sensitif terhadap error besar, sedangkan MAE memberikan estimasi rata-rata kesalahan model.

#### Hasil Evaluasi
Berdasarkan hasil validasi silang:
- **RMSE: 0.7814**
- **MAE: 0.5896**

Nilai ini menunjukkan bahwa model cukup akurat dalam memprediksi rating pengguna terhadap film.

#### Kesesuaian Metrik dengan Konteks
- Relevan digunakan dalam sistem rekomendasi berbasis rating.
- Menilai performa prediksi terhadap preferensi pengguna.
- Mendukung strategi personalisasi rekomendasi.

---

### Kesimpulan Evaluasi

| Aspek                       | Content-Based Filtering          | Collaborative Filtering (SVD)     |
|----------------------------|----------------------------------|-----------------------------------|
| **Jenis Evaluasi**         | Kualitatif                       | Kuantitatif                       |
| **Metrik**                 | Cosine Similarity                | RMSE, MAE                         |
| **Output**                 | Rekomendasi mirip secara isi     | Prediksi rating dari komunitas    |
| **Kelebihan**              | Cocok untuk item baru            | Mencerminkan preferensi kolektif  |
| **Kekurangan**             | Tidak mempertimbangkan user lain | Tidak cocok untuk cold-start user |

Dua pendekatan ini saling melengkapi: content-based cocok untuk rekomendasi awal, sedangkan collaborative filtering unggul dalam personalisasi berdasarkan pola komunitas pengguna.


---

## Referensi
1. Larasati, F. B. A., & Februariyanti, H. (2021). Sistem Rekomendasi Produk Emina Cosmetics dengan Menggunakan Metode Content-Based Filtering. Jurnal Manajemen Informatika dan Sistem Informasi, 4(1), 45–54.
https://e-journal.upr.ac.id/index.php/JTI/article/view/12543
3. Putri, M. W., & Wibowo, A. T. (2024). Content-Based Filtering pada Sistem Rekomendasi Buku Informatika. Jurnal Ilmiah SINUS (JIS), 22(2), 58–64.
https://doi.org/10.30646/sinus.v22i2.840
4. Rachmat, R. (2024). Analysis of Algorithms and Data Processing Efficiency in Movie Recommendation Systems. Jurnal Mandiri IT, 13(2), 273–279. https://ejournal.isha.or.id/index.php/Mandiri/article/download/358/388/2234
5. Setiawan, A. H., & Kurniawan, I. (2021). Penerapan Collaborative Filtering untuk Rekomendasi Produk di Platform E-Commerce. Jurnal Ilmu Komputer AMIKOM, 7(1), 71–78. https://ejournal.amikom.ac.id/index.php/jik/article/view/1234
6. Wulandari, F., & Hermawan, D. (2019). Perbandingan Collaborative Filtering dan Content-Based Filtering untuk Sistem Rekomendasi Buku. Jurnal Ilmiah Teknik Informatika Komputer (JITIK), 5(1), 23–30. https://jitik.stmikjayakarta.ac.id/index.php/jitik/article/view/1904
