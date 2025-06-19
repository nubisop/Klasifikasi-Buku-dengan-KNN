"""
File: proyek_knn.py

Program utama untuk klasifikasi kategori buku menggunakan algoritma k-NN berbasis TF-IDF.
Dataset berisi judul dan kategori buku. Program melakukan preprocessing, pelatihan model,
dan prediksi kategori untuk judul buku baru.
"""

import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

def preprocess_judul(judul):
    """
    Membersihkan teks judul buku:
    - Mengubah huruf menjadi lowercase.
    - Menghapus tanda baca.
    Args:
        judul (str): Judul buku.
    Returns:
        str: Judul yang sudah dibersihkan.
    """
    return judul.lower().translate(str.maketrans('', '', string.punctuation))

def jalankan_klasifikasi_knn():
    """
    Menjalankan proses klasifikasi kategori buku:
    1. Membuat dataset judul dan kategori buku.
    2. Preprocessing teks judul.
    3. Vektorisasi judul dengan TF-IDF.
    4. Melatih model k-NN.
    5. Melakukan prediksi kategori pada judul buku baru.
    """
    data_buku = [
        {'judul': 'Belajar Pemrograman Python untuk Pemula', 'kategori': 'Teknologi'},
        {'judul': 'Dasar-Dasar Jaringan Komputer dan Internet', 'kategori': 'Teknologi'},
        {'judul': 'Panduan Lengkap Machine Learning dengan Scikit-Learn', 'kategori': 'Teknologi'},
        {'judul': 'Cloud Computing dan Teknologi Virtualisasi', 'kategori': 'Teknologi'},
        {'judul': 'Pengantar Keamanan Siber', 'kategori': 'Teknologi'},
        {'judul': 'Bumi Manusia dan Jejak Langkah', 'kategori': 'Fiksi'},
        {'judul': 'Laskar Pelangi Sang Pemimpi', 'kategori': 'Fiksi'},
        {'judul': 'Kisah Tanah Jawa', 'kategori': 'Fiksi'},
        {'judul': 'Pulang Pergi Naik Haji', 'kategori': 'Fiksi'},
        {'judul': 'Cerita dari Blora', 'kategori': 'Fiksi'},
        {'judul': 'Sejarah Singkat Waktu dan Alam Semesta', 'kategori': 'Sains'},
        {'judul': 'Kosmos Carl Sagan Sebuah Perjalanan', 'kategori': 'Sains'},
        {'judul': 'Asal-Usul Spesies Charles Darwin', 'kategori': 'Sains'},
        {'judul': 'Teori Relativitas Einstein', 'kategori': 'Sains'},
        {'judul': 'Kehidupan di Bawah Mikroskop', 'kategori': 'Sains'},
        {'judul': 'Cara Cerdas Mengelola Keuangan Pribadi', 'kategori': 'Bisnis'},
        {'judul': 'Strategi Pemasaran Digital untuk Bisnis UKM', 'kategori': 'Bisnis'},
        {'judul': 'Berpikir Seperti Seorang CEO', 'kategori': 'Bisnis'},
        {'judul': 'Seni Negosiasi dan Komunikasi Bisnis', 'kategori': 'Bisnis'},
        {'judul': 'Investasi Saham untuk Pemula', 'kategori': 'Bisnis'}
    ]
    df = pd.DataFrame(data_buku)

    df['judul_bersih'] = df['judul'].apply(preprocess_judul)
    
    vectorizer = TfidfVectorizer()
    X = df['judul_bersih']
    y = df['kategori']
    X_vec = vectorizer.fit_transform(X)

    k = 3
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_vec, y)

    buku_baru = [
        'Pemasaran Saham untuk Pemula',
        'Pemrograman Python untuk Pemula',
        'Informasi Teknologi Terkini',
        'Belajar JavaScript untuk Web',
        'Kisah Perahu Kertas',
        'Cinta dalam Ayat-Ayat',
        'Fisika SMA Modern',
        'Matematika dan Kehidupan',
        'Mengelola Keuangan Pribadi dengan Cerdas',
        'Sukses Bisnis Online',
        'Panduan Investasi Saham',
        'Manajemen SDM Modern',
        'Marketing Digital Masa Kini'
    ]
    
    buku_baru_bersih = [preprocess_judul(judul) for judul in buku_baru]
    vektor_buku_baru = vectorizer.transform(buku_baru_bersih)
    prediksi_kategori = model.predict(vektor_buku_baru)

    for i in range(len(buku_baru)):
        print(f"\nJudul: '{buku_baru[i]}'")
        print(f"--> Prediksi Kategori: {prediksi_kategori[i]}")

if __name__ == "__main__":
    jalankan_klasifikasi_knn()
