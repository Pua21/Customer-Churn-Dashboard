# 📡 Telco Customer Churn Prediction & Analytics Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

Aplikasi analitik dan machine learning interaktif untuk mendeteksi pelanggan telekomunikasi yang berisiko berhenti berlangganan (*customer churn*). Dibangun menggunakan **Python**, **Scikit-Learn**, dan **Streamlit**, dashboard ini membantu tim operasional dan pemasaran memprioritaskan strategi retensi pelanggan secara proaktif berdasarkan data riil.

---

## 📌 Latar Belakang & Masalah Bisnis

Dalam industri telekomunikasi, biaya akuisisi pelanggan baru (*Customer Acquisition Cost*) dapat mencapai 5 hingga 25 kali lipat lebih tinggi dibandingkan mempertahankan pelanggan lama. Churn yang tidak terdeteksi sejak dini menyebabkan penurunan pendapatan secara langsung (*revenue leakage*).

Tantangan utama yang diselesaikan oleh dashboard ini:
1. **Identifikasi Dini:** Mendeteksi pelanggan yang memiliki indikasi kuat untuk berhenti sebelum mereka benar-benar memutus layanan.
2. **Faktor Pemicu:** Mengetahui fitur/variabel mana yang paling berkontribusi terhadap keputusan pelanggan untuk churn (misal: tingginya biaya panggilan harian atau frekuensi kontak ke layanan pelanggan).
3. **Tindakan Retensi Terukur:** Menyediakan simulasi profil individual serta daftar unduhan batch pelanggan berisiko tinggi untuk segera ditindaklanjuti tim pemasaran.

---

## 🌟 Fitur Utama Dashboard

- **📊 Exploratory Data Analysis (EDA):** Visualisasi interaktif distribusi churn, korelasi fitur pemakaian, pola waktu panggilan, serta perbandingan metrik utama.
- **🔮 Prediksi Pelanggan Tunggal (Single Customer Prediction):** Simulasi input profil nasabah dengan output probabilitas churn, kategori risiko (*Rendah, Sedang, Tinggi*), serta rekomendasi retensi otomatis.
- **📁 Pemrosesan Batch & Ekspor CSV:** Menilai risiko pelanggan massal secara simultan dan menyediakan tombol unduh daftar pelanggan berisiko tinggi (`pelanggan_target_retensi.csv`).
- **📈 Evaluasi & Feature Importance:** Perbandingan metrik model (*Confusion Matrix*, *ROC-AUC Curve*, dan bobot pengaruh fitur utama).

---

## 🔬 Spesifikasi & Metrik Model Machine Learning

Model klasifikasi dilatih menggunakan pipeline `scikit-learn` dengan penanganan *imbalanced dataset* (`class_weight="balanced"`):

| Algoritma | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (Selected)** | **~94.8%** | **~85.2%** | **~79.5%** | **~82.2%** | **~0.92** |
| Decision Tree | ~91.2% | ~70.5% | ~68.4% | ~69.4% | ~0.83 |
| Logistic Regression | ~78.4% | ~35.6% | ~74.2% | ~48.1% | ~0.81 |

> **Catatan:** Model Random Forest dipilih sebagai model utama produksi karena memiliki performa seimbang antara meminimalkan *false alarm* (Precision) dan menangkap sebanyak mungkin pelanggan yang benar-benar akan churn (Recall).

---

## 📂 Struktur Direktori Proyek

```text
customer-churn-dashboard/
├── .streamlit/
│   └── config.toml         # Konfigurasi tema gelap & server Streamlit
├── Analisis_Customer_Churn.ipynb  # Notebook eksplorasi data & eksperimen
├── app.py                  # Aplikasi dashboard utama Streamlit
├── train_model.py          # Script pelatihan dan ekspor model (.pkl)
├── model_rf_bigml.pkl      # Artefak model Random Forest & pipeline encoder
├── bigml_59c28831336c6604c800002a.csv # Dataset historis telekomunikasi
├── requirements.txt        # Daftar dependensi library Python
├── .gitignore              # Filter file temporary & cache
└── README.md               # Dokumentasi proyek
```

---

## 🚀 Panduan Menjalankan secara Lokal

### 1. Prasyarat
- Python 3.10 atau versi yang lebih baru
- Git

### 2. Kloning Repository
```bash
git clone https://github.com/Pua21/customer-churn-dashboard.git
cd customer-churn-dashboard
```

### 3. Buat Virtual Environment (Disarankan)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 4. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 5. Melatih Model (Opsional, file .pkl sudah tersedia)
```bash
python train_model.py
```

### 6. Jalankan Dashboard Streamlit
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

---

## 👨‍💻 Pengembang

- **Ignatius Papuandri Mansawan**
- Universitas Katolik Soegijapranata Semarang
- Program Studi: S1 Sistem Informasi
- GitHub: [@Pua21](https://github.com/Pua21)
- LinkedIn: [Ignatius Mansawan](https://www.linkedin.com/in/ignatius-mansawan)
