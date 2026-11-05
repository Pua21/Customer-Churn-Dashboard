"""
train_model.py  —  Pipeline pelatihan model Customer Churn
===========================================================
Urutan menjalankan:
  1. python train_model.py   -> menghasilkan model_rf_bigml.pkl
  2. streamlit run app.py    -> membuka dashboard

Keputusan desain:
- Pipeline sklearn digunakan agar preprocessing (OrdinalEncoder) dan model
  dikemas dalam SATU objek; app.py tidak perlu encoder manual.
- class_weight="balanced" untuk menangani imbalanced class (churn ~14.5%).
- stratify=y agar distribusi kelas sama di train & test.
- Membandingkan 3 model: Logistic Regression, Decision Tree, Random Forest.
- Model terbaik (berdasarkan ROC-AUC) disimpan ke pkl.
"""

import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. BACA & BERSIHKAN DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  PIPELINE PELATIHAN MODEL CUSTOMER CHURN")
print("=" * 60)

DATASET_PATH = "bigml_59c28831336c6604c800002a.csv"
TARGET_COL = "churn"
DROP_COLS = ["phone number"]  # identifier unik, tidak berguna untuk prediksi

print(f"\n[1/6] Membaca dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"      Dimensi awal    : {df.shape[0]} baris x {df.shape[1]} kolom")

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# Ubah target ke integer (False->0, True->1)
df[TARGET_COL] = df[TARGET_COL].map(
    {False: 0, True: 1, "False": 0, "True": 1}
).astype(int)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"      Total fitur     : {X.shape[1]} kolom")
print(f"      Distribusi churn: {y.value_counts().to_dict()}  (rasio: {y.mean():.1%})")

# ─────────────────────────────────────────────
# 2. IDENTIFIKASI TIPE KOLOM
# ─────────────────────────────────────────────
# area code diperlakukan sebagai kategori (408/415/510 = kode wilayah)
CAT_COLS = ["state", "international plan", "voice mail plan", "area code"]
NUM_COLS = [c for c in X.columns if c not in CAT_COLS]

print(f"\n[2/6] Identifikasi kolom")
print(f"      Kategorikal     : {CAT_COLS}")
print(f"      Numerik ({len(NUM_COLS)} kol) : {NUM_COLS}")

# Simpan kategori valid untuk validasi input di app.py
kategori_valid = {col: sorted(X[col].astype(str).unique().tolist()) for col in CAT_COLS}

# ─────────────────────────────────────────────
# 3. SPLIT DATA (stratify agar distribusi kelas terjaga)
# ─────────────────────────────────────────────
print("\n[3/6] Split data 80% Train / 20% Test (stratified)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {X_train.shape[0]} baris  |  Test: {X_test.shape[0]} baris")
print(f"      Churn di train: {y_train.mean():.1%}  |  Test: {y_test.mean():.1%}")

# Statistik fitur dari data LATIH (untuk default & validasi di app.py)
stats_fitur = {}
for col in NUM_COLS:
    stats_fitur[col] = {
        "min": float(X_train[col].min()),
        "max": float(X_train[col].max()),
        "median": float(X_train[col].median()),
        "mean": float(X_train[col].mean()),
        "q25": float(X_train[col].quantile(0.25)),
        "q75": float(X_train[col].quantile(0.75)),
    }

# Statistik per kelas untuk insight dinamis (perbandingan dengan rata-rata churn)
stats_per_kelas = {}
df_train_full = X_train.copy()
df_train_full[TARGET_COL] = y_train.values
for col in NUM_COLS:
    stats_per_kelas[col] = {
        "mean_churn": float(
            df_train_full[df_train_full[TARGET_COL] == 1][col].mean()
        ),
        "mean_no_churn": float(
            df_train_full[df_train_full[TARGET_COL] == 0][col].mean()
        ),
    }

# ─────────────────────────────────────────────
# 4. DEFINISI PIPELINE PREPROCESSING + MODEL
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            CAT_COLS,
        ),
    ],
    remainder="drop",
)

feature_names_out = NUM_COLS + CAT_COLS

kandidat_model = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# ─────────────────────────────────────────────
# 5. LATIH & BANDINGKAN KETIGA MODEL
# ─────────────────────────────────────────────
print("\n[4/6] Melatih & membandingkan 3 model...")
print("-" * 62)
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("-" * 62)

metrik_semua_model = {}
pipeline_semua_model = {}
best_auc = -1
best_name = ""
best_pipeline = None

for nama, clf in kandidat_model.items():
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    m = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    metrik_semua_model[nama] = m
    pipeline_semua_model[nama] = pipe

    print(
        f"{nama:<22} {m['Accuracy']:>6.4f} {m['Precision']:>6.4f} "
        f"{m['Recall']:>6.4f} {m['F1-Score']:>6.4f} {m['ROC-AUC']:>6.4f}"
    )

    if m["ROC-AUC"] > best_auc:
        best_auc = m["ROC-AUC"]
        best_name = nama
        best_pipeline = pipe

print("-" * 62)
print(f"\n  Model Terbaik (ROC-AUC): {best_name}  [{best_auc:.4f}]")

# ─────────────────────────────────────────────
# 6. SIMPAN PAKET PKL
# ─────────────────────────────────────────────
print("\n[5/6] Mengemas artefak...")

# Feature importances
importances = None
try:
    importances = best_pipeline.named_steps["model"].feature_importances_.tolist()
except AttributeError:
    try:
        importances = np.abs(best_pipeline.named_steps["model"].coef_[0]).tolist()
    except Exception:
        importances = [0.0] * len(feature_names_out)

# ROC curve data per model
roc_data = {}
for nama, pipe in pipeline_semua_model.items():
    yp = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, yp)
    roc_data[nama] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": metrik_semua_model[nama]["ROC-AUC"],
    }

# Data untuk tab eksplorasi di app.py
df_eksplorasi = df.copy()

paket_model = {
    # Model
    "model": best_pipeline,
    "nama_model": best_name,
    # Fitur
    "fitur": feature_names_out,
    "cat_cols": CAT_COLS,
    "num_cols": NUM_COLS,
    # Metrik model terbaik
    "metrics": {k: v for k, v in metrik_semua_model[best_name].items()
                if k != "Confusion Matrix"},
    "confusion_matrix": metrik_semua_model[best_name]["Confusion Matrix"],
    # Perbandingan semua model
    "metrik_semua_model": metrik_semua_model,
    "roc_data": roc_data,
    # Importances
    "importances": importances,
    # Kategori valid & statistik
    "kategori_valid": kategori_valid,
    "stats_fitur": stats_fitur,
    "stats_per_kelas": stats_per_kelas,
    # Data eksplorasi
    "df_eksplorasi": df_eksplorasi,
    # Metadata
    "versi_sklearn": sklearn.__version__,
    "tanggal_training": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

OUTPUT_PATH = "model_rf_bigml.pkl"
print(f"[6/6] Menyimpan ke {OUTPUT_PATH} ...")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(paket_model, f)

print(f"\n  Selesai! File '{OUTPUT_PATH}' berhasil dibuat.")
print(f"  Versi sklearn : {sklearn.__version__}")
print(f"  Tanggal       : {paket_model['tanggal_training']}")
print("\n  Selanjutnya jalankan: streamlit run app.py")