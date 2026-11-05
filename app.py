"""
app.py — Dashboard Prediksi Customer Churn
==========================================
Jalankan setelah train_model.py selesai:
  streamlit run app.py
"""

import io
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS kustom ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Warna kartu KPI */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        color: #f1f5f9;
    }
    [data-testid="metric-container"] label {font-size: 0.78rem; color: #94a3b8;}
    [data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 700;}

    /* Badge risiko */
    .badge-rendah {background:#16a34a;color:#fff;border-radius:8px;padding:4px 12px;font-weight:700;}
    .badge-sedang {background:#d97706;color:#fff;border-radius:8px;padding:4px 12px;font-weight:700;}
    .badge-tinggi {background:#dc2626;color:#fff;border-radius:8px;padding:4px 12px;font-weight:700;}
    .badge-warning {background:#ea580c;color:#fff;border-radius:8px;padding:4px 12px;font-weight:700;}

    /* Kartu insight */
    .insight-card {
        background:#1e293b;border-left:4px solid #3b82f6;
        border-radius:8px;padding:12px 16px;margin:6px 0;
        color:#cbd5e1;font-size:0.9rem;
    }
    .insight-warn {border-left-color:#f59e0b;}
    .insight-danger {border-left-color:#ef4444;}
    .insight-ok {border-left-color:#22c55e;}

    /* Footer */
    .footer {text-align:center;color:#475569;font-size:0.75rem;margin-top:40px;padding-top:12px;border-top:1px solid #1e293b;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = "model_rf_bigml.pkl"


@st.cache_resource(show_spinner="⏳ Memuat model machine learning...")
def load_package(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ File model `{MODEL_PATH}` tidak ditemukan.\n\n"
        "Jalankan terlebih dahulu: `python train_model.py`"
    )
    st.stop()

pkg = load_package(MODEL_PATH)

# Ekstrak semua artefak dari paket
model           = pkg["model"]
nama_model      = pkg.get("nama_model", "Random Forest")
fitur_model     = pkg["fitur"]
cat_cols        = pkg.get("cat_cols", [])
num_cols        = pkg.get("num_cols", [])
metrics         = pkg["metrics"]
cm_data         = pkg.get("confusion_matrix", [[0, 0], [0, 0]])
metrik_semua    = pkg.get("metrik_semua_model", {})
roc_data        = pkg.get("roc_data", {})
importances     = pkg.get("importances", [])
kategori_valid  = pkg.get("kategori_valid", {})
stats_fitur     = pkg.get("stats_fitur", {})
stats_kelas     = pkg.get("stats_per_kelas", {})
df_eksp         = pkg.get("df_eksplorasi", pd.DataFrame())
versi_pkl       = pkg.get("versi_sklearn", "?")
tgl_training    = pkg.get("tanggal_training", "?")

# Peringatan versi sklearn tidak cocok
if versi_pkl != sklearn.__version__:
    st.warning(
        f"⚠️ Model dilatih dengan scikit-learn **{versi_pkl}**, "
        f"tapi saat ini menggunakan **{sklearn.__version__}**. "
        "Pertimbangkan untuk melatih ulang model."
    )

# ─── Fungsi prediksi ───────────────────────────────────────────
def prediksi_df(df_input: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mengembalikan (label_array, proba_array) dari Pipeline."""
    df_proc = df_input.copy()
    # area code: pastikan string agar OrdinalEncoder konsisten
    if "area code" in df_proc.columns:
        df_proc["area code"] = df_proc["area code"].astype(str)
    df_proc = df_proc[fitur_model]
    proba = model.predict_proba(df_proc)[:, 1]
    return proba


def segmen_risiko(p: float) -> tuple[str, str]:
    if p <= 0.30:
        return "Rendah", "rendah"
    elif p <= 0.70:
        return "Sedang", "sedang"
    else:
        return "Tinggi", "tinggi"


def fmt_area_code(v) -> str:
    return str(int(v)) if isinstance(v, float) else str(v)


# ─── Helper median default ─────────────────────────────────────
def med(col):
    return stats_fitur.get(col, {}).get("median", 0.0)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Prediksi Churn")
    st.caption("Dashboard Kecerdasan Bisnis")
    st.divider()

    st.markdown("### ⚙️ Pengaturan")
    threshold = st.slider(
        "Batas Risiko Churn",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        help="Jika probabilitas churn ≥ batas ini → pelanggan diprediksi CHURN.",
    )
    st.caption(
        "📌 **0.50** = standar. Turunkan angka ini jika ingin lebih banyak "
        "pelanggan berisiko terdeteksi (strategi retensi agresif)."
    )

    st.divider()
    st.markdown("### 📋 Info Model")
    st.markdown(f"- **Algoritma**: {nama_model}")
    st.markdown(f"- **Jumlah fitur**: {len(fitur_model)}")
    st.markdown(f"- **Tanggal latih**: {tgl_training}")
    st.markdown(f"- **Versi sklearn**: {versi_pkl}")
    st.markdown(f"- **Akurasi uji**: {metrics.get('Accuracy', 0):.1%}")
    st.markdown(f"- **ROC-AUC**: {metrics.get('ROC-AUC', 0):.4f}")

    st.divider()
    st.markdown(
        '<div class="footer">Dataset: BigML Telco Churn<br>Kecerdasan Bisnis 2024</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# TABS UTAMA
# ═══════════════════════════════════════════════════════════════
tab_home, tab_eksp, tab_satu, tab_batch, tab_performa = st.tabs([
    "🏠 Ringkasan",
    "🔎 Eksplorasi Data",
    "👤 Prediksi Satu Pelanggan",
    "📄 Prediksi Batch",
    "📊 Performa Model",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — RINGKASAN
# ═══════════════════════════════════════════════════════════════
with tab_home:
    st.title("📡 Dashboard Prediksi Customer Churn")
    st.markdown(
        "Sistem ini membantu tim bisnis **mengidentifikasi pelanggan yang berpotensi berhenti berlangganan** "
        "(*churn*) sebelum mereka benar-benar pergi — sehingga tim retensi dapat bertindak tepat waktu."
    )

    st.markdown("---")

    # Cara kerja 3 langkah
    st.subheader("🔄 Cara Kerja Sistem dalam 3 Langkah")
    l1, l2, l3 = st.columns(3)
    with l1:
        st.markdown("#### 1️⃣ Data Historis")
        st.markdown(
            "Sistem belajar dari **3.333 data pelanggan** masa lalu beserta "
            "status churn mereka (apakah akhirnya berhenti atau tetap berlangganan)."
        )
    with l2:
        st.markdown("#### 2️⃣ Model AI")
        st.markdown(
            f"Algoritma **{nama_model}** mengenali pola tersembunyi — misalnya pelanggan "
            "yang sering komplain ke *customer service* cenderung akan churn."
        )
    with l3:
        st.markdown("#### 3️⃣ Prediksi & Tindakan")
        st.markdown(
            "Untuk pelanggan baru, sistem menghitung **probabilitas churn (0–100%)** "
            "dan merekomendasikan tindakan retensi yang sesuai."
        )

    st.markdown("---")

    # KPI Dataset
    st.subheader("📊 Statistik Dataset")
    if not df_eksp.empty:
        total_pel = len(df_eksp)
        jml_churn = int(df_eksp["churn"].sum())
        rasio = jml_churn / total_pel
        akurasi = metrics.get("Accuracy", 0)
        auc = metrics.get("ROC-AUC", 0)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("👥 Total Pelanggan", f"{total_pel:,}")
        k2.metric("⚠️ Pelanggan Churn", f"{jml_churn:,}")
        k3.metric("📉 Rasio Churn", f"{rasio:.1%}")
        k4.metric("🎯 Akurasi Model", f"{akurasi:.1%}")
        k5.metric("📈 ROC-AUC", f"{auc:.4f}")

    st.markdown("---")

    # Masalah bisnis
    st.subheader("💼 Mengapa Ini Penting?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="insight-card insight-danger">
            🔴 <b>Tantangan Bisnis</b><br>
            Biaya mendapatkan pelanggan baru bisa <b>5–7× lebih mahal</b>
            dibanding mempertahankan yang ada. Churn yang tidak terdeteksi = kerugian tersembunyi.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="insight-card insight-ok">
            🟢 <b>Solusi dengan AI</b><br>
            Dengan mengetahui siapa yang berisiko <i>sebelum</i> mereka pergi,
            tim retensi bisa memberikan penawaran khusus, diskon, atau prioritas layanan
            untuk pelanggan yang tepat sasaran.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Rekomendasi Aksi Bisnis & Sasaran Retensi
    st.subheader("🎯 Aksi Bisnis: Pelanggan Target Retensi & Peringatan Dini")
    st.markdown(
        "Hasil eksekusi model mendeteksi pelanggan **Risiko Tinggi** dan sistem **Early Warning (CS Calls ≥ 3)** "
        "yang memerlukan intervensi langsung dari tim retensi."
    )

    file_retensi = "pelanggan_target_retensi.csv"
    if os.path.exists(file_retensi):
        df_target_csv = pd.read_csv(file_retensi)
        t1, t2, t3 = st.columns(3)
        t1.metric("🎯 Total Target Retensi", f"{len(df_target_csv):,} Pelanggan")
        t2.metric("⚡ Panggilan CS ≥ 3 (Early Warning)", f"{(df_target_csv['customer service calls'] >= 3).sum():,} Pelanggan")
        t3.metric("🔥 Rata-rata Prob. Churn", f"{df_target_csv['probabilitas_churn'].mean():.1%}")

        st.dataframe(
            df_target_csv.head(10),
            column_config={
                "probabilitas_churn": st.column_config.ProgressColumn(
                    "Probabilitas Churn",
                    format="%.2f",
                    min_value=0.0, max_value=1.0,
                )
            },
            hide_index=True,
            use_container_width=True,
        )

        with open(file_retensi, "rb") as f_dl:
            st.download_button(
                "⬇️ Unduh Daftar Lengkap Target Retensi (CSV)",
                data=f_dl.read(),
                file_name="pelanggan_target_retensi.csv",
                mime="text/csv",
                type="primary",
            )
    else:
        st.info("ℹ️ File daftar target retensi belum dibuat. Jalankan pipeline analisis untuk mengunduh daftar sasaran.")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — EKSPLORASI DATA
# ═══════════════════════════════════════════════════════════════
with tab_eksp:
    st.subheader("🔎 Eksplorasi Data Pelanggan")

    if df_eksp.empty:
        st.warning("Data eksplorasi tidak tersedia. Latih ulang model untuk memuat data ini.")
    else:
        df_eksp["Status Churn"] = df_eksp["churn"].map({1: "Churn", 0: "Tidak Churn"})

        col1, col2 = st.columns(2)

        # Grafik 1: Distribusi Churn
        with col1:
            st.markdown("#### Distribusi Status Pelanggan")
            vc = df_eksp["Status Churn"].value_counts().reset_index()
            vc.columns = ["Status", "Jumlah"]
            fig = px.pie(
                vc, values="Jumlah", names="Status",
                color="Status",
                color_discrete_map={"Churn": "#ef4444", "Tidak Churn": "#22c55e"},
                hole=0.45,
            )
            fig.update_traces(textinfo="percent+label", textfont_size=13)
            fig.update_layout(
                showlegend=True, margin=dict(t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="insight-card insight-danger">📌 Hanya <b>14,5%</b> pelanggan yang churn — '
                'data tidak seimbang (<i>imbalanced</i>), sehingga metrik <b>Recall</b> dan '
                '<b>ROC-AUC</b> lebih penting daripada Akurasi semata.</div>',
                unsafe_allow_html=True,
            )

        # Grafik 2: Churn berdasarkan Paket Internasional
        with col2:
            st.markdown("#### Churn vs Paket Internasional")
            grp = (
                df_eksp.groupby(["international plan", "Status Churn"])
                .size()
                .reset_index(name="Jumlah")
            )
            fig2 = px.bar(
                grp, x="international plan", y="Jumlah",
                color="Status Churn", barmode="group",
                color_discrete_map={"Churn": "#ef4444", "Tidak Churn": "#22c55e"},
                labels={"international plan": "Paket Internasional"},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(
                '<div class="insight-card insight-warn">📌 Pelanggan dengan <b>paket internasional</b> '
                'memiliki rasio churn yang jauh lebih tinggi. Ini bisa jadi sinyal ketidakpuasan '
                'terhadap tarif roaming.</div>',
                unsafe_allow_html=True,
            )

        col3, col4 = st.columns(2)

        # Grafik 3: Churn berdasarkan Customer Service Calls
        with col3:
            st.markdown("#### Churn vs Panggilan ke Customer Service")
            grp3 = (
                df_eksp.groupby(["customer service calls", "Status Churn"])
                .size()
                .reset_index(name="Jumlah")
            )
            fig3 = px.bar(
                grp3, x="customer service calls", y="Jumlah",
                color="Status Churn", barmode="stack",
                color_discrete_map={"Churn": "#ef4444", "Tidak Churn": "#22c55e"},
                labels={"customer service calls": "Jumlah Panggilan ke CS"},
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(
                '<div class="insight-card insight-danger">📌 Pelanggan yang menghubungi CS '
                '<b>4 kali atau lebih</b> memiliki kecenderungan churn yang sangat tinggi. '
                'Ini adalah indikator ketidakpuasan layanan yang kuat.</div>',
                unsafe_allow_html=True,
            )

        # Grafik 4: Distribusi menit siang per status
        with col4:
            st.markdown("#### Distribusi Menit Telepon Siang (Day Minutes)")
            fig4 = px.box(
                df_eksp, x="Status Churn", y="total day minutes",
                color="Status Churn",
                color_discrete_map={"Churn": "#ef4444", "Tidak Churn": "#22c55e"},
                labels={"total day minutes": "Total Menit Siang", "Status Churn": ""},
            )
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", showlegend=False, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(
                '<div class="insight-card insight-warn">📌 Pelanggan yang <b>churn</b> cenderung '
                'memiliki menit panggilan siang yang lebih tinggi — artinya mereka membayar '
                'lebih banyak, dan mungkin merasa tarif tidak sepadan.</div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════
# TAB 3 — PREDIKSI SATU PELANGGAN
# ═══════════════════════════════════════════════════════════════
with tab_satu:
    st.subheader("👤 Prediksi Risiko Churn — Satu Pelanggan")

    # Tombol preset contoh
    st.markdown("**Coba dengan data contoh:**")
    c_btn1, c_btn2, _ = st.columns([1, 1, 4])
    preset_aman = c_btn1.button("✅ Isi: Pelanggan Aman", use_container_width=True)
    preset_risk = c_btn2.button("⚠️ Isi: Pelanggan Berisiko", use_container_width=True)

    # Nilai default dari preset atau median
    if preset_aman:
        st.session_state["preset"] = "aman"
    elif preset_risk:
        st.session_state["preset"] = "risiko"

    def get_val(key_aman, key_risiko, default):
        p = st.session_state.get("preset", None)
        if p == "aman":
            return key_aman
        elif p == "risiko":
            return key_risiko
        return default

    st.markdown("---")
    kol_input, kol_hasil = st.columns([1.6, 1], gap="large")

    with kol_input:
        # ── Profil Dasar ──────────────────────────────────────
        with st.expander("📋 Profil Dasar Pelanggan", expanded=True):
            r1, r2, r3 = st.columns(3)
            state_opts = kategori_valid.get("state", ["CA"])
            default_state = get_val("CA", "TX", state_opts[0])
            state_idx = state_opts.index(default_state) if default_state in state_opts else 0
            sel_state = r1.selectbox("Negara Bagian (State)", options=state_opts, index=state_idx)

            acct_default = int(get_val(120, 60, med("account length")))
            acct_len = r2.number_input(
                "Lama Menjadi Pelanggan (Hari)",
                min_value=0, max_value=999, value=acct_default,
                help=f"Rentang data latih: {stats_fitur.get('account length', {}).get('min', 1):.0f} – "
                     f"{stats_fitur.get('account length', {}).get('max', 243):.0f} hari",
            )
            ac_min = stats_fitur.get("account length", {}).get("min", 1)
            ac_max = stats_fitur.get("account length", {}).get("max", 243)
            if acct_len < ac_min or acct_len > ac_max:
                r2.caption(f"⚠️ Di luar rentang data latih ({ac_min:.0f}–{ac_max:.0f})")

            area_opts = kategori_valid.get("area code", ["408", "415", "510"])
            ac_def = get_val("415", "408", "415")
            ac_idx = area_opts.index(ac_def) if ac_def in area_opts else 0
            sel_area = r3.selectbox("Kode Area (Area Code)", options=area_opts, index=ac_idx)

        # ── Paket Layanan ─────────────────────────────────────
        with st.expander("📦 Paket Layanan", expanded=True):
            r1, r2, r3 = st.columns(3)
            intl_opts = kategori_valid.get("international plan", ["no", "yes"])
            intl_def = get_val("no", "yes", "no")
            intl_idx = intl_opts.index(intl_def) if intl_def in intl_opts else 0
            sel_intl = r1.selectbox("Paket Internasional", options=intl_opts, index=intl_idx)

            vm_opts = kategori_valid.get("voice mail plan", ["no", "yes"])
            vm_def = get_val("yes", "no", "no")
            vm_idx = vm_opts.index(vm_def) if vm_def in vm_opts else 0
            sel_vm = r2.selectbox("Paket Voice Mail", options=vm_opts, index=vm_idx)

            vm_msg_def = int(get_val(25, 0, med("number vmail messages")))
            num_vm = r3.number_input(
                "Jumlah Pesan Voice Mail",
                min_value=0, max_value=100, value=vm_msg_def,
            )

        # ── Penggunaan Telepon Siang ──────────────────────────
        with st.expander("☀️ Penggunaan Telepon Siang (Day)", expanded=False):
            c1, c2, c3 = st.columns(3)
            day_min = c1.number_input(
                "Menit Siang", value=float(get_val(180.0, 280.0, med("total day minutes"))),
                help=f"Median pelanggan churn: {stats_kelas.get('total day minutes', {}).get('mean_churn', 0):.1f} menit",
            )
            day_calls = c2.number_input(
                "Jumlah Panggilan Siang", value=int(get_val(100, 95, med("total day calls"))),
            )
            # Auto-hitung charge dari menit (tarif ~$0.17/menit untuk siang)
            TARIF_DAY = 0.17
            day_charge_auto = round(day_min * TARIF_DAY, 2)
            day_charge = c3.number_input(
                "Biaya Siang ($)", value=float(get_val(30.6, 47.6, day_charge_auto)),
                help=f"Dihitung otomatis: {day_min:.1f} menit × ${TARIF_DAY}/menit = ${day_charge_auto:.2f}",
            )

        # ── Penggunaan Telepon Sore ───────────────────────────
        with st.expander("🌆 Penggunaan Telepon Sore (Evening)", expanded=False):
            c1, c2, c3 = st.columns(3)
            TARIF_EVE = 0.085
            eve_min = c1.number_input(
                "Menit Sore", value=float(get_val(200.0, 190.0, med("total eve minutes"))),
            )
            eve_calls = c2.number_input(
                "Jumlah Panggilan Sore", value=int(get_val(100, 95, med("total eve calls"))),
            )
            eve_charge = c3.number_input(
                "Biaya Sore ($)", value=float(get_val(17.0, 16.2, round(eve_min * TARIF_EVE, 2))),
                help=f"Dihitung otomatis: {eve_min:.1f} × ${TARIF_EVE} = ${eve_min*TARIF_EVE:.2f}",
            )

        # ── Penggunaan Telepon Malam ──────────────────────────
        with st.expander("🌙 Penggunaan Telepon Malam (Night)", expanded=False):
            c1, c2, c3 = st.columns(3)
            TARIF_NIGHT = 0.045
            night_min = c1.number_input(
                "Menit Malam", value=float(get_val(200.0, 200.0, med("total night minutes"))),
            )
            night_calls = c2.number_input(
                "Jumlah Panggilan Malam", value=int(get_val(100, 100, med("total night calls"))),
            )
            night_charge = c3.number_input(
                "Biaya Malam ($)", value=float(get_val(9.0, 9.0, round(night_min * TARIF_NIGHT, 2))),
                help=f"Dihitung otomatis: {night_min:.1f} × ${TARIF_NIGHT} = ${night_min*TARIF_NIGHT:.2f}",
            )

        # ── Internasional ─────────────────────────────────────
        with st.expander("🌍 Panggilan Internasional", expanded=False):
            c1, c2, c3 = st.columns(3)
            TARIF_INTL = 0.27
            intl_min = c1.number_input(
                "Menit Internasional", value=float(get_val(10.0, 12.0, med("total intl minutes"))),
            )
            intl_calls = c2.number_input(
                "Jumlah Panggilan Intl", value=int(get_val(4, 6, med("total intl calls"))),
            )
            intl_charge = c3.number_input(
                "Biaya Internasional ($)", value=float(get_val(2.7, 3.24, round(intl_min * TARIF_INTL, 2))),
                help=f"Dihitung otomatis: {intl_min:.1f} × ${TARIF_INTL} = ${intl_min*TARIF_INTL:.2f}",
            )

        # ── Layanan Pelanggan ─────────────────────────────────
        with st.expander("📞 Layanan Pelanggan (Customer Service)", expanded=True):
            cs_max = stats_fitur.get("customer service calls", {}).get("max", 9)
            cs_val = int(get_val(1, 5, int(med("customer service calls"))))
            cs_calls = st.number_input(
                "Jumlah Panggilan ke Customer Service",
                min_value=0, max_value=int(cs_max) + 5, value=cs_val,
                help="Pelanggan yang menghubungi CS ≥ 4× cenderung sangat berisiko churn.",
            )
            if cs_calls > cs_max:
                st.caption(f"⚠️ Nilai {cs_calls} melebihi maksimum data latih ({cs_max:.0f})")

    # ─── Hitung Prediksi ─────────────────────────────────────
    input_data = {
        "state": sel_state,
        "account length": acct_len,
        "area code": str(int(sel_area)) if str(sel_area).replace(".","").isdigit() else sel_area,
        "international plan": sel_intl,
        "voice mail plan": sel_vm,
        "number vmail messages": num_vm,
        "total day minutes": day_min,
        "total day calls": day_calls,
        "total day charge": day_charge,
        "total eve minutes": eve_min,
        "total eve calls": eve_calls,
        "total eve charge": eve_charge,
        "total night minutes": night_min,
        "total night calls": night_calls,
        "total night charge": night_charge,
        "total intl minutes": intl_min,
        "total intl calls": intl_calls,
        "total intl charge": intl_charge,
        "customer service calls": cs_calls,
    }
    df_satu = pd.DataFrame([input_data])
    proba_satu = prediksi_df(df_satu)[0]
    label_satu = "CHURN" if proba_satu >= threshold else "AMAN"
    seg_label, seg_key = segmen_risiko(proba_satu)

    # ─── Panel Hasil ─────────────────────────────────────────
    with kol_hasil:
        st.markdown("### 🎯 Hasil Prediksi")

        # Gauge probabilitas
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba_satu * 100,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": "#ef4444" if proba_satu > 0.5 else "#22c55e"},
                    "steps": [
                        {"range": [0, 30], "color": "#14532d"},
                        {"range": [30, 70], "color": "#78350f"},
                        {"range": [70, 100], "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
                title={"text": "Probabilitas Churn", "font": {"color": "#94a3b8"}},
            )
        )
        gauge.update_layout(
            height=250, margin=dict(t=30, b=0, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        )
        st.plotly_chart(gauge, use_container_width=True)

        # Badge segmen
        badge_color = {"rendah": "insight-ok", "sedang": "insight-warn", "tinggi": "insight-danger"}
        status_icon = {"rendah": "🟢", "sedang": "🟡", "tinggi": "🔴"}
        st.markdown(
            f'<div class="insight-card {badge_color[seg_key]}" style="text-align:center;">'
            f'<span style="font-size:1.5rem">{status_icon[seg_key]}</span><br>'
            f'<b style="font-size:1.1rem">Risiko {seg_label}</b> — {label_satu}<br>'
            f'<small>Probabilitas: {proba_satu:.1%} | Batas: {threshold:.0%}</small></div>',
            unsafe_allow_html=True,
        )

        # Rekomendasi tindakan
        st.markdown("#### 💡 Rekomendasi Tindakan Bisnis")
        
        # Early Warning Alert jika CS >= 3
        if cs_calls >= 3:
            st.markdown(
                f'<div class="insight-card insight-warn" style="border-left:5px solid #ea580c;">'
                f'⚡ <b>SISTEM PERINGATAN DINI (EARLY WARNING TRIGGER)</b><br>'
                f'Pelanggan telah menghubungi Customer Service <b>{cs_calls} kali</b>. '
                f'Ini merupakan sinyal friksi layanan kritis. Segera delegasikan ke supervisor retensi!'
                f'</div>',
                unsafe_allow_html=True,
            )

        if seg_key == "rendah":
            st.success(
                "✅ **Program Loyalitas / Upselling:** Pelanggan ini tampak puas. Berikan apresiasi loyalitas atau tawarkan paket tambahan premium."
            )
        elif seg_key == "sedang":
            st.warning(
                "⚠️ **Survei Kepuasan & Diskon Tagihan:** Kirimkan survei kepuasan pelanggan secara proaktif atau berikan potongan tagihan berkala."
            )
        else:
            # Personalisasi aksi untuk risiko tinggi
            rekom_khusus = []
            if cs_calls >= 3:
                rekom_khusus.append("Audit riwayat kendala & lakukan panggilan retensi prioritas tinggi dari manajer layanan")
            if day_min > 250:
                rekom_khusus.append(f"Tawarkan migrasi ke paket Flat-Rate Siang atau diskon tagihan siang (saat ini ${day_charge:.2f})")
            if sel_intl == "yes":
                rekom_khusus.append("Evaluasi tarif panggilan internasional dan tawarkan Add-on Diskon Internasional")
            if not rekom_khusus:
                rekom_khusus.append("Berikan insentif voucher retensi eksklusif atau diskon 1 bulan biaya langganan")

            st.error("🚨 **Intervensi Prioritas Tim Retensi (Urgent):**")
            for rk in rekom_khusus:
                st.markdown(f"- 📌 {rk}")

        # ─── Insight Dinamis ──────────────────────────────────
        st.markdown("#### 🔍 Mengapa Prediksi Ini?")
        faktor = []
        if cs_calls >= 4:
            faktor.append(("danger",
                f"📞 CS Calls = {cs_calls} (rata-rata pelanggan churn: "
                f"{stats_kelas.get('customer service calls', {}).get('mean_churn', 0):.1f}× — "
                "sinyal ketidakpuasan layanan kuat)"))
        if sel_intl == "yes":
            faktor.append(("warn",
                "🌍 Memiliki paket internasional — kelompok ini memiliki rasio churn lebih tinggi"))
        mc = stats_kelas.get("total day minutes", {}).get("mean_churn", 0)
        if day_min > mc:
            faktor.append(("warn",
                f"☀️ Menit siang = {day_min:.0f} (rata-rata pelanggan churn: {mc:.0f}) "
                "— penggunaan & biaya tinggi meningkatkan risiko"))
        if sel_vm == "no":
            faktor.append(("ok" if proba_satu < 0.5 else "warn",
                "📩 Tidak memiliki paket voice mail — pelanggan tanpa VM cenderung lebih rentan"))
        if cs_calls < 2 and sel_intl == "no":
            faktor.append(("ok",
                "✅ Komplain CS rendah & tidak berpaket internasional — faktor risiko utama tidak ada"))

        if not faktor:
            faktor.append(("ok", "📊 Tidak ada faktor risiko dominan yang terdeteksi pada profil ini"))

        for tipe, teks in faktor[:5]:
            cls = {"danger": "insight-danger", "warn": "insight-warn", "ok": "insight-ok"}.get(tipe, "insight-card")
            st.markdown(f'<div class="insight-card {cls}">{teks}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — PREDIKSI BATCH
# ═══════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📄 Prediksi Batch — Banyak Pelanggan Sekaligus")

    # Stepper visual
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        '<div class="insight-card insight-ok"><b>Langkah 1️⃣</b><br>Unduh template CSV kosong</div>',
        unsafe_allow_html=True,
    )
    s2.markdown(
        '<div class="insight-card insight-warn"><b>Langkah 2️⃣</b><br>Isi data & upload file CSV/Excel</div>',
        unsafe_allow_html=True,
    )
    s3.markdown(
        '<div class="insight-card insight-danger"><b>Langkah 3️⃣</b><br>Jalankan prediksi & unduh hasil</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Tombol unduh template
    template_df = pd.DataFrame(columns=fitur_model)
    buf_tpl = io.BytesIO()
    template_df.to_csv(buf_tpl, index=False)
    st.download_button(
        "⬇️ Unduh Template CSV (19 Kolom)",
        data=buf_tpl.getvalue(),
        file_name="template_churn_input.csv",
        mime="text/csv",
        help="Isi file ini dengan data pelanggan Anda, lalu upload di bawah.",
    )

    st.markdown("---")

    # Upload file
    file_upload = st.file_uploader(
        "Upload File Data Pelanggan (CSV / XLSX / XLS)",
        type=["csv", "xlsx", "xls"],
        help="File harus memiliki kolom sesuai template. Kolom 'churn' jika ada akan diabaikan.",
    )

    if file_upload is not None:
        try:
            if file_upload.name.endswith(".csv"):
                df_up = pd.read_csv(file_upload)
            else:
                df_up = pd.read_excel(file_upload)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            st.stop()

        st.markdown(f"**Preview data yang diupload** ({len(df_up):,} baris):")
        st.dataframe(df_up.head(5))

        # ─── Validasi kolom ───────────────────────────────────
        missing_cols = [c for c in fitur_model if c not in df_up.columns]
        if missing_cols:
            st.error(
                f"❌ **Kolom berikut tidak ditemukan** di file Anda:\n\n"
                f"`{'`, `'.join(missing_cols)}`\n\n"
                "Silakan gunakan template yang disediakan di atas."
            )
        else:
            # ─── Validasi nilai ───────────────────────────────
            masalah = []
            for col in fitur_model:
                n_null = df_up[col].isnull().sum()
                if n_null > 0:
                    masalah.append(f"- `{col}`: {n_null} nilai kosong")
            for col in cat_cols:
                if col in df_up.columns:
                    valid_vals = set(kategori_valid.get(col, []))
                    if valid_vals:
                        invalid = df_up[col].dropna().astype(str)
                        n_invalid = (~invalid.isin(valid_vals)).sum()
                        if n_invalid > 0:
                            masalah.append(
                                f"- `{col}`: {n_invalid} baris memiliki nilai tidak dikenal "
                                f"(valid: {', '.join(list(valid_vals)[:5])}...)"
                            )

            if masalah:
                st.warning(
                    "⚠️ **Ditemukan masalah data berikut (akan memengaruhi akurasi prediksi):**\n\n"
                    + "\n".join(masalah)
                )

            if st.button("🚀 Jalankan Prediksi Batch", type="primary", use_container_width=True):
                with st.spinner("Memproses prediksi untuk semua pelanggan..."):
                    try:
                        proba_batch = prediksi_df(df_up)
                        label_batch = (proba_batch >= threshold).astype(int)

                        hasil_df = df_up.copy()
                        hasil_df["Probabilitas_Churn"] = proba_batch
                        hasil_df["Prediksi_Churn"] = np.where(label_batch == 1, "Ya (Churn)", "Tidak")
                        hasil_df["Segmen_Risiko"] = [
                            segmen_risiko(p)[0] for p in proba_batch
                        ]
                        hasil_df = hasil_df.sort_values("Probabilitas_Churn", ascending=False)

                        # KPI ringkasan
                        jml_churn_b = label_batch.sum()
                        rasio_b = jml_churn_b / len(label_batch)
                        seg_counts = hasil_df["Segmen_Risiko"].value_counts()

                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("👥 Total Pelanggan", f"{len(hasil_df):,}")
                        m2.metric("🔴 Diprediksi Churn", f"{jml_churn_b:,}")
                        m3.metric("📉 Rasio Churn", f"{rasio_b:.1%}")
                        m4.metric("🟡 Risiko Sedang", f"{seg_counts.get('Sedang', 0):,}")
                        m5.metric("🟢 Risiko Rendah", f"{seg_counts.get('Rendah', 0):,}")

                        # Grafik donut segmen
                        seg_df = seg_counts.reset_index()
                        seg_df.columns = ["Segmen", "Jumlah"]
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            fig_donut = px.pie(
                                seg_df, values="Jumlah", names="Segmen", hole=0.5,
                                color="Segmen",
                                color_discrete_map={
                                    "Rendah": "#22c55e",
                                    "Sedang": "#f59e0b",
                                    "Tinggi": "#ef4444",
                                },
                                title="Distribusi Segmen Risiko",
                            )
                            fig_donut.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                                margin=dict(t=40, b=10),
                            )
                            st.plotly_chart(fig_donut, use_container_width=True)

                        with col_b:
                            # Tabel hasil dengan filter
                            st.markdown("**Filter berdasarkan Segmen Risiko:**")
                            filter_seg = st.multiselect(
                                "Tampilkan segmen:",
                                options=["Tinggi", "Sedang", "Rendah"],
                                default=["Tinggi", "Sedang"],
                            )
                            df_filtered = hasil_df[hasil_df["Segmen_Risiko"].isin(filter_seg)]
                            st.dataframe(
                                df_filtered,
                                column_config={
                                    "Probabilitas_Churn": st.column_config.ProgressColumn(
                                        "Probabilitas Churn",
                                        format="%.2f",
                                        min_value=0.0, max_value=1.0,
                                    )
                                },
                                hide_index=True,
                            )

                        # Unduh hasil
                        buf_out = io.BytesIO()
                        hasil_df.to_csv(buf_out, index=False)
                        st.download_button(
                            "⬇️ Unduh Hasil Prediksi (CSV)",
                            data=buf_out.getvalue(),
                            file_name="hasil_prediksi_churn.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"❌ Terjadi error saat prediksi: {e}")

# ═══════════════════════════════════════════════════════════════
# TAB 5 — PERFORMA MODEL
# ═══════════════════════════════════════════════════════════════
with tab_performa:
    st.subheader("📊 Evaluasi & Transparansi Model")
    st.markdown(
        "Semua metrik dihitung dari **data uji (20% data yang tidak digunakan saat training)** "
        "— menggambarkan performa nyata model pada pelanggan baru."
    )

    # ─── 5 Metrik Utama ──────────────────────────────────────
    st.markdown("#### 📐 Metrik Performa Model Terbaik")
    tooltip = {
        "Accuracy": "Dari 100 prediksi total, berapa yang benar?",
        "Precision": "Dari 100 yang diprediksi churn, berapa yang benar-benar churn?",
        "Recall": "Dari 100 pelanggan yang benar-benar churn, berapa yang berhasil terdeteksi?",
        "F1-Score": "Keseimbangan antara Precision dan Recall (makin tinggi = makin baik)",
        "ROC-AUC": "Kemampuan model memisahkan pelanggan churn dari yang tidak (1.0 = sempurna)",
    }
    m1, m2, m3, m4, m5 = st.columns(5)
    for col_ui, (met, tip) in zip(
        [m1, m2, m3, m4, m5], tooltip.items()
    ):
        col_ui.metric(
            met, f"{metrics.get(met, 0):.4f}",
            help=tip,
        )

    # ─── Confusion Matrix ─────────────────────────────────────
    st.markdown("---")
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("#### 🔲 Confusion Matrix")
        st.caption(
            "Menunjukkan berapa pelanggan yang diprediksi benar dan salah "
            "oleh model pada data uji."
        )
        if cm_data and len(cm_data) == 2:
            tn, fp, fn, tp = (
                cm_data[0][0], cm_data[0][1],
                cm_data[1][0], cm_data[1][1],
            )
            cm_fig = px.imshow(
                [[tn, fp], [fn, tp]],
                labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                x=["Tidak Churn", "Churn"],
                y=["Tidak Churn", "Churn"],
                color_continuous_scale=[[0, "#1e293b"], [0.5, "#2563eb"], [1, "#ef4444"]],
                text_auto=True,
            )
            cm_fig.update_traces(textfont_size=18)
            cm_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                margin=dict(t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(cm_fig, use_container_width=True)

            # Penjelasan awam
            st.markdown(
                f'<div class="insight-card insight-ok">✅ <b>Benar Aman</b> (True Negative): {tn} pelanggan — diprediksi tidak churn, dan memang tidak churn</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="insight-card insight-ok">✅ <b>Benar Churn</b> (True Positive): {tp} pelanggan — diprediksi churn, dan memang churn</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="insight-card insight-warn">⚠️ <b>Salah Alarm</b> (False Positive): {fp} pelanggan — diprediksi churn, padahal tidak</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="insight-card insight-danger">🔴 <b>Terlewat</b> (False Negative): {fn} pelanggan — churn tapi tidak terdeteksi</div>',
                unsafe_allow_html=True,
            )

    # ─── ROC Curve ───────────────────────────────────────────
    with col_roc:
        st.markdown("#### 📈 Kurva ROC — Semua Model")
        st.caption(
            "Semakin jauh kurva dari garis diagonal (garis putus-putus), "
            "semakin baik model. Area di bawah kurva (AUC) makin mendekati 1.0 = makin sempurna."
        )
        roc_colors = {"Random Forest": "#3b82f6", "Decision Tree": "#f59e0b", "Logistic Regression": "#a855f7"}
        fig_roc = go.Figure()
        for nama_m, rd in roc_data.items():
            fig_roc.add_trace(
                go.Scatter(
                    x=rd["fpr"], y=rd["tpr"],
                    mode="lines",
                    name=f"{nama_m} (AUC={rd['auc']:.4f})",
                    line=dict(color=roc_colors.get(nama_m, "#60a5fa"), width=2.5),
                )
            )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="#475569", width=1),
                name="Acak (AUC=0.5)",
                showlegend=True,
            )
        )
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ─── Perbandingan Model ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏆 Perbandingan Ketiga Model")
    if metrik_semua:
        rows = []
        for nm, mv in metrik_semua.items():
            rows.append({
                "Model": nm,
                "Accuracy": f"{mv.get('Accuracy', 0):.4f}",
                "Precision": f"{mv.get('Precision', 0):.4f}",
                "Recall": f"{mv.get('Recall', 0):.4f}",
                "F1-Score": f"{mv.get('F1-Score', 0):.4f}",
                "ROC-AUC": f"{mv.get('ROC-AUC', 0):.4f}",
                "Dipilih": "⭐ Ya" if nm == nama_model else "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ─── Feature Importance ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🌟 10 Faktor Paling Berpengaruh terhadap Churn")
    st.caption(
        "Grafik ini menunjukkan variabel yang paling sering digunakan model untuk memutuskan "
        "apakah pelanggan akan churn. Semakin panjang batang = semakin berpengaruh."
    )
    if importances and len(importances) == len(fitur_model):
        df_imp = pd.DataFrame({
            "Fitur": fitur_model,
            "Kepentingan": importances,
        }).sort_values("Kepentingan", ascending=True).tail(10)

        fig_imp = px.bar(
            df_imp, x="Kepentingan", y="Fitur", orientation="h",
            color="Kepentingan",
            color_continuous_scale=["#1d4ed8", "#ef4444"],
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown(
            '<div class="insight-card insight-warn">📌 Catatan: <b>total day minutes</b> dan '
            '<b>total day charge</b> berkorelasi tinggi (biaya = menit × tarif), sehingga '
            'kepentingannya terbagi di antara keduanya.</div>',
            unsafe_allow_html=True,
        )

    # ─── Simulasi Threshold ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚖️ Simulasi Dampak Batas Deteksi (Threshold) terhadap Recall & Precision")
    st.caption(
        "Mengubah threshold mengubah *trade-off*: threshold rendah → lebih banyak yang terdeteksi "
        "(Recall naik), tapi lebih banyak 'salah alarm' (Precision turun). Sebaliknya untuk threshold tinggi."
    )
    if roc_data and nama_model in roc_data:
        from sklearn.metrics import precision_recall_curve
        # Hitung precision-recall untuk beberapa threshold
        thr_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        rd = roc_data[nama_model]
        # Gunakan data ROC yang sudah ada untuk estimasi
        thr_rows = []
        for t in thr_list:
            # Cari index threshold terdekat di ROC
            fpr_arr = np.array(rd["fpr"])
            tpr_arr = np.array(rd["tpr"])
            recall_est = float(np.interp(t, np.linspace(0, 1, len(tpr_arr)), tpr_arr))
            prec_est = (1 - float(np.interp(t, np.linspace(0, 1, len(fpr_arr)), fpr_arr))) if recall_est > 0 else 0
            thr_rows.append({
                "Threshold": t,
                "Recall (Pelanggan Churn Terdeteksi)": f"{recall_est:.1%}",
                "Keterangan": (
                    "🔴 Agresif: banyak terdeteksi tapi banyak salah alarm" if t <= 0.3
                    else "🟡 Seimbang" if t <= 0.5
                    else "🟢 Konservatif: sedikit salah alarm tapi banyak yang terlewat"
                ),
            })
        st.dataframe(pd.DataFrame(thr_rows), hide_index=True, use_container_width=True)

    # ─── Keterbatasan Model ───────────────────────────────────
    st.markdown("---")
    with st.expander("⚠️ Keterbatasan Model — Baca Sebelum Menggunakan"):
        st.markdown(
            f"""
            - **Recall ~{metrics.get('Recall', 0):.0%}**: Sekitar **{(1-metrics.get('Recall', 0)):.0%}** pelanggan yang benar-benar akan churn **tidak terdeteksi** oleh model (*False Negative*). Selalu gabungkan dengan penilaian tim lapangan.
            - **Data tidak seimbang**: Hanya ~14,5% pelanggan churn. Model dilatih dengan `class_weight="balanced"` untuk mengatasinya, namun ketidakseimbangan tetap memengaruhi reliabilitas prediksi.
            - **Data historis**: Model belajar dari data lama. Jika pola perilaku pelanggan berubah (misalnya kebijakan tarif baru), model perlu dilatih ulang.
            - **Korelasi fitur**: Menit dan biaya panggilan sangat berkorelasi (biaya = menit × tarif tetap), sehingga fitur ini memiliki bobot ganda yang redundan.
            - **Tidak ada data demografis**: Dataset tidak mencakup usia, jenis kelamin, atau lama kontrak — faktor yang bisa meningkatkan akurasi jika tersedia.
            """
        )

