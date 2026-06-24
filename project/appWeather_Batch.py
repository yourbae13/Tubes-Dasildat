import streamlit as st
import pandas as pd
import joblib
import os
import glob
import numpy as np
from sklearn.base import is_regressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Weather Prediction",
    page_icon="🌦️",
    layout="wide"
)

# ========================
# BIODATA KELOMPOK
# ========================
# Silakan isi data di bawah ini (masih kosong / placeholder)
BIODATA = {
    "judul_tugas": "Tugas Besar Dasar Ilmu Data - Weather Type & UV Index Prediction",       # contoh: "Prediksi Cuaca Menggunakan Machine Learning"
    "mata_kuliah": "Dasar Ilmu Data",      
    "nama_kelompok": "Kelompok 6",     # contoh: "Kelompok 1"
    "anggota": [
        {"nama": "Muhammad Fauzan Firdaus", "nim": "707012400036"},
        {"nama": "Rovalina Andini", "nim": "707012400111"},
        {"nama": "Bayu Firmansyah", "nim": "707012400135"},
    ],
}

# ========================
# KONSTANTA
# ========================
CLASSIFICATION_TARGET = "Weather Type"
REGRESSION_TARGET     = "UV Index"

NUMERIC_FEATURES = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Precipitation (%)",
    "Atmospheric Pressure",
    "Visibility (km)",
]

# 9 fitur asli (tanpa one-hot) — dipakai untuk input CSV/Excel
RAW_FEATURES = NUMERIC_FEATURES + [
    "Cloud Cover",
    "Season",
    "Location",
]

ALL_FEATURES = NUMERIC_FEATURES + [
    "Cc_clear", "Cc_cloudy", "Cc_overcast", "Cc_partly cloudy",
    "Ss_Autumn", "Ss_Spring", "Ss_Summer", "Ss_Winter",
    "Lc_coastal", "Lc_inland", "Lc_mountain",
]

CLASSIFICATION_FEATURES = ALL_FEATURES
REGRESSION_FEATURES     = ALL_FEATURES

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "project", "weather_classification_data_cleaned.csv")

CLOUD_COVER_OPTIONS  = ["clear", "cloudy", "overcast", "partly cloudy"]
SEASON_OPTIONS       = ["Autumn", "Spring", "Summer", "Winter"]
LOCATION_OPTIONS     = ["coastal", "inland", "mountain"]

COLUMN_DESCRIPTIONS = {
    "Temperature"          : "Suhu udara dalam derajat Celsius (°C).",
    "Humidity"             : "Kelembapan relatif udara dalam persen (%).",
    "Wind Speed"           : "Kecepatan angin dalam km/jam.",
    "Precipitation (%)"    : "Persentase kemungkinan curah hujan.",
    "Atmospheric Pressure" : "Tekanan atmosfer dalam hPa.",
    "Visibility (km)"      : "Jarak pandang dalam kilometer.",
    "Cloud Cover"          : "Kondisi tutupan awan (clear / cloudy / overcast / partly cloudy).",
    "Season"               : "Musim saat pengukuran dilakukan.",
    "Location"             : "Tipe lokasi geografis pengukuran.",
    "UV Index"             : "Indeks ultraviolet matahari (0–14). **Target regresi.**",
    "Weather Type"         : "Jenis cuaca: Sunny, Rainy, Cloudy, Snowy. **Target klasifikasi.**",
}

WEATHER_COLORS = {
    "Sunny" : "#F4A261",
    "Cloudy": "#90B4CE",
    "Rainy" : "#457B9D",
    "Snowy" : "#A8DADC",
}

# ========================
# HELPERS
# ========================
def get_model_files(mode: str):
    folder = os.path.join(BASE_DIR, "model", mode)
    if not os.path.exists(folder):
        return []
    return glob.glob(os.path.join(folder, "*.joblib")) + glob.glob(os.path.join(folder, "*.pkl"))

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def read_file(file):
    return pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

def validate_columns(df, features):
    return [c for c in features if c not in df.columns]

def one_hot_from_radio(prefix_map: dict) -> dict:
    """
    prefix_map: { "Cc": "cloudy", "Ss": "Summer", "Lc": "inland" }
    Returns dict of one-hot columns.
    """
    result = {}
    for prefix, selected in prefix_map.items():
        for col in ALL_FEATURES:
            if col.startswith(prefix + "_"):
                suffix = col[len(prefix)+1:]
                result[col] = 1 if suffix == selected else 0
    return result

def encode_raw_to_onehot(df: pd.DataFrame, target_features: list) -> pd.DataFrame:
    """
    Konversi DataFrame dengan 9 fitur asli (Cloud Cover, Season, Location)
    menjadi DataFrame dengan one-hot encoding sesuai target_features model.
    """
    df = df.copy()

    # One-hot Cloud Cover → Cc_*
    for val in CLOUD_COVER_OPTIONS:
        df[f"Cc_{val}"] = (df["Cloud Cover"].str.strip().str.lower() == val).astype(int)

    # One-hot Season → Ss_*
    for val in SEASON_OPTIONS:
        df[f"Ss_{val}"] = (df["Season"].str.strip() == val).astype(int)

    # One-hot Location → Lc_*
    for val in LOCATION_OPTIONS:
        df[f"Lc_{val}"] = (df["Location"].str.strip().str.lower() == val).astype(int)

    # Kembalikan hanya kolom yang dibutuhkan model
    return df[target_features]

# ========================
# EVALUASI
# ========================
def evaluate_classification(model):
    df   = load_data()
    X    = df[CLASSIFICATION_FEATURES]
    y    = df[CLASSIFICATION_TARGET]
    pred = model.predict(X)
    return (
        accuracy_score(y, pred),
        confusion_matrix(y, pred, labels=model.classes_),
        classification_report(y, pred),
    )

def evaluate_regression(model):
    df   = load_data()
    X    = df[REGRESSION_FEATURES]
    y    = df[REGRESSION_TARGET]
    pred = model.predict(X)
    return (
        mean_absolute_error(y, pred),
        mean_squared_error(y, pred),
        np.sqrt(mean_squared_error(y, pred)),
        r2_score(y, pred),
    )

# ========================
# TAMPILAN METRIK
# ========================
def show_regression_metrics(mae, mse, rmse, r2):
    st.subheader("📊 Regression Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",  f"{mae:.4f}")
    c2.metric("MSE",  f"{mse:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")
    c4.metric("R²",   f"{r2:.4f}")

def show_classification_metrics(acc, cm, rep, model):
    st.subheader("📊 Classification Metrics")
    st.metric("Accuracy", f"{acc * 100:.2f}%")
    st.code(rep)
    st.dataframe(
        pd.DataFrame(cm, index=model.classes_, columns=model.classes_),
        use_container_width=True,
    )

# ========================
# HALAMAN BIODATA
# ========================
def page_biodata():
    st.header("👤 Biodata Kelompok")

    judul   = BIODATA["judul_tugas"] or "_(belum diisi)_"
    matkul  = BIODATA["mata_kuliah"] or "_(belum diisi)_"
    nama_kp = BIODATA["nama_kelompok"] or "_(belum diisi)_"

    st.markdown(f"### 📌 Judul Tugas\n{judul}")
    st.markdown(f"**Mata Kuliah:** {matkul}")
    st.markdown(f"**Nama Kelompok:** {nama_kp}")

    st.divider()
    st.subheader("👥 Anggota Kelompok")

    cols = st.columns(min(4, max(1, len(BIODATA["anggota"]))))
    for i, anggota in enumerate(BIODATA["anggota"]):
        nama = anggota.get("nama") or "_(belum diisi)_"
        nim  = anggota.get("nim") or "_(belum diisi)_"
        with cols[i % len(cols)]:
            st.markdown(f"**{i + 1}. {nama}**")
            st.caption(f"NIM: {nim}")


# ========================
# HALAMAN VISUALISASI
# ========================
def page_visualisasi():
    st.header("📊 Visualisasi Dataset")

    df = load_data()

    # ---- Deskripsi Dataset ----
    with st.expander("📖 Tentang Dataset", expanded=True):
        st.markdown("""
Dataset ini berisi **11.102 data cuaca** dari berbagai lokasi dan musim.
Digunakan untuk dua tugas:
- 🏷️ **Klasifikasi** → memprediksi jenis cuaca *(Sunny, Rainy, Cloudy, Snowy)*
- 📈 **Regresi** → memprediksi nilai *UV Index* (0–14)
        """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Kolom Numerik**")
            for col in ["Temperature", "Humidity", "Wind Speed",
                        "Precipitation (%)", "Atmospheric Pressure",
                        "Visibility (km)", "UV Index"]:
                st.markdown(f"- **{col}**: {COLUMN_DESCRIPTIONS[col]}")
        with col_b:
            st.markdown("**Kolom Kategorikal (One-Hot Encoded)**")
            for col in ["Cloud Cover", "Season", "Location",
                        "Weather Type"]:
                st.markdown(f"- **{col}**: {COLUMN_DESCRIPTIONS[col]}")

    st.divider()

    # ---- Distribusi Weather Type ----
    st.subheader("🌤️ Distribusi Jenis Cuaca")
    wt_count = df[CLASSIFICATION_TARGET].value_counts().reset_index()
    wt_count.columns = ["Weather Type", "Count"]
    fig_wt = px.bar(
        wt_count, x="Weather Type", y="Count",
        color="Weather Type",
        color_discrete_map=WEATHER_COLORS,
        text="Count",
        template="plotly_dark",
    )
    fig_wt.update_traces(textposition="outside")
    fig_wt.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_wt, use_container_width=True)

    st.divider()

    # ---- Distribusi Fitur Numerik ----
    st.subheader("📐 Distribusi Fitur Numerik")
    feat_sel = st.selectbox("Pilih fitur:", NUMERIC_FEATURES + ["UV Index"])
    fig_hist = px.histogram(
        df, x=feat_sel, color=CLASSIFICATION_TARGET,
        color_discrete_map=WEATHER_COLORS,
        barmode="overlay", nbins=40,
        template="plotly_dark",
        opacity=0.75,
    )
    fig_hist.update_layout(height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ---- Boxplot per Weather Type ----
    st.subheader("📦 Boxplot Fitur vs Weather Type")
    feat_box = st.selectbox("Pilih fitur untuk boxplot:", NUMERIC_FEATURES + ["UV Index"], key="box")
    fig_box = px.box(
        df, x=CLASSIFICATION_TARGET, y=feat_box,
        color=CLASSIFICATION_TARGET,
        color_discrete_map=WEATHER_COLORS,
        template="plotly_dark",
        points=False,
    )
    fig_box.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ---- Korelasi Heatmap ----
    st.subheader("🔥 Heatmap Korelasi")
    numeric_cols = NUMERIC_FEATURES + ["UV Index"]
    corr = df[numeric_cols].corr().round(2)
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ---- Scatter Plot ----
    st.subheader("🔵 Scatter Plot Antar Fitur")
    col1, col2 = st.columns(2)
    axis_options = NUMERIC_FEATURES + ["UV Index"]
    with col1:
        x_axis = st.selectbox("Sumbu X", axis_options, index=0)
    with col2:
        y_axis = st.selectbox("Sumbu Y", axis_options, index=6)

    fig_sc = px.scatter(
        df.sample(min(2000, len(df)), random_state=42),
        x=x_axis, y=y_axis,
        color=CLASSIFICATION_TARGET,
        color_discrete_map=WEATHER_COLORS,
        opacity=0.6,
        template="plotly_dark",
    )
    fig_sc.update_layout(height=420)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # ---- Distribusi Season & Location ----
    st.subheader("🗓️ Distribusi Musim & Lokasi")
    c1, c2 = st.columns(2)

    season_map = {"Autumn": "Ss_Autumn", "Spring": "Ss_Spring",
                  "Summer": "Ss_Summer", "Winter": "Ss_Winter"}
    season_counts = {s: df[col].sum() for s, col in season_map.items()}
    fig_season = px.pie(
        values=list(season_counts.values()),
        names=list(season_counts.keys()),
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4,
    )
    fig_season.update_layout(height=320, margin=dict(t=10, b=10))
    c1.markdown("**Musim**")
    c1.plotly_chart(fig_season, use_container_width=True)

    loc_map = {"Coastal": "Lc_coastal", "Inland": "Lc_inland", "Mountain": "Lc_mountain"}
    loc_counts = {l: df[col].sum() for l, col in loc_map.items()}
    fig_loc = px.pie(
        values=list(loc_counts.values()),
        names=list(loc_counts.keys()),
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig_loc.update_layout(height=320, margin=dict(t=10, b=10))
    c2.markdown("**Lokasi**")
    c2.plotly_chart(fig_loc, use_container_width=True)

# ========================
# NAVBAR SIDEBAR
# ========================
st.sidebar.title("🌦️ Weather Prediction")

menu_utama = st.sidebar.radio(
    "Navigasi",
    ["👤 Biodata", "📊 Visualisasi Data", "🏷️ Klasifikasi", "📈 Regresi"],
)

# ========================
# HALAMAN BIODATA
# ========================
if menu_utama == "👤 Biodata":
    page_biodata()
    st.stop()

# ========================
# HALAMAN VISUALISASI
# ========================
if menu_utama == "📊 Visualisasi Data":
    page_visualisasi()
    st.stop()

# ========================
# SETUP MODE PREDIKSI
# ========================
is_regression_mode = menu_utama == "📈 Regresi"
mode_key           = "regression" if is_regression_mode else "classification"
active_features    = REGRESSION_FEATURES if is_regression_mode else CLASSIFICATION_FEATURES
active_target      = REGRESSION_TARGET if is_regression_mode else CLASSIFICATION_TARGET

st.sidebar.divider()
input_mode = st.sidebar.radio("Input Data", ["Manual", "CSV/Excel"])

# ========================
# PILIH MODEL
# ========================
st.title("🌦️ Weather Prediction App")

model_files = get_model_files(mode_key)

if not model_files:
    st.error(f"❌ Tidak ada model di `model/{mode_key}/`. Pastikan file `.joblib` atau `.pkl` sudah ada.")
    st.stop()

model_names   = [os.path.basename(m) for m in model_files]
selected_name = st.selectbox(
    f"Pilih Model {'Regresi' if is_regression_mode else 'Klasifikasi'}",
    model_names,
)
model = load_model(model_files[model_names.index(selected_name)])

st.caption(f"🎯 Target: `{active_target}` | 📂 `model/{mode_key}/{selected_name}`")

# ========================
# INFO DATASET
# ========================
with st.expander("📖 Tentang Dataset & Kolom"):
    st.markdown("""
Dataset ini berisi **11.102 data cuaca** dari berbagai lokasi dan musim.
Digunakan untuk dua tugas:
- 🏷️ **Klasifikasi** → memprediksi jenis cuaca *(Sunny, Rainy, Cloudy, Snowy)*
- 📈 **Regresi** → memprediksi nilai *UV Index* (0–14)
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Kolom Numerik**")
        for col in ["Temperature", "Humidity", "Wind Speed",
                    "Precipitation (%)", "Atmospheric Pressure",
                    "Visibility (km)", "UV Index"]:
            st.markdown(f"- **{col}**: {COLUMN_DESCRIPTIONS[col]}")
    with col_b:
        st.markdown("**Kolom Kategorikal**")
        for col in ["Cloud Cover", "Season", "Location", "Weather Type"]:
            st.markdown(f"- **{col}**: {COLUMN_DESCRIPTIONS[col]}")

st.divider()

# ========================
# INPUT MANUAL
# ========================
if input_mode == "Manual":
    st.header("🖊️ Prediksi Manual")

    input_data = {}

    # --- Fitur Numerik ---
    st.subheader("Fitur Numerik")
    num_cols = st.columns(3)
    for i, f in enumerate(NUMERIC_FEATURES):
        with num_cols[i % 3]:
            input_data[f] = st.number_input(f, value=0.0, key=f"num_{f}")

    st.divider()

    # --- Radio Button ---
    st.subheader("Kondisi Cuaca")
    r1, r2, r3 = st.columns(3)

    with r1:
        cc_sel = st.radio("☁️ Cloud Cover", CLOUD_COVER_OPTIONS, index=0)
    with r2:
        ss_sel = st.radio("🗓️ Season", SEASON_OPTIONS, index=0)
    with r3:
        lc_sel = st.radio("📍 Location", LOCATION_OPTIONS, index=0)

    one_hot = one_hot_from_radio({"Cc": cc_sel, "Ss": ss_sel, "Lc": lc_sel})
    input_data.update(one_hot)

    input_df = pd.DataFrame([input_data])[active_features]

    with st.expander("Lihat Data Input"):
        st.dataframe(input_df, use_container_width=True)

    if st.button("🔍 Prediksi", use_container_width=True):
        pred = model.predict(input_df)[0]

        if is_regression_mode:
            st.success(f"✅ Prediksi **{active_target}**: `{pred:.4f}`")
            st.divider()
            mae, mse, rmse, r2 = evaluate_regression(model)
            show_regression_metrics(mae, mse, rmse, r2)
        else:
            st.success(f"✅ Prediksi **{active_target}**: `{pred}`")
            st.divider()
            acc, cm, rep = evaluate_classification(model)
            show_classification_metrics(acc, cm, rep, model)

# ========================
# INPUT CSV / EXCEL
# ========================
else:
    st.header("📂 Upload CSV / Excel")

    # Panduan format kolom yang dibutuhkan

    file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])

    if file:
        df = read_file(file)

        st.subheader("Preview Data")
        st.dataframe(df.head(), use_container_width=True)

        # Validasi 9 fitur asli
        missing = validate_columns(df, RAW_FEATURES)

        if missing:
            st.error("❌ Kolom berikut tidak ditemukan di file:")
            st.write(missing)
        else:
            if st.button("🔍 Prediksi Semua Baris", use_container_width=True):
                try:
                    # Encode kolom kategorikal → one-hot sebelum prediksi
                    input_encoded = encode_raw_to_onehot(df[RAW_FEATURES], active_features)

                    pred = model.predict(input_encoded)
                    df["prediction"] = pred

                    st.subheader("Hasil Prediksi")
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        "⬇️ Download Hasil CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_prediksi.csv",
                        mime="text/csv",
                    )

                    st.divider()

                    if is_regression_mode:
                        mae, mse, rmse, r2 = evaluate_regression(model)
                        show_regression_metrics(mae, mse, rmse, r2)
                    else:
                        acc, cm, rep = evaluate_classification(model)
                        show_classification_metrics(acc, cm, rep, model)

                except Exception as e:
                    st.error(f"❌ Gagal memproses prediksi: {e}")
                    st.info("Pastikan nilai kolom **Cloud Cover**, **Season**, dan **Location** sesuai format yang ditentukan.")