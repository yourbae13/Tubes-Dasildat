import streamlit as st
import pandas as pd
import joblib
import os
import glob

st.set_page_config(
    page_title="Weather Classification - Batch Prediction",
    page_icon="🌦️",
    layout="wide"
)

# ==========================
# SIDEBAR MENU
# ==========================
st.sidebar.title("📋 Menu Navigasi")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "🏠 Prediksi Manual",
        "📊 Prediksi Batch",
        "📈 Evaluasi Model"
    ],
    format_func=lambda x: x
)

# Navigasi ke halaman lain
if menu == "🏠 Prediksi Manual":
    st.switch_page("appManual.py")
elif menu == "📈 Evaluasi Model":
    st.switch_page("appEvaluation.py")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Informasi:**
    - Upload file CSV/Excel untuk prediksi batch
    - Hasil akan ditampilkan dan bisa di-download
    """
)

# ==========================
# FITUR DATASET
# ==========================
FEATURE_NAMES = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Precipitation (%)",
    "Atmospheric Pressure",
    "Visibility (km)",
    "Cc_clear",
    "Cc_cloudy",
    "Cc_overcast",
    "Cc_partly cloudy",
    "Ss_Autumn",
    "Ss_Spring",
    "Ss_Summer",
    "Ss_Winter",
    "Lc_coastal",
    "Lc_inland",
    "Lc_mountain"
]

def get_model_files():
    model_folder = "model"
    joblib_files = glob.glob(os.path.join(model_folder, "*.joblib"))
    pkl_files = glob.glob(os.path.join(model_folder, "*.pkl"))
    model_files = joblib_files + pkl_files
    return model_files


@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model


def predict_label(prediction):
    return str(prediction)


def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")


def validate_input_columns(df):
    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
    return missing_cols


def prepare_input_data(df):
    input_df = df[FEATURE_NAMES].copy()

    for col in FEATURE_NAMES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    return input_df


st.title("🌦️ Weather Classification - Prediksi Batch")
st.write(
    "Upload file CSV/Excel yang berisi data cuaca untuk diprediksi secara batch."
)

# =========================
# PILIH MODEL
# =========================
st.header("📌 Pilih Model")

model_files = get_model_files()

if len(model_files) == 0:
    st.error("Belum ada file model di folder `model/`.")
    st.stop()

model_names = [os.path.basename(file) for file in model_files]

selected_model_name = st.selectbox(
    "Pilih model machine learning",
    model_names
)

selected_model_path = model_files[model_names.index(selected_model_name)]

try:
    model = load_model(selected_model_path)
    st.success(f"Model aktif: {selected_model_name}")
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

# =========================
# UPLOAD FILE
# =========================
st.header("📂 Upload File Data")

uploaded_file = st.file_uploader(
    "Upload file CSV atau Excel",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        df = read_uploaded_file(uploaded_file)

        st.subheader("📋 Preview Data Uji")
        st.dataframe(df.head(), use_container_width=True)

        st.write(f"Jumlah data: {df.shape[0]} baris")
        st.write(f"Jumlah kolom: {df.shape[1]} kolom")

        missing_cols = validate_input_columns(df)

        if len(missing_cols) > 0:
            st.error("❌ Kolom pada file belum sesuai dengan fitur yang dibutuhkan model.")

            st.write("**Kolom yang belum ada:**")
            st.write(missing_cols)

            st.write("**Kolom yang harus ada:**")
            st.write(FEATURE_NAMES)

        else:
            input_df = prepare_input_data(df)

            if input_df.isnull().sum().sum() > 0:
                st.error(
                    "❌ Ada nilai kosong atau data non-numerik pada kolom fitur. "
                    "Silakan cek kembali file CSV/Excel."
                )

                missing_value_info = input_df.isnull().sum()
                missing_value_info = missing_value_info[missing_value_info > 0]

                st.write("Jumlah nilai bermasalah per kolom:")
                st.write(missing_value_info)

            else:
                st.success("✅ Format data sudah sesuai. Data siap diprediksi.")

                st.subheader("📊 Data Fitur yang Digunakan untuk Prediksi")
                st.dataframe(input_df.head(), use_container_width=True)

                if st.button("🔍 Prediksi Data", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Sedang melakukan prediksi..."):
                            predictions = model.predict(input_df)

                            result_df = df.copy()
                            result_df["prediction"] = predictions
                            result_df["prediction_label"] = [
                                predict_label(pred) for pred in predictions
                            ]

                            # Hitung distribusi prediksi
                            st.subheader("📈 Distribusi Hasil Prediksi")
                            prediction_counts = result_df["prediction_label"].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.dataframe(prediction_counts, use_container_width=True)
                            
                            with col2:
                                st.bar_chart(prediction_counts)

                            st.subheader("📋 Hasil Prediksi")
                            st.dataframe(result_df, use_container_width=True)

                            csv_result = result_df.to_csv(index=False).encode("utf-8")

                            st.download_button(
                                label="💾 Download Hasil Prediksi (CSV)",
                                data=csv_result,
                                file_name=f"hasil_prediksi_{selected_model_name.replace('.joblib', '').replace('.pkl', '')}.csv",
                                mime="text/csv"
                            )

                            # Tambahkan probabilitas jika model support
                            if hasattr(model, "predict_proba"):
                                st.subheader("📊 Probabilitas Prediksi (Sample)")
                                
                                probabilities = model.predict_proba(input_df.head(5))
                                class_names = model.classes_
                                
                                proba_sample_df = pd.DataFrame(
                                    probabilities,
                                    columns=[f"Prob_{c}" for c in class_names]
                                )
                                
                                st.dataframe(proba_sample_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Terjadi error saat prediksi: {e}")

    except Exception as e:
        st.error(f"File gagal dibaca: {e}")