import streamlit as st
import pandas as pd
import joblib
import os
import glob

st.set_page_config(
    page_title="Weather Classification Prediction",
    page_icon="🌦️",
    layout="wide"
)

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


# ==========================
# SIDEBAR MENU (PILIH HALAMAN)
# ==========================
st.sidebar.title("📋 Menu")
st.sidebar.markdown("---")

# Pilihan halaman
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Prediksi Batch", "✏️ Prediksi Manual", "📈 Evaluasi Model"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Informasi:**
    - Upload file CSV/Excel untuk prediksi batch
    - Hasil akan ditampilkan dan bisa di-download
    """
)

# ==========================
# TAMPILAN BERDASARKAN PILIHAN
# ==========================

# ========== HALAMAN 1: PREDIKSI BATCH ==========
if page == "📊 Prediksi Batch":
    st.title("🌦️ Weather Classification Prediction - Batch")
    st.write(
        "Aplikasi ini digunakan untuk memprediksi jenis cuaca "
        "berdasarkan data meteorologi menggunakan beberapa model machine learning."
    )

    # PILIH MODEL
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

    # UPLOAD FILE
    st.header("📂 Upload File Data")

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)

            st.subheader("Preview Data Uji")
            st.dataframe(df.head(), use_container_width=True)

            st.write(f"Jumlah data: {df.shape[0]} baris")
            st.write(f"Jumlah kolom: {df.shape[1]} kolom")

            missing_cols = validate_input_columns(df)

            if len(missing_cols) > 0:
                st.error("Kolom pada file belum sesuai dengan fitur yang dibutuhkan model.")
                st.write("Kolom yang belum ada:", missing_cols)
                st.write("Kolom yang harus ada:", FEATURE_NAMES)

            else:
                input_df = prepare_input_data(df)

                if input_df.isnull().sum().sum() > 0:
                    st.error(
                        "Ada nilai kosong atau data non-numerik pada kolom fitur. "
                        "Silakan cek kembali file CSV/Excel."
                    )
                    missing_value_info = input_df.isnull().sum()
                    missing_value_info = missing_value_info[missing_value_info > 0]
                    st.write("Jumlah nilai bermasalah per kolom:", missing_value_info)

                else:
                    st.success("Format data sudah sesuai. Data siap diprediksi.")

                    st.subheader("Data Fitur yang Digunakan untuk Prediksi")
                    st.dataframe(input_df.head(), use_container_width=True)

                    if st.button("Prediksi Data CSV/Excel", type="primary", use_container_width=True):
                        try:
                            with st.spinner("Sedang melakukan prediksi..."):
                                predictions = model.predict(input_df)

                                result_df = df.copy()
                                result_df["prediction"] = predictions
                                result_df["prediction_label"] = [
                                    predict_label(pred) for pred in predictions
                                ]

                                st.subheader("📈 Distribusi Hasil Prediksi")
                                prediction_counts = result_df["prediction_label"].value_counts()
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.dataframe(prediction_counts, use_container_width=True)
                                with col2:
                                    st.bar_chart(prediction_counts)

                                st.subheader("Hasil Prediksi")
                                st.dataframe(result_df, use_container_width=True)

                                csv_result = result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="Download Hasil Prediksi (CSV)",
                                    data=csv_result,
                                    file_name="hasil_prediksi_weather.csv",
                                    mime="text/csv"
                                )

                                if hasattr(model, "predict_proba"):
                                    st.subheader("📊 Probabilitas Prediksi (Sample 5 data pertama)")
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


# ========== HALAMAN 2: PREDIKSI MANUAL ==========
elif page == "✏️ Prediksi Manual":
    st.title("✏️ Prediksi Manual - Weather Type")
    
    # Mapping untuk tampilan user-friendly
    CLOUD_COVER_OPTIONS = {
        "Clear": "Cc_clear",
        "Cloudy": "Cc_cloudy",
        "Overcast": "Cc_overcast",
        "Partly Cloudy": "Cc_partly cloudy"
    }
    SEASON_OPTIONS = {
        "Autumn": "Ss_Autumn",
        "Spring": "Ss_Spring",
        "Summer": "Ss_Summer",
        "Winter": "Ss_Winter"
    }
    LOCATION_OPTIONS = {
        "Coastal": "Lc_coastal",
        "Inland": "Lc_inland",
        "Mountain": "Lc_mountain"
    }

    # PILIH MODEL
    st.header("📌 Pilih Model")
    model_files = get_model_files()
    if len(model_files) == 0:
        st.error("Belum ada file model di folder `model/`.")
        st.stop()
    model_names = [os.path.basename(file) for file in model_files]
    selected_model_name = st.selectbox("Pilih model machine learning", model_names)
    selected_model_path = model_files[model_names.index(selected_model_name)]
    try:
        model = load_model(selected_model_path)
        st.success(f"Model aktif: {selected_model_name}")
    except Exception as e:
        st.error(f"Model gagal dimuat: {e}")
        st.stop()

    # INPUT DATA
    st.header("📝 Input Data")
    input_data = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["Temperature"] = st.number_input("🌡️ Temperature (°C)", value=20.0, step=0.5, format="%.1f")
        input_data["Humidity"] = st.number_input("💧 Humidity (%)", value=50.0, min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
        input_data["Wind Speed"] = st.number_input("💨 Wind Speed (km/h)", value=10.0, step=0.5, format="%.1f")
        input_data["Precipitation (%)"] = st.number_input("🌧️ Precipitation (%)", value=20.0, min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
        input_data["Atmospheric Pressure"] = st.number_input("📊 Atmospheric Pressure (hPa)", value=1013.0, step=0.5, format="%.2f")

    with col2:
        input_data["Visibility (km)"] = st.number_input("👁️ Visibility (km)", value=10.0, min_value=0.0, step=0.5, format="%.1f")
        st.markdown("### ☁️ Cloud Cover")
        cloud_cover_selection = st.radio("Pilih kondisi awan:", options=list(CLOUD_COVER_OPTIONS.keys()), horizontal=True)
        for col_name in CLOUD_COVER_OPTIONS.values():
            input_data[col_name] = 0
        selected_cloud_col = CLOUD_COVER_OPTIONS[cloud_cover_selection]
        input_data[selected_cloud_col] = 1

    with col3:
        st.markdown("### 🌸 Season")
        season_selection = st.radio("Pilih musim:", options=list(SEASON_OPTIONS.keys()), horizontal=True)
        for col_name in SEASON_OPTIONS.values():
            input_data[col_name] = 0
        selected_season_col = SEASON_OPTIONS[season_selection]
        input_data[selected_season_col] = 1

        st.markdown("### 🗺️ Location")
        location_selection = st.radio("Pilih lokasi:", options=list(LOCATION_OPTIONS.keys()), horizontal=True)
        for col_name in LOCATION_OPTIONS.values():
            input_data[col_name] = 0
        selected_location_col = LOCATION_OPTIONS[location_selection]
        input_data[selected_location_col] = 1

    # RINGKASAN INPUT
    st.subheader("📊 Ringkasan Input")
    display_data = {"Parameter": [], "Nilai": []}
    numeric_features = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", "Atmospheric Pressure", "Visibility (km)"]
    for feat in numeric_features:
        display_data["Parameter"].append(feat)
        display_data["Nilai"].append(input_data[feat])
    display_data["Parameter"].append("Cloud Cover")
    cloud_cover_value = [k for k, v in CLOUD_COVER_OPTIONS.items() if input_data[v] == 1][0]
    display_data["Nilai"].append(cloud_cover_value)
    display_data["Parameter"].append("Season")
    season_value = [k for k, v in SEASON_OPTIONS.items() if input_data[v] == 1][0]
    display_data["Nilai"].append(season_value)
    display_data["Parameter"].append("Location")
    location_value = [k for k, v in LOCATION_OPTIONS.items() if input_data[v] == 1][0]
    display_data["Nilai"].append(location_value)
    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # PREDIKSI
    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_NAMES]

    if st.button("🔍 Prediksi Cuaca", type="primary", use_container_width=True):
        try:
            prediction = model.predict(input_df)[0]
            st.subheader("🎯 Hasil Prediksi")
            if prediction == "Rainy":
                st.success("🌧️ **Prediksi Cuaca: Rainy**")
                st.balloons()
            elif prediction == "Sunny":
                st.success("☀️ **Prediksi Cuaca: Sunny**")
                st.balloons()
            elif prediction == "Cloudy":
                st.success("☁️ **Prediksi Cuaca: Cloudy**")
            elif prediction == "Snowy":
                st.success("❄️ **Prediksi Cuaca: Snowy**")
                st.snow()
            else:
                st.success(f"**Hasil Prediksi: {prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                class_names = model.classes_
                proba_df = pd.DataFrame({"Weather Type": class_names, "Probability": probabilities})
                st.subheader("📈 Probabilitas Prediksi")
                st.dataframe(proba_df.sort_values(by="Probability", ascending=False), use_container_width=True, hide_index=True)
                st.bar_chart(proba_df.set_index("Weather Type"), use_container_width=True)
            else:
                st.info("Model tidak mendukung predict_proba().")
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            st.exception(e)


# ========== HALAMAN 3: EVALUASI MODEL ==========
elif page == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model - Weather Classification")
    
    TARGET_COLUMN = "Weather Type"

    # PILIH MODEL
    st.header("📌 Pilih Model")
    model_files = get_model_files()
    if len(model_files) == 0:
        st.error("Belum ada file model di folder `model/`.")
        st.stop()
    model_names = [os.path.basename(file) for file in model_files]
    selected_model_name = st.selectbox("Pilih model machine learning", model_names)
    selected_model_path = model_files[model_names.index(selected_model_name)]
    try:
        model = load_model(selected_model_path)
        st.success(f"Model aktif: {selected_model_name}")
    except Exception as e:
        st.error(f"Model gagal dimuat: {e}")
        st.stop()

    # UPLOAD FILE TEST
    st.header("📂 Upload Data Test")
    st.write("Upload file CSV/Excel yang berisi data test **LENGKAP DENGAN LABEL** (kolom 'Weather Type').")
    
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name.lower()
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.subheader("📋 Preview Data Test")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Jumlah data: {df.shape[0]} baris, {df.shape[1]} kolom")
            
            if TARGET_COLUMN not in df.columns:
                st.error(f"❌ Kolom '{TARGET_COLUMN}' tidak ditemukan!")
                st.stop()
            
            missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Kolom fitur yang hilang: {missing_cols[:5]}...")
                st.stop()
            
            X_test = df[FEATURE_NAMES].copy()
            y_true = df[TARGET_COLUMN].copy()
            
            for col in FEATURE_NAMES:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            
            if X_test.isnull().sum().sum() > 0:
                st.error("❌ Ada nilai kosong pada data test.")
                st.stop()
            
            st.success("✅ Data test valid dan siap dievaluasi!")
            
            if st.button("📊 Evaluasi Model", type="primary", use_container_width=True):
                with st.spinner("Sedang melakukan evaluasi..."):
                    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_true, y_pred)
                    cm = confusion_matrix(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True)
                    
                    st.header("📈 Hasil Evaluasi")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Accuracy Score", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
                    with col2:
                        st.metric("📊 Jumlah Kelas", len(set(y_true)))
                    with col3:
                        st.metric("📝 Jumlah Data Test", len(X_test))
                    with col4:
                        correct_pred = (y_pred == y_true).sum()
                        st.metric("✅ Prediksi Benar", correct_pred, f"{(correct_pred/len(X_test))*100:.1f}%")
                    
                    st.subheader("🔢 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    classes = sorted(set(y_true) | set(y_pred))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax, annot_kws={'size': 12})
                    ax.set_xlabel('Predicted Label', fontsize=12)
                    ax.set_ylabel('True Label', fontsize=12)
                    ax.set_title(f'Confusion Matrix - {selected_model_name}', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    
                    st.subheader("📋 Classification Report")
                    report_df = pd.DataFrame(report).transpose()
                    if 'accuracy' in report_df.index:
                        report_df = report_df.drop(index=['accuracy'])
                    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                    
                    st.subheader("💾 Download Hasil Evaluasi")
                    result_df = df.copy()
                    result_df['Predicted'] = y_pred
                    result_df['Correct'] = y_true == y_pred
                    csv_result = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Hasil Prediksi (CSV)", data=csv_result, file_name=f"evaluation_results.csv", mime="text/csv")
                    
        except Exception as e:
            st.error(f"Terjadi error: {e}")
            st.exception(e)
    else:
        st.info("📂 Silakan upload file data test (CSV atau Excel) untuk memulai evaluasi.")