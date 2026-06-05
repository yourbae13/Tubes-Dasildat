import streamlit as st
import pandas as pd
import joblib
import os
import glob

st.set_page_config(
    page_title="Weather Type Prediction - Manual",
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
if menu == "📊 Prediksi Batch":
    st.switch_page("appWeather_Batch.py")
elif menu == "📈 Evaluasi Model":
    st.switch_page("appEvaluation.py")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Informasi:**
    - Model ML untuk klasifikasi cuaca
    - Jenis cuaca: Rainy, Sunny, Cloudy, Snowy
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

# ==========================
# LOAD MODEL
# ==========================
def get_model_files():
    model_folder = "model"
    joblib_files = glob.glob(os.path.join(model_folder, "*.joblib"))
    pkl_files = glob.glob(os.path.join(model_folder, "*.pkl"))
    return joblib_files + pkl_files


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# ==========================
# HEADER
# ==========================
st.title("🌦️ Weather Type Prediction - Prediksi Manual")

st.write("""
Aplikasi ini digunakan untuk memprediksi jenis cuaca berdasarkan
data meteorologi dan kondisi lingkungan secara manual.

Jenis cuaca yang dapat diprediksi:
- 🌧️ Rainy
- ☀️ Sunny
- ☁️ Cloudy
- ❄️ Snowy
""")

# ==========================
# PILIH MODEL
# ==========================
st.header("📌 Pilih Model")

model_files = get_model_files()

if len(model_files) == 0:
    st.error("Belum ada file model pada folder 'model'.")
    st.stop()

model_names = [os.path.basename(file) for file in model_files]

selected_model_name = st.selectbox(
    "Pilih Model Machine Learning",
    model_names
)

selected_model_path = model_files[
    model_names.index(selected_model_name)
]

try:
    model = load_model(selected_model_path)
    st.success(f"Model aktif: {selected_model_name}")

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==========================
# INPUT DATA
# ==========================
st.header("📝 Input Data")

input_data = {}

col1, col2, col3 = st.columns(3)

with col1:
    input_data["Temperature"] = st.number_input(
        "🌡️ Temperature (°C)",
        value=20.0,
        step=0.5,
        format="%.1f"
    )

    input_data["Humidity"] = st.number_input(
        "💧 Humidity (%)",
        value=50.0,
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        format="%.1f"
    )

    input_data["Wind Speed"] = st.number_input(
        "💨 Wind Speed (km/h)",
        value=10.0,
        step=0.5,
        format="%.1f"
    )

    input_data["Precipitation (%)"] = st.number_input(
        "🌧️ Precipitation (%)",
        value=20.0,
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        format="%.1f"
    )

    input_data["Atmospheric Pressure"] = st.number_input(
        "📊 Atmospheric Pressure (hPa)",
        value=1013.0,
        step=0.5,
        format="%.2f"
    )


with col2:
    input_data["Visibility (km)"] = st.number_input(
        "👁️ Visibility (km)",
        value=10.0,
        min_value=0.0,
        step=0.5,
        format="%.1f"
    )

    st.markdown("### ☁️ Cloud Cover")
    
    cloud_cover_selection = st.radio(
        "Pilih kondisi awan:",
        options=list(CLOUD_COVER_OPTIONS.keys()),
        horizontal=True
    )
    
    for col_name in CLOUD_COVER_OPTIONS.values():
        input_data[col_name] = 0
    
    selected_cloud_col = CLOUD_COVER_OPTIONS[cloud_cover_selection]
    input_data[selected_cloud_col] = 1


with col3:
    st.markdown("### 🌸 Season")
    
    season_selection = st.radio(
        "Pilih musim:",
        options=list(SEASON_OPTIONS.keys()),
        horizontal=True
    )
    
    for col_name in SEASON_OPTIONS.values():
        input_data[col_name] = 0
    
    selected_season_col = SEASON_OPTIONS[season_selection]
    input_data[selected_season_col] = 1


    st.markdown("### 🗺️ Location")
    
    location_selection = st.radio(
        "Pilih lokasi:",
        options=list(LOCATION_OPTIONS.keys()),
        horizontal=True
    )
    
    for col_name in LOCATION_OPTIONS.values():
        input_data[col_name] = 0
    
    selected_location_col = LOCATION_OPTIONS[location_selection]
    input_data[selected_location_col] = 1

# ==========================
# TAMPILAN RINGKASAN INPUT
# ==========================
st.subheader("📊 Ringkasan Input")

display_data = {
    "Parameter": [],
    "Nilai": []
}

numeric_features = [
    "Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
    "Atmospheric Pressure", "Visibility (km)"
]

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

# ==========================
# DATAFRAME INPUT UNTUK MODEL
# ==========================
input_df = pd.DataFrame([input_data])
input_df = input_df[FEATURE_NAMES]

# ==========================
# PREDIKSI
# ==========================
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

        # ==========================
        # PROBABILITAS
        # ==========================
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            class_names = model.classes_

            proba_df = pd.DataFrame({
                "Weather Type": class_names,
                "Probability": probabilities
            })

            st.subheader("📈 Probabilitas Prediksi")

            proba_df_sorted = proba_df.sort_values(
                by="Probability",
                ascending=False
            )

            st.dataframe(
                proba_df_sorted,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                proba_df.set_index("Weather Type"),
                use_container_width=True
            )

        else:
            st.info("Model tidak mendukung predict_proba().")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        st.exception(e)