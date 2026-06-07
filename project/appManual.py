import streamlit as st
import pandas as pd
import joblib
import os
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Weather Type Prediction",
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
# MODEL
# ==========================
def get_model_files():
    model_folder = "model"
    return glob.glob(os.path.join(model_folder, "*.joblib")) + glob.glob(os.path.join(model_folder, "*.pkl"))

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ==========================
# EVALUASI MODEL
# ==========================
def evaluate_model(model):
    try:
        df = pd.read_csv("weather_classification_data_cleaned.csv")

        X = df[FEATURE_NAMES]
        y_true = df["Weather Type"]
        y_pred = model.predict(X)

        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
        report = classification_report(y_true, y_pred)

        return accuracy, cm, report

    except Exception as e:
        return None, None, str(e)

# ==========================
# UI
# ==========================
st.title("🌦️ Weather Type Prediction App")

model_files = get_model_files()

if not model_files:
    st.error("Model belum ada")
    st.stop()

model_names = [os.path.basename(m) for m in model_files]

selected_model = st.selectbox("Pilih Model", model_names)
model = load_model(model_files[model_names.index(selected_model)])

st.success(f"Model aktif: {selected_model}")

st.header("📝 Input Data")

input_data = {}

col1, col2, col3 = st.columns(3)

with col1:
    input_data["Temperature"] = st.number_input("Temperature", 20.0)
    input_data["Humidity"] = st.number_input("Humidity", 50.0)
    input_data["Wind Speed"] = st.number_input("Wind Speed", 10.0)
    input_data["Precipitation (%)"] = st.number_input("Precipitation (%)", 20.0)
    input_data["Atmospheric Pressure"] = st.number_input("Pressure", 1013.0)

with col2:
    input_data["Visibility (km)"] = st.number_input("Visibility", 10.0)
    cloud = st.selectbox("Cloud Cover", list(CLOUD_COVER_OPTIONS.keys()))

with col3:
    season = st.selectbox("Season", list(SEASON_OPTIONS.keys()))
    location = st.selectbox("Location", list(LOCATION_OPTIONS.keys()))

    for c in LOCATION_OPTIONS.values():
        input_data[c] = 0

    input_data[LOCATION_OPTIONS[location]] = 1

    for c in SEASON_OPTIONS.values():
        input_data[c] = 0

    input_data[SEASON_OPTIONS[season]] = 1

    for c in CLOUD_COVER_OPTIONS.values():
        input_data[c] = 0

    input_data[CLOUD_COVER_OPTIONS[cloud]] = 1

input_df = pd.DataFrame([input_data])[FEATURE_NAMES]

st.dataframe(input_df)

if st.button("🔍 Prediksi"):
    pred = model.predict(input_df)[0]
    st.success(f"Hasil: {pred}")

    accuracy, cm, report = evaluate_model(model)

    if accuracy:
        st.divider()
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
        st.code(report)
        st.dataframe(pd.DataFrame(cm))