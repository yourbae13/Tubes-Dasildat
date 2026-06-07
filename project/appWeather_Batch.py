import streamlit as st
import pandas as pd
import joblib
import os
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Weather Batch Prediction",
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
    return glob.glob("model/*.joblib") + glob.glob("model/*.pkl")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def evaluate_model(model):
    df = pd.read_csv("weather_classification_data_cleaned.csv")

    X = df[FEATURE_NAMES]
    y = df["Weather Type"]

    pred = model.predict(X)

    return (
        accuracy_score(y, pred),
        confusion_matrix(y, pred, labels=model.classes_),
        classification_report(y, pred)
    )

def read_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def validate(df):
    return [c for c in FEATURE_NAMES if c not in df.columns]

st.title("🌦️ Weather Batch Prediction")

models = get_model_files()

if not models:
    st.error("Model kosong")
    st.stop()

model_name = st.selectbox("Model", [os.path.basename(m) for m in models])
model = load_model(models[[os.path.basename(m) for m in models].index(model_name)])

menu = st.sidebar.radio("Menu", ["Manual", "CSV/Excel"])

# ======================
# MANUAL
# ======================
if menu == "Manual":
    st.header("Prediksi Manual")

    input_data = {}

    cols = st.columns(3)

    for i, f in enumerate(FEATURE_NAMES):
        with cols[i % 3]:
            input_data[f] = st.number_input(f, 0.0)

    input_df = pd.DataFrame([input_data])[FEATURE_NAMES]

    st.dataframe(input_df)

    if st.button("Prediksi Manual"):
        pred = model.predict(input_df)[0]
        st.success(f"Hasil: {pred}")

        acc, cm, rep = evaluate_model(model)

        st.divider()
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.code(rep)
        st.dataframe(pd.DataFrame(cm))

# ======================
# CSV
# ======================
else:
    st.header("Upload CSV/Excel")

    file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])

    if file:
        df = read_file(file)

        st.dataframe(df.head())

        missing = validate(df)

        if missing:
            st.error("Kolom kurang")
            st.write(missing)
        else:
            input_df = df[FEATURE_NAMES]

            if st.button("Prediksi CSV"):
                pred = model.predict(input_df)

                df["prediction"] = pred

                st.dataframe(df)

                acc, cm, rep = evaluate_model(model)

                st.divider()
                st.metric("Accuracy", f"{acc*100:.2f}%")
                st.code(rep)
                st.dataframe(pd.DataFrame(cm))