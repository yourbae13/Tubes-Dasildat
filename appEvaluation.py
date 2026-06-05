import streamlit as st
import pandas as pd
import joblib
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Model Evaluation - Weather Classification",
    page_icon="📊",
    layout="wide"
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

TARGET_COLUMN = "Weather Type"

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("📋 Menu")
st.sidebar.markdown("---")

st.sidebar.page_link("appManual.py", label="🏠 Prediksi Manual", icon="🏠")
st.sidebar.page_link("appWeather_Batch.py", label="📊 Prediksi Batch", icon="📊")
st.sidebar.page_link("appEvaluation.py", label="📈 Evaluasi Model", icon="📈")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Informasi:**
    - Evaluasi performa model
    - Upload file dengan label 'Weather Type'
    - Menampilkan Accuracy, Confusion Matrix, Classification Report
    """
)

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
st.title("📊 Model Evaluation - Weather Classification")

st.write("""
Halaman ini digunakan untuk mengevaluasi performa model machine learning 
dengan menghitung **Accuracy**, **Confusion Matrix**, dan **Classification Report**.
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

selected_model_path = model_files[model_names.index(selected_model_name)]

try:
    model = load_model(selected_model_path)
    st.success(f"✅ Model aktif: {selected_model_name}")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==========================
# UPLOAD DATA TEST
# ==========================
st.header("📂 Upload Data Test")

st.write("""
Upload file CSV/Excel yang berisi data test **LENGKAP DENGAN LABEL** (kolom 'Weather Type').
""")

uploaded_file = st.file_uploader(
    "Upload file CSV atau Excel",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        # Baca file
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.subheader("📋 Preview Data Test")
        st.dataframe(df.head(), use_container_width=True)
        
        st.write(f"Jumlah data: {df.shape[0]} baris, {df.shape[1]} kolom")
        
        # Cek apakah kolom target ada
        if TARGET_COLUMN not in df.columns:
            st.error(f"❌ Kolom '{TARGET_COLUMN}' tidak ditemukan! File harus berisi label sebenarnya.")
            st.stop()
        
        # Cek kolom fitur
        missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Kolom fitur yang hilang: {missing_cols[:5]}...")
            if len(missing_cols) > 5:
                st.write(f"Dan {len(missing_cols)-5} kolom lainnya")
            st.stop()
        
        # Siapkan data
        X_test = df[FEATURE_NAMES].copy()
        y_true = df[TARGET_COLUMN].copy()
        
        # Konversi ke numerik
        for col in FEATURE_NAMES:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        
        # Cek nilai kosong
        if X_test.isnull().sum().sum() > 0:
            st.error("❌ Ada nilai kosong pada data test. Silakan bersihkan data terlebih dahulu.")
            st.write("Jumlah nilai kosong per kolom:")
            st.write(X_test.isnull().sum())
            st.stop()
        
        st.success("✅ Data test valid dan siap dievaluasi!")
        
        # ==========================
        # TOMBOL EVALUASI
        # ==========================
        if st.button("📊 Evaluasi Model", type="primary", use_container_width=True):
            with st.spinner("Sedang melakukan evaluasi..."):
                # Prediksi
                y_pred = model.predict(X_test)
                
                # Hitung metrics
                accuracy = accuracy_score(y_true, y_pred)
                
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Classification Report
                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                # ==========================
                # TAMPILAN HASIL
                # ==========================
                st.header("📈 Hasil Evaluasi")
                
                # Metrics dalam card
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="🎯 Accuracy Score",
                        value=f"{accuracy:.4f}",
                        delta=f"{accuracy*100:.2f}%"
                    )
                
                with col2:
                    unique_classes = len(set(y_true))
                    st.metric(
                        label="📊 Jumlah Kelas",
                        value=unique_classes
                    )
                
                with col3:
                    st.metric(
                        label="📝 Jumlah Data Test",
                        value=len(X_test)
                    )
                
                with col4:
                    correct_pred = (y_pred == y_true).sum()
                    st.metric(
                        label="✅ Prediksi Benar",
                        value=correct_pred,
                        delta=f"{(correct_pred/len(X_test))*100:.1f}%"
                    )
                
                # ==========================
                # CONFUSION MATRIX
                # ==========================
                st.subheader("🔢 Confusion Matrix")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                classes = sorted(set(y_true) | set(y_pred))
                
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=classes,
                    yticklabels=classes,
                    ax=ax,
                    annot_kws={'size': 12}
                )
                
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_title(f'Confusion Matrix - {selected_model_name}', fontsize=14, fontweight='bold')
                
                st.pyplot(fig)
                
                # Tampilkan juga dalam bentuk dataframe
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"True: {c}" for c in classes],
                    columns=[f"Pred: {c}" for c in classes]
                )
                
                with st.expander("📊 Lihat Confusion Matrix dalam bentuk tabel"):
                    st.dataframe(cm_df, use_container_width=True)
                
                # ==========================
                # CLASSIFICATION REPORT
                # ==========================
                st.subheader("📋 Classification Report")
                
                display_report = report_df.copy()
                
                accuracy_value = None
                if 'accuracy' in display_report.index:
                    accuracy_value = display_report.loc['accuracy', 'precision'] if 'precision' in display_report.columns else None
                    display_report = display_report.drop(index=['accuracy'])
                
                st.dataframe(
                    display_report.style.format("{:.4f}"),
                    use_container_width=True
                )
                
                if accuracy_value:
                    st.info(f"**📊 Overall Accuracy:** {accuracy_value:.4f} ({accuracy_value*100:.2f}%)")
                
                # ==========================
                # DOWNLOAD BUTTON
                # ==========================
                st.subheader("💾 Download Hasil Evaluasi")
                
                result_df = df.copy()
                result_df['Predicted'] = y_pred
                result_df['Correct'] = y_true == y_pred
                
                csv_result = result_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv_result,
                    file_name=f"evaluation_results_{selected_model_name.replace('.joblib', '').replace('.pkl', '')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # ==========================
                # INTERPRETASI
                # ==========================
                st.subheader("📖 Interpretasi Hasil")
                
                if accuracy >= 0.9:
                    st.success(f"✅ Model **{selected_model_name}** memiliki performa **SANGAT BAIK** dengan akurasi {accuracy*100:.2f}%.")
                elif accuracy >= 0.75:
                    st.info(f"📈 Model **{selected_model_name}** memiliki performa **BAIK** dengan akurasi {accuracy*100:.2f}%.")
                elif accuracy >= 0.6:
                    st.warning(f"⚠️ Model **{selected_model_name}** memiliki performa **CUKUP** dengan akurasi {accuracy*100:.2f}%.")
                else:
                    st.error(f"❌ Model **{selected_model_name}** memiliki performa **KURANG** dengan akurasi {accuracy*100:.2f}%. Perlu improvement.")
                
                # Analisis per kelas
                st.write("**📊 Performa per kelas:**")
                
                class_summary = []
                for class_name in classes:
                    if class_name in report:
                        class_summary.append({
                            "Kelas": class_name,
                            "Precision": f"{report[class_name]['precision']:.3f}",
                            "Recall": f"{report[class_name]['recall']:.3f}",
                            "F1-Score": f"{report[class_name]['f1-score']:.3f}",
                            "Support": report[class_name]['support']
                        })
                
                st.dataframe(pd.DataFrame(class_summary), use_container_width=True, hide_index=True)
                
    except Exception as e:
        st.error(f"Terjadi error: {e}")
        st.exception(e)

else:
    st.info("📂 Silakan upload file data test (CSV atau Excel) untuk memulai evaluasi.")
    
    with st.expander("ℹ️ Lihat contoh format file yang diperlukan"):
        st.write("**File harus memiliki kolom-kolom berikut:**")
        
        sample_data = {}
        for feat in FEATURE_NAMES[:6]:
            sample_data[feat] = [20.0, 50.0, 10.0, 20.0, 1013.0, 10.0]
        
        for feat in FEATURE_NAMES[6:10]:
            sample_data[feat] = [1, 0, 0, 0]
        
        for feat in FEATURE_NAMES[10:14]:
            sample_data[feat] = [0, 1, 0, 0]
        
        for feat in FEATURE_NAMES[14:]:
            sample_data[feat] = [1, 0, 0]
        
        sample_data[TARGET_COLUMN] = ["Sunny"]
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.caption("**Catatan:** Nilai fitur one-hot encoding hanya boleh 0 atau 1, dan hanya satu yang bernilai 1 per grup.")
        st.caption(f"**Kolom target:** '{TARGET_COLUMN}' berisi label sebenarnya (Rainy, Sunny, Cloudy, Snowy).")