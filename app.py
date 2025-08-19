
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Zalemo • OSHA Incident Classifier", page_icon="🛡️", layout="wide")

# --- Paths
BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "salidas_osha"

# --- Header / Branding
col_logo, col_title = st.columns([1,5])
with col_logo:
    logo_path = BASE / "zalemo_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_column_width=True)
with col_title:
    st.title("OSHA Incident Classifier")
    st.caption("Predicción de **familia de accidente** a partir de la descripción.")

# --- Utility to load models (joblib pipelines that already include TF-IDF)
@st.cache_resource
def load_model(model_filename: str):
    model_path = OUT_DIR / model_filename
    if not model_path.exists():
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"No se pudo cargar el modelo {model_filename}: {e}")
        return None

# --- Sidebar
st.sidebar.header("Opciones")
available_models = [f.name for f in OUT_DIR.glob("*.joblib")]
if not available_models:
    st.sidebar.warning("Copia tus archivos *.joblib en **./salidas_osha**.")
model_name = st.sidebar.selectbox("Modelo", options=available_models or ["(sin modelos)"])

# Try to load explainability terms if present
top_terms = None
top_terms_path = OUT_DIR / "top_terms_per_class.json"
if top_terms_path.exists():
    try:
        with open(top_terms_path, "r", encoding="utf-8") as f:
            top_terms = json.load(f)
    except Exception as e:
        st.sidebar.warning(f"No se pudieron leer los términos por clase: {e}")

# --- Main prediction box
st.subheader("Haz una predicción")
default_text = "Employee slipped on wet floor and fell, injuring lower back and wrist."
text = st.text_area("Descripción del incidente (inglés):", value=default_text, height=140)

predict_btn = st.button("Predecir", use_container_width=True, type="primary")
if predict_btn:
    if model_name and model_name != "(sin modelos)":
        model = load_model(model_name)
        if model is not None:
            try:
                y_pred = model.predict([text])[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([text])[0]
                    classes = getattr(model, "classes_", None)
                else:
                    # fallback: decision_function -> softmax-like
                    classes = getattr(model, "classes_", None)
                    proba = None
                st.success(f"**Predicción:** {y_pred}")
                if proba is not None and classes is not None:
                    dfp = pd.DataFrame({"class": classes, "prob": proba}).sort_values("prob", ascending=False).head(5)
                    st.bar_chart(dfp.set_index("class"))
                # Show top terms for predicted class if available
                if top_terms and y_pred in top_terms:
                    st.markdown("**Términos más característicos (según el modelo):**")
                    st.write(", ".join(top_terms[y_pred][:20]))
            except Exception as e:
                st.error(f"Error al predecir: {e}")
    else:
        st.warning("Selecciona un modelo en la barra lateral.")

st.divider()

# --- Metrics and artifacts
st.subheader("Artefactos del modelo")
col1, col2 = st.columns(2)

with col1:
    # Metrics comparison table
    mcsv = OUT_DIR / "metricas_comparacion.csv"
    if mcsv.exists():
        dfm = pd.read_csv(mcsv)
        st.markdown("**Comparación de modelos (valid/test)**")
        st.dataframe(dfm, use_container_width=True)
    else:
        st.info("Sube **metricas_comparacion.csv** para ver el resumen.")

    # Text metrics summary
    mtxt = OUT_DIR / "resumen_metricas.txt"
    if mtxt.exists():
        st.markdown("**Resumen de métricas**")
        st.code(mtxt.read_text(encoding="utf-8"), language="text")

with col2:
    # Confusion matrix and F1 comparison images
    img_paths = [
        OUT_DIR / "matriz_confusion_svm_cal_test.png",
        OUT_DIR / "comparacion_f1_test.png",
    ]
    for p in img_paths:
        if p.exists():
            st.image(str(p), caption=p.name, use_column_width=True)

st.divider()

# --- Data preview (optional small file)
st.subheader("Muestra de datos (opcional)")
small_csv = OUT_DIR / "df_min.csv"
if small_csv.exists():
    try:
        df_small = pd.read_csv(small_csv).head(200)
        st.dataframe(df_small, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo leer df_min.csv: {e}")
else:
    st.caption("Puedes incluir un subconjunto anónimo en **df_min.csv** (no obligatoria).")

st.caption("© Zalemo — Demo académica (TFM).")
