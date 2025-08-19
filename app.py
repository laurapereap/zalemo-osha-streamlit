import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Config & Branding
# -----------------------------
st.set_page_config(
    page_title="Zalemo • OSHA Hazard Assistant",
    page_icon="🦺",
    layout="wide"
)

st.image("zalemo_logo.png", width=200)
st.title("OSHA Hazard Assistant")
st.caption("Type a hazard, review similar incidents, and get prediction for Event Title, PPE, Training, Root Causes and OSHA reporting time.")

# -----------------------------
# Load models & data
# -----------------------------
BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "salidas_osha"

@st.cache_resource
def load_model():
    return joblib.load(OUT_DIR / "modelo_family_tfidf_logreg.joblib")

@st.cache_resource
def load_data():
    df = pd.read_csv(OUT_DIR / "df_min.csv", low_memory=False)
    # Ensure we have a "description" column
    if "final_narrative" in df.columns:
        df = df.rename(columns={"final_narrative": "description"})
    if "description" not in df.columns:
        st.error("⚠️ df_min.csv must have a column named 'final_narrative' or 'description'")
    return df

model = load_model()
df = load_data()

# Build TF-IDF index for similarity search
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["description"].astype(str))

# -----------------------------
# Dictionaries
# -----------------------------
ppe_map = {
    "fall": ["Helmet", "Safety shoes", "Harness"],
    "chemical": ["Gloves", "Goggles", "Apron", "Respirator"],
    "fire_explosion": ["Fire-resistant clothing", "Face shield", "Gloves"],
    "caught_in": ["Gloves", "Cut-resistant sleeves"],
    "electrical": ["Insulated gloves", "Face shield", "Arc-rated clothing"],
    "other": ["Gloves"],
}

training_map = {
    "fall": ["Fall protection training", "Working at heights"],
    "chemical": ["HazCom training", "Chemical handling"],
    "fire_explosion": ["Fire safety", "Explosion response"],
    "caught_in": ["Machine guarding", "Lockout/Tagout"],
    "electrical": ["Electrical safety (NFPA 70E)", "Arc flash training"],
    "other": ["General safety awareness"],
}

root_cause_map = {
    "fall": ["Slippery surfaces", "Lack of fall protection", "Poor housekeeping"],
    "chemical": ["Improper labeling", "No PPE use", "Inadequate ventilation"],
    "fire_explosion": ["Flammable storage issues", "Hot work without permit"],
    "caught_in": ["Missing machine guard", "Improper lockout"],
    "electrical": ["Exposed wires", "Overloaded circuits", "Improper grounding"],
    "other": ["Human error", "Lack of supervision"],
}

osha_time_map = {
    "fall": "24 hours",
    "chemical": "8 hours",
    "fire_explosion": "8 hours",
    "electrical": "8 hours",
    "caught_in": "24 hours",
    "other": "24 hours",
}

# -----------------------------
# Input hazard
# -----------------------------
hazard_text = st.text_area("Describe the hazard:", placeholder="Example: Worker slipped on wet floor and fell...")

if hazard_text:
    # Find top-N similar incidents
    hazard_vec = vectorizer.transform([hazard_text])
    sims = cosine_similarity(hazard_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:20]   # show top 20 instead of 10
    matches = df.iloc[top_idx][["description"]].copy()
    matches["similarity"] = sims[top_idx]

    st.subheader("Similar incidents found")
    selected = st.selectbox("Select an incident for analysis:", matches["description"].tolist())

    if selected and st.button("Predict", use_container_width=True, type="primary"):
        y_pred = model.predict([selected])[0]

        st.success(f"**Predicted Event Title:** {y_pred}")
        st.info(f"**Recommended PPE:** {', '.join(ppe_map.get(y_pred, ['N/A']))}")
        st.info(f"**Training Recommendations:** {', '.join(training_map.get(y_pred, ['N/A']))}")
        st.info(f"**Possible Root Causes:** {', '.join(root_cause_map.get(y_pred, ['N/A']))}")
        st.warning(f"**OSHA Reporting Time:** {osha_time_map.get(y_pred, '24 hours')}")
