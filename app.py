import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ===============================
# Page config + simple CSS
# ===============================
st.set_page_config(page_title="Zalemo • OSHA Hazard Assistant", page_icon="🦺", layout="wide")

CUSTOM_CSS = """
<style>
/* Center header area */
.header-wrap {text-align:center; margin-top: 0.5rem; margin-bottom: 0.25rem;}
.header-title {font-size: 2.2rem; font-weight: 800; margin: 0.25rem 0 0.1rem;}
.header-sub {color:#475569; font-size: 0.98rem; margin-bottom: 1.25rem;}
/* Nice info blocks */
.block {padding: 0.9rem 1rem; border-radius: 0.75rem; margin: 0.35rem 0;}
.block.green {background:#ecfdf5; border:1px solid #a7f3d0;}
.block.blue {background:#eff6ff; border:1px solid #bfdbfe;}
.block.yellow {background:#fffbeb; border:1px solid #fde68a;}
/* Footer */
.footer {margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; color:#64748b; font-size:0.9rem; text-align:center;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===============================
# Header (centered)
# ===============================
colL, colC, colR = st.columns([1,2,1])
with colC:
    st.image("zalemo_logo.png", use_container_width=True)
    st.markdown('<div class="header-wrap"><div class="header-title">OSHA Hazard Assistant</div>'
                '<div class="header-sub">Type a hazard, review similar incidents, and get prediction for Event Title, PPE, Training, Root Causes and OSHA reporting time.</div></div>',
                unsafe_allow_html=True)

# ===============================
# Load model & data
# ===============================
BASE = Path(__file__).resolve().parent
OUT = BASE / "salidas_osha"

@st.cache_resource
def load_model():
    # Your pipeline with TF-IDF + classifier
    return joblib.load(OUT / "modelo_family_tfidf_logreg.joblib")

@st.cache_resource
def load_data():
    df = pd.read_csv(OUT / "df_min.csv", low_memory=False)
    # Standardize text column name
    if "final_narrative" in df.columns:
        df = df.rename(columns={"final_narrative": "description"})
    if "description" not in df.columns:
        st.error("df_min.csv must contain 'final_narrative' or 'description' column.")
        st.stop()
    # Clean
    df["description"] = df["description"].astype(str).str.strip()
    df = df.dropna(subset=["description"])
    df = df[df["description"].str.len() > 5]
    df = df.drop_duplicates(subset=["description"])
    return df.reset_index(drop=True)

model = load_model()
df = load_data()

@st.cache_resource
def build_tfidf_index(texts: pd.Series):
    vect = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1,2),
        sublinear_tf=True,
        max_df=0.9,
        min_df=3
    )
    X = vect.fit_transform(texts.astype(str))
    return vect, X

vectorizer, tfidf_matrix = build_tfidf_index(df["description"])

# ===============================
# Dictionaries (refine as needed)
# ===============================
ppe_map = {
    "fall": ["Helmet", "Safety shoes", "Harness"],
    "chemical": ["Gloves", "Goggles", "Apron", "Respirator"],
    "fire_explosion": ["Fire-resistant clothing", "Face shield", "Gloves"],
    "caught_in": ["Gloves", "Cut-resistant sleeves"],
    "electrical": ["Insulated gloves", "Face shield", "Arc-rated clothing"],
    "struck_by_vehicle": ["High-visibility vest", "Safety shoes", "Hard hat"],
    "other": ["Gloves"],
}

training_map = {
    "fall": ["Fall protection", "Working at heights"],
    "chemical": ["HazCom", "Chemical handling / SDS"],
    "fire_explosion": ["Fire safety", "Hot work permits"],
    "caught_in": ["Machine guarding", "Lockout/Tagout"],
    "electrical": ["Electrical safety (NFPA 70E)", "Arc flash"],
    "struck_by_vehicle": ["Powered industrial trucks (forklift)", "Traffic management / spotter"],
    "other": ["General safety awareness"],
}

root_cause_map = {
    "fall": ["Slippery surfaces", "Lack of fall protection", "Poor housekeeping"],
    "chemical": ["Improper labeling", "Inadequate ventilation", "Missing PPE"],
    "fire_explosion": ["Flammable storage issues", "Hot work without permit"],
    "caught_in": ["Missing machine guard", "Improper lockout"],
    "electrical": ["Exposed wires", "Overloaded circuits", "Improper grounding"],
    "struck_by_vehicle": ["No traffic plan", "Poor visibility", "Speeding / inattentive driving"],
    "other": ["Human error", "Lack of supervision"],
}

osha_time_map = {
    "fall": "24 hours",
    "chemical": "8 hours",
    "fire_explosion": "8 hours",
    "electrical": "8 hours",
    "caught_in": "24 hours",
    "struck_by_vehicle": "24 hours",
    "other": "24 hours",
}

# Post-process label fixer for common keywords (improves UX when model returns "other")
def fix_label(text: str, pred: str) -> str:
    t = text.lower()
    if re.search(r"\bforklift|powered\s*(industrial\s*)?truck|pallet\s*jack|hi-?lo\b", t):
        return "struck_by_vehicle"
    if "electric" in t or "shock" in t or "arc flash" in t:
        return "electrical"
    if "acid" in t or "chemical" in t or "solvent" in t or "fume" in t:
        return "chemical"
    if "burn" in t or "fire" in t or "explosion" in t:
        return "fire_explosion"
    if "ladder" in t or "fell" in t or "slip" in t or "trip" in t:
        return "fall"
    if "caught" in t or "pinch" in t or "unguarded" in t:
        return "caught_in"
    return pred

# ===============================
# UI – Search & Results
# ===============================
st.write("")

# Controls
colA, colB, colC = st.columns([2,1,1])
with colA:
    hazard_text = st.text_input("Describe the hazard:", placeholder="e.g., Forklift struck a worker while reversing in warehouse")
with colB:
    topn = st.slider("Results to show", 10, 100, 30, step=10)
with colC:
    min_sim = st.slider("Min similarity", 0.0, 0.5, 0.05, step=0.01)

if hazard_text:
    # Similarity search
    q_vec = vectorizer.transform([hazard_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    idx = sims.argsort()[::-1]
    # Filter by threshold & take topn
    filtered = [(i, sims[i]) for i in idx if sims[i] >= min_sim][:topn]
    res_df = pd.DataFrame(
        {"description": [df.iloc[i]["description"] for i, _ in filtered],
         "similarity": [float(s) for _, s in filtered]}
    ).drop_duplicates(subset=["description"]).reset_index(drop=True)

    if res_df.empty:
        st.warning("No similar incidents found. Try adjusting the threshold or rewriting the query.")
    else:
        st.subheader("Similar incidents found")
        # Select with a radio (click = immediate predict; no Ctrl+Enter)
        choice = st.radio("Select an incident for analysis:", res_df["description"].tolist(), index=0)

        # Auto-predict on selection
        if choice:
            pred = model.predict([choice])[0]
            label = fix_label(choice, pred)

            st.markdown(f'<div class="block green"><b>Predicted Event Title:</b> {label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="block blue"><b>Recommended PPE:</b> {", ".join(ppe_map.get(label, ["N/A"]))}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="block blue"><b>Training Recommendations:</b> {", ".join(training_map.get(label, ["N/A"]))}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="block blue"><b>Possible Root Causes:</b> {", ".join(root_cause_map.get(label, ["N/A"]))}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="block yellow"><b>OSHA Reporting Time:</b> {osha_time_map.get(label, "24 hours")}</div>', unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown('<div class="footer">© 2025 Zalemo Corporation — Internal demo for safety support. '
            'Not a substitute for professional judgment or legal advice.</div>', unsafe_allow_html=True)
