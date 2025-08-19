import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re, base64

# ===============================
# Page config + CSS
# ===============================
st.set_page_config(page_title="Zalemo • OSHA Hazard Assistant", page_icon="🦺", layout="wide")

CUSTOM_CSS = """
<style>
.header-wrap {text-align:center; margin-top: 0.25rem; margin-bottom: 0.25rem;}
.header-title {font-size: 2.0rem; font-weight: 800; margin: 0.25rem 0 0.1rem;}
.header-sub {color:#475569; font-size: 0.98rem; margin-bottom: 1.0rem;}
.block {padding: 0.9rem 1rem; border-radius: 0.75rem; margin: 0.35rem 0;}
.block.green {background:#ecfdf5; border:1px solid #a7f3d0;}
.block.blue {background:#eff6ff; border:1px solid #bfdbfe;}
.block.yellow {background:#fffbeb; border:1px solid #fde68a;}
.footer {margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; color:#64748b; font-size:0.9rem; text-align:center;}
.small-note {color:#64748b; font-size:0.85rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===============================
# Header (centered + larger logo)
# ===============================
def centered_logo_html(logo_path: Path, width_px: int = 220) -> str:
    try:
        b64 = base64.b64encode(logo_path.read_bytes()).decode()
        return f"""
        <div class="header-wrap">
            <img src="data:image/png;base64,{b64}" width="{width_px}" />
            <div class="header-title">OSHA Hazard Assistant</div>
            <div class="header-sub">
                Type a hazard, review similar incidents, and get prediction for Event Title,
                PPE, Training, Root Causes and OSHA reporting time.
            </div>
        </div>
        """
    except Exception:
        # Fallback if logo not found
        return """
        <div class="header-wrap">
            <div class="header-title">OSHA Hazard Assistant</div>
            <div class="header-sub">
                Type a hazard, review similar incidents, and get prediction for Event Title,
                PPE, Training, Root Causes and OSHA reporting time.
            </div>
        </div>
        """

BASE = Path(__file__).resolve().parent
st.markdown(centered_logo_html(BASE / "zalemo_logo.png", width_px=220), unsafe_allow_html=True)

# ===============================
# Load model & data
# ===============================
OUT = BASE / "salidas_osha"

@st.cache_resource
def load_model():
    # sklearn Pipeline with TF-IDF + classifier
    return joblib.load(OUT / "modelo_family_tfidf_logreg.joblib")

@st.cache_resource
def load_data():
    df = pd.read_csv(OUT / "df_min.csv", low_memory=False)
    # use OSHA narrative column
    if "final_narrative" in df.columns:
        df = df.rename(columns={"final_narrative": "description"})
    if "description" not in df.columns:
        st.error("df_min.csv must contain 'final_narrative' or 'description'.")
        st.stop()
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
# Dictionaries
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

# Heuristic label fixer (corrige "other" en casos frecuentes)
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
c1, c2 = st.columns([3,1])
with c1:
    hazard_text = st.text_input("Describe the hazard:", placeholder="e.g., Forklift struck a worker while reversing in warehouse")
with c2:
    topn = st.slider("Results to show", 10, 100, 30, step=10)

if hazard_text:
    # TF‑IDF similarity search
    q_vec = vectorizer.transform([hazard_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()
    order = sims.argsort()[::-1]
    # top-N unique descriptions
    rows, seen = [], set()
    for i in order:
        desc = df.iloc[i]["description"]
        if desc in seen:
            continue
        seen.add(desc)
        rows.append((desc, float(sims[i])))
        if len(rows) >= topn:
            break
    res_df = pd.DataFrame(rows, columns=["description", "similarity"])

    st.subheader("Similar incidents")
    st.markdown('<span class="small-note">List ordered by similarity (highest first). Select one to get recommendations.</span>', unsafe_allow_html=True)

    # Selectbox (dropdown with scroll)
    choice = st.selectbox("Select an incident for analysis:", res_df["description"].tolist())

    if choice:
        pred_raw = model.predict([choice])[0]
        label = fix_label(choice, pred_raw)

        # Friendly blocks with emojis
        st.markdown(f'<div class="block green">✅ <b>Predicted Event Title:</b> {label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="block blue">🧤 <b>Recommended PPE:</b> {", ".join(ppe_map.get(label, ["N/A"]))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="block blue">🎓 <b>Training Recommendations:</b> {", ".join(training_map.get(label, ["N/A"]))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="block blue">🧩 <b>Possible Root Causes:</b> {", ".join(root_cause_map.get(label, ["N/A"]))}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="block yellow">⏱️ <b>OSHA Reporting Time:</b> {osha_time_map.get(label, "24 hours")}</div>', unsafe_allow_html=True)

# ===============================
# Footer (credits)
# ===============================
st.markdown(
    '<div class="footer">© 2025 Zalemo Corporation — Built by <b>Laura Perea</b>. '
    'Internal demo for safety support. Not a substitute for professional judgment or legal advice.</div>',
    unsafe_allow_html=True
)
