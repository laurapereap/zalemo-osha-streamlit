# Zalemo OSHA Incident Classifier (Streamlit)

Streamlit app for Laura's TFM to predict the **family** of an OSHA incident from the free-text description.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
.
├── app.py
├── requirements.txt
├── zalemo_logo.png             # optional (branding)
├── salidas_osha/               # model artifacts go here
│   ├── modelo_family_tfidf_logreg.joblib
│   ├── modelo_family_tfidf_linearSVM.joblib
│   ├── metricas_comparacion.csv
│   ├── resumen_metricas.txt
│   ├── matriz_confusion_svm_cal_test.png
│   ├── comparacion_f1_test.png
│   └── top_terms_per_class.json
└── sample_data/                # optional small sample, not used in app
```

> **Note:** The joblib files are expected to be full sklearn `Pipeline`s that already include the TF-IDF vectorizer. If not, re-save as a single pipeline.

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repo.
2. On https://streamlit.io/cloud → "New app" → connect your repo, select `app.py`.
3. Set Python version (e.g., 3.10/3.11) and deploy.

---

© Zalemo — academic demo.
