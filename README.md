# Zalemo OSHA Incident Classifier (Streamlit)

Este proyecto forma parte del **Trabajo Fin de Máster en Big Data, Data Science e Inteligencia Artificial (UCM, 2025)**.  
Se desarrolló un modelo de aprendizaje automático capaz de **clasificar descripciones textuales de accidentes laborales en 10 categorías** a partir de los *OSHA Severe Injury Reports (2015–2024)*.  

El modelo final —**Linear SVM calibrado con Platt scaling**— alcanzó un rendimiento de **84,2% de exactitud** y un **F1-macro de 0,77**, ofreciendo además curvas de calibración con un **Brier score macro de 0,0249**.

---

## 🚀 Demo Online
La aplicación está disponible en Streamlit:  
👉 [https://zalemo-osha-app-m8rg36dxfxvxwjjv593b5j.streamlit.app](https://zalemo-osha-app-m8rg36dxfxvxwjjv593b5j.streamlit.app)

---

## ⚙️ Ejecutar en local
```bash
pip install -r requirements.txt
streamlit run app.py
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

> **Note:** The joblib files are expected to be full sklearn `Pipeline`s that already include the TF-IDF vectorizer.

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repo.
2. On https://streamlit.io/cloud → "New app" → connect your repo, select `app.py`.
3. Set Python version (e.g., 3.10/3.11) and deploy.

---

© Zalemo — academic demo.
