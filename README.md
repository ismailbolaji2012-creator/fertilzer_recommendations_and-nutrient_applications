# FertilizerAI — Streamlit App

ML-powered crop fertilizer recommendation system.

## Folder structure

```
fertilizer-streamlit/
├── app.py                  ← Streamlit frontend
├── fertilizer.db           ← SQLite database (prediction history)
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── ml/
    ├── predict.py          ← ML prediction engine
    ├── model.pkl           ← Trained RandomForest model
    ├── encoder.pkl         ← Label encoder
    ├── train_model.py      ← Retrain script (optional)
    └── dataset.csv         ← Training dataset (8103 rows)
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## (Optional) Retrain the model

```bash
cd ml
python train_model.py
```

This regenerates `model.pkl` and `encoder.pkl` from `dataset.csv`.

## Features

- **Predict** — Enter soil sample data (N, P, K, pH, organic carbon, climate, crop, soil type) and get a full ML-driven fertilizer recommendation
- **History** — Table of all past predictions stored in SQLite
- **Statistics** — Charts showing fertilizer and crop distribution across all predictions
# fertilzer_recommendations_and-nutrient_applications
# fertilzer_recommendations_and-nutrient_applications
# fertilzer_recommendations_and-nutrient_applications
