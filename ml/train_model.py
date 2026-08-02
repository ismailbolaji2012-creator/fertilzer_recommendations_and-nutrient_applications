"""
FertilizerAI - Training Script v3.0

Feature set: N, P, K, pH, Organic_Carbon, Temperature, Humidity, Rainfall,
             Crop (OHE), Soil_Type (OHE)

Label noise: ~10% random fertilizer reassignment so the RF generalises rather
than memorising a lookup table.

Predict.py contains NO rule overrides — all fertilizer decisions come from
RandomForest.predict_proba().  NPK status classification is deterministic
threshold-based (not ML) and fully independent of the fertilizer model.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

FERTILIZERS = [
    "NPK 15-15-15", "NPK 20-10-10", "Urea",
    "DAP", "Potash", "Superphosphate", "Organic Compost",
]
CROPS      = ["Wheat","Rice","Maize","Cassava","Sorghum","Tomato","Millet","Beans"]
SOIL_TYPES = ["Sandy","Loamy","Clay","Silt"]

# ─── Ground-truth seed rows ────────────────────────────────────────────────────
# (N, P, K, pH, OC, Temp, Humidity, Rainfall, Crop, SoilType, Fertilizer)

SEED_ROWS = [
    (43,39,54,5.12,1.42,28,72,1100,"Wheat",   "Loamy","Organic Compost"),
    (56,59,55,6.78,0.52,25,68,1400,"Wheat",   "Clay", "Organic Compost"),
    (33,44,23,5.33,0.99,27,75,1200,"Rice",    "Loamy","NPK 20-10-10"),
    (19,56,25,4.85,1.95,30,82,1600,"Cassava", "Sandy","NPK 15-15-15"),
    (47,20, 9,5.35,2.48,29,70,950, "Maize",   "Loamy","DAP"),
    (12,17,22,7.03,0.70,26,65,800, "Sorghum", "Silt", "NPK 20-10-10"),
    (25,54,32,7.49,1.30,31,60,700, "Tomato",  "Loamy","NPK 15-15-15"),
    (43,46,46,7.41,2.10,24,74,1300,"Tomato",  "Clay", "Superphosphate"),
    (23,34,26,5.89,0.91,28,78,1500,"Maize",   "Loamy","NPK 20-10-10"),
    (27,23,25,6.84,1.61,27,72,1100,"Wheat",   "Loamy","Potash"),
    (15,21,10,5.22,1.97,32,85,1800,"Beans",   "Sandy","NPK 20-10-10"),
    (15,23, 5,5.53,1.73,26,79,1400,"Rice",    "Silt", "Organic Compost"),
    (28,32, 9,7.64,0.88,25,64,900, "Wheat",   "Loamy","NPK 15-15-15"),
    (57,59,45,4.55,1.21,33,88,1900,"Tomato",  "Sandy","DAP"),
    (40,30,16,4.80,2.07,29,80,1600,"Millet",  "Loamy","DAP"),
    (44,41,30,5.23,1.61,27,73,1200,"Beans",   "Clay", "NPK 15-15-15"),
    (28,30,50,4.59,0.51,34,90,2000,"Tomato",  "Sandy","Organic Compost"),
    ( 7,57,38,5.14,2.02,28,76,1300,"Millet",  "Loamy","Urea"),
    (26,27,53,6.54,0.57,26,68,1000,"Wheat",   "Silt", "NPK 20-10-10"),
    (57,13,18,5.97,1.99,31,83,1700,"Maize",   "Sandy","Organic Compost"),
    ( 6,16,30,7.62,0.90,27,71,1100,"Maize",   "Loamy","Potash"),
    (28,57,49,7.36,2.42,24,75,1350,"Beans",   "Clay", "Superphosphate"),
    (48, 5,31,5.70,1.24,30,80,1500,"Tomato",  "Loamy","NPK 15-15-15"),
    (34, 5,13,5.41,1.15,33,86,1850,"Cassava", "Sandy","NPK 20-10-10"),
    (42,51,30,5.83,0.80,28,78,1450,"Millet",  "Loamy","DAP"),
    ( 6,38,51,6.57,1.11,26,67,980, "Rice",    "Silt", "Superphosphate"),
    (25,36,26,5.44,2.25,27,74,1250,"Tomato",  "Loamy","Urea"),
    (37,58,51,6.68,2.49,24,72,1300,"Rice",    "Clay", "Potash"),
    (16,52,34,5.93,1.24,30,82,1650,"Tomato",  "Sandy","DAP"),
    (26,29,47,6.43,1.40,28,77,1400,"Cassava", "Loamy","NPK 15-15-15"),
    (48,44,52,6.03,1.94,25,70,1200,"Cassava", "Clay", "NPK 20-10-10"),
    (29,49,21,5.53,2.27,27,75,1300,"Beans",   "Loamy","NPK 15-15-15"),
    (53,57,30,7.82,1.69,24,63,850, "Beans",   "Silt", "Superphosphate"),
    (31, 5,40,7.17,1.28,32,84,1750,"Cassava", "Sandy","NPK 20-10-10"),
    (46,20, 5,4.99,1.33,29,79,1450,"Sorghum", "Loamy","Organic Compost"),
    (32,43,12,7.54,1.89,25,71,1100,"Cassava", "Clay", "Urea"),
    (20, 9,53,6.21,0.51,27,76,1350,"Maize",   "Silt", "Organic Compost"),
    (19,26,39,7.63,1.74,26,73,1250,"Maize",   "Loamy","Organic Compost"),
    (51,33,56,7.30,1.21,24,68,1050,"Sorghum", "Clay", "NPK 15-15-15"),
    (55,59,19,5.99,2.09,29,80,1550,"Wheat",   "Loamy","Superphosphate"),
    (48, 7,51,4.58,0.69,34,90,2000,"Millet",  "Sandy","Urea"),
    (59,16,26,5.44,1.68,27,73,1200,"Wheat",   "Loamy","NPK 20-10-10"),
    (56,30,18,6.40,1.46,26,71,1100,"Rice",    "Silt", "NPK 20-10-10"),
    ( 7,20,30,6.72,1.78,28,78,1450,"Wheat",   "Loamy","Potash"),
    (41,55,32,5.40,0.63,31,85,1800,"Tomato",  "Clay", "DAP"),
    (55,41,27,4.99,1.66,33,88,1900,"Wheat",   "Sandy","Potash"),
    (11,26,18,7.42,1.62,26,72,1150,"Tomato",  "Loamy","Potash"),
    (25,33,28,7.95,1.62,25,65,900, "Sorghum", "Silt", "NPK 20-10-10"),
    (13,18, 6,6.34,1.71,32,86,1800,"Maize",   "Sandy","NPK 15-15-15"),
    (43,32,49,5.10,1.85,28,77,1400,"Beans",   "Loamy","Organic Compost"),
    (22, 9,30,5.45,2.11,27,74,1250,"Tomato",  "Silt", "Superphosphate"),
    ( 8,51,18,4.56,1.04,31,84,1750,"Maize",   "Clay", "Organic Compost"),
    (29,53,55,7.70,2.15,24,70,1150,"Millet",  "Loamy","Organic Compost"),
    (18,34,11,4.91,1.50,34,91,2000,"Rice",    "Sandy","Urea"),
    (54,50, 7,6.52,0.65,25,67,950, "Millet",  "Clay", "NPK 20-10-10"),
    (13,56,51,5.46,0.62,30,82,1650,"Sorghum", "Loamy","Superphosphate"),
    (30, 9,27,6.44,1.17,27,73,1200,"Wheat",   "Silt", "NPK 15-15-15"),
    (57,16,50,6.78,2.07,28,78,1450,"Wheat",   "Loamy","Urea"),
    ( 6,20,47,7.40,1.92,31,83,1700,"Cassava", "Sandy","DAP"),
    (24,30,51,5.22,2.08,29,80,1550,"Cassava", "Loamy","NPK 15-15-15"),
    (32,30,49,4.54,1.53,26,72,1150,"Cassava", "Clay", "Potash"),
    (51,52,22,4.98,1.38,33,87,1900,"Millet",  "Sandy","NPK 15-15-15"),
    (11,25,42,7.65,0.79,27,74,1300,"Tomato",  "Loamy","NPK 15-15-15"),
    (48,43,39,7.56,1.16,25,68,1050,"Rice",    "Silt", "Organic Compost"),
    (12,40,55,6.59,1.37,29,80,1550,"Cassava", "Loamy","NPK 15-15-15"),
    (51,37,19,6.60,0.68,24,71,1150,"Sorghum", "Clay", "Potash"),
    (39,34,29,6.83,0.94,27,76,1350,"Sorghum", "Loamy","Potash"),
    (18,41,59,5.11,1.70,32,85,1800,"Cassava", "Sandy","NPK 20-10-10"),
    (21,27,41,7.70,1.97,26,72,1200,"Wheat",   "Loamy","NPK 15-15-15"),
    (40,14,32,5.97,2.50,28,77,1400,"Cassava", "Silt", "Organic Compost"),
    (54,58,14,5.84,2.37,24,70,1150,"Rice",    "Clay", "NPK 15-15-15"),
    (44, 9,43,6.32,1.79,29,80,1500,"Cassava", "Loamy","DAP"),
    ( 8,40,21,4.66,1.34,33,88,1900,"Rice",    "Sandy","DAP"),
    ( 6,38,43,5.08,1.77,28,78,1450,"Sorghum", "Loamy","Organic Compost"),
    (10,56,26,7.08,2.07,25,70,1200,"Cassava", "Clay", "NPK 20-10-10"),
    (58,35,30,4.79,0.74,34,90,2000,"Millet",  "Sandy","NPK 20-10-10"),
    (46,14,48,6.61,1.32,27,74,1300,"Beans",   "Loamy","Potash"),
    ( 8,23,29,5.36,2.18,26,71,1100,"Millet",  "Silt", "Potash"),
    (58,36,21,5.86,1.27,28,78,1450,"Sorghum", "Loamy","Urea"),
    (33, 5,17,5.51,1.64,32,86,1800,"Rice",    "Sandy","Organic Compost"),
    (22, 9,24,5.74,1.68,27,74,1300,"Millet",  "Loamy","NPK 15-15-15"),
    (30,49,29,7.02,0.87,24,68,1050,"Rice",    "Clay", "Superphosphate"),
    (48, 8, 8,5.54,1.22,33,87,1900,"Millet",  "Sandy","Organic Compost"),
    (38,20,14,6.48,1.17,27,75,1350,"Millet",  "Loamy","Urea"),
    (14,28, 7,6.17,0.55,26,72,1200,"Rice",    "Silt", "Potash"),
    (40,20,45,6.82,0.55,28,78,1450,"Tomato",  "Loamy","Organic Compost"),
    (18,59,49,7.78,2.16,24,70,1150,"Millet",  "Clay", "Organic Compost"),
    (35, 6,22,7.06,1.05,29,80,1500,"Cassava", "Loamy","Potash"),
    (52,53,51,5.25,1.54,34,91,2000,"Beans",   "Sandy","Urea"),
    (19,32,40,4.61,1.10,27,74,1300,"Sorghum", "Silt", "Potash"),
    (12,36,51,5.42,2.38,28,78,1450,"Tomato",  "Loamy","NPK 20-10-10"),
    (18,31,26,6.58,1.02,25,70,1200,"Rice",    "Clay", "NPK 15-15-15"),
    (27,24,38,4.68,1.36,33,88,1900,"Tomato",  "Sandy","Potash"),
    (44,28,51,6.24,2.25,27,75,1350,"Maize",   "Loamy","Urea"),
    (25,16,12,6.59,2.18,26,72,1200,"Maize",   "Silt", "DAP"),
    (20,54,44,5.67,0.87,31,84,1750,"Millet",  "Clay", "NPK 20-10-10"),
    (49,39,53,7.20,2.11,34,90,2000,"Millet",  "Sandy","DAP"),
    (22,37,48,4.87,1.42,28,78,1450,"Cassava", "Loamy","Superphosphate"),
    (51,37,23,4.76,1.47,25,70,1200,"Beans",   "Clay", "Potash"),
    (57,55,46,7.05,0.77,32,85,1800,"Sorghum", "Sandy","NPK 20-10-10"),
    (28,47,45,6.23,0.66,27,74,1300,"Tomato",  "Loamy","NPK 15-15-15"),
    (30,41,41,6.91,1.96,26,72,1200,"Maize",   "Silt", "Potash"),
    (29,16,10,6.02,1.49,28,78,1450,"Millet",  "Loamy","NPK 15-15-15"),
]


def _label_fertilizer(n, p, k, ph, soil_type):
    """Label for data generation ONLY. Never called at prediction time."""
    ns = 2 if n < 20 else (1 if n < 40 else 0)
    ps = 2 if p < 20 else (1 if p < 40 else 0)
    ks = 2 if k < 20 else (1 if k < 40 else 0)
    total = ns + ps + ks
    sandy = soil_type == "Sandy"
    clay  = soil_type == "Clay"

    if ns == 2 and ps <= 1 and ks <= 1:
        return "NPK 20-10-10" if sandy else "Urea"
    if ps == 2 and ns <= 1 and ks <= 1:
        return "DAP"
    if ks == 2 and ns <= 1 and ps <= 1:
        return "Potash"
    if total == 6:
        return "NPK 15-15-15"
    if ns == 2 and ps == 2:
        return "NPK 20-10-10"
    if ns == 2 and ks == 2:
        return "NPK 20-10-10"
    if ps == 2 and ks == 2:
        return "Superphosphate"
    if clay and total <= 3:
        return "Organic Compost"
    if total == 0:
        return "Organic Compost"
    if ns == 1 and ps == 1:
        return "NPK 15-15-15"
    if ks == 1 and ns == 0:
        return "Potash"
    if sandy and total >= 2:
        return "NPK 20-10-10"
    if total >= 3:
        return "NPK 15-15-15"
    return "NPK 20-10-10"


def generate_synthetic_data(n_samples=8000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    noise_rate = 0.10

    for _ in range(n_samples):
        soil  = rng.choice(SOIL_TYPES)
        crop  = rng.choice(CROPS)

        if soil == "Sandy":
            n,p,k = float(rng.uniform(5,35)), float(rng.uniform(5,35)), float(rng.uniform(5,35))
        elif soil == "Clay":
            n,p,k = float(rng.uniform(20,62)), float(rng.uniform(20,62)), float(rng.uniform(20,62))
        elif soil == "Loamy":
            n,p,k = float(rng.uniform(12,55)), float(rng.uniform(12,55)), float(rng.uniform(12,55))
        else:  # Silt
            n,p,k = float(rng.uniform(8,50)), float(rng.uniform(8,50)), float(rng.uniform(8,50))

        ph   = round(float(rng.uniform(4.5, 8.2)), 2)
        oc   = round(float(rng.uniform(0.4, 2.6)), 2)
        temp = round(float(rng.uniform(18.0, 38.0)), 1)
        hum  = round(float(rng.uniform(40.0, 95.0)), 1)
        rain = round(float(rng.uniform(400.0, 2000.0)), 0)

        fertilizer = _label_fertilizer(n, p, k, ph, soil)
        if rng.random() < noise_rate:
            fertilizer = rng.choice(FERTILIZERS)

        rows.append((round(n,1), round(p,1), round(k,1),
                     ph, oc, temp, hum, rain,
                     crop, soil, fertilizer))

    return rows


def build_dataframe(rows):
    cols = ["N","P","K","pH","Organic_Carbon","Temperature","Humidity","Rainfall",
            "Crop","Soil_Type","Fertilizer"]
    return pd.DataFrame(rows, columns=cols)


def train(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_rows = list(SEED_ROWS) + generate_synthetic_data(8000)
    df       = build_dataframe(all_rows)

    print(f"Dataset size: {len(df)} rows")
    print(f"\nFertilizer distribution:\n{df['Fertilizer'].value_counts()}\n")

    numeric_cols     = ["N","P","K","pH","Organic_Carbon","Temperature","Humidity","Rainfall"]
    categorical_cols = ["Crop","Soil_Type"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    fert_encoder = LabelEncoder()
    y = fert_encoder.fit_transform(df["Fertilizer"])
    X = df[numeric_cols + categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=25, min_samples_split=3,
            min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    model_pipeline.fit(X_train, y_train)
    y_pred   = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Fertilizer model accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=fert_encoder.classes_, zero_division=0))

    rf         = model_pipeline.named_steps["classifier"]
    ohe_names  = (model_pipeline.named_steps["preprocessor"]
                  .named_transformers_["cat"]
                  .get_feature_names_out(categorical_cols).tolist())
    feat_names = numeric_cols + ohe_names
    importances = dict(zip(feat_names, rf.feature_importances_))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop-5 feature importances:")
    for f, i in top5:
        print(f"  {f}: {i:.4f}")

    artifacts = {
        "model_pipeline":        model_pipeline,
        "fert_encoder":          fert_encoder,
        "numeric_features":      numeric_cols,
        "categorical_features":  categorical_cols,
        "fertilizers":           fert_encoder.classes_.tolist(),
        "crops":                 sorted(df["Crop"].unique().tolist()),
        "soil_types":            sorted(df["Soil_Type"].unique().tolist()),
        "feature_importances":   importances,
        "model_accuracy":        accuracy,
        "version":               "3.0",
    }

    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    encoder_path = os.path.join(output_dir, "encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump({"fert_encoder": fert_encoder}, f)

    print(f"\nModel saved   → {model_path}")
    print(f"Encoder saved → {encoder_path}")
    return artifacts


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    train(out)
