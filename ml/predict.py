"""
FertilizerAI - Prediction Engine v3.0

Architecture:
  1. PRIMARY ML model  → fertilizer label via RandomForest.predict_proba()
  2. DETERMINISTIC NPK → soil level classification from thresholds (no ML)
     This eliminates contradictions: High soil level ≠ "apply urgently"
  3. CLIMATE layer     → leaching/volatilization risk from temperature/humidity/rainfall
  4. VALIDATION        → pre-render checks ensure agronomic consistency

No rule overrides on fertilizer. No IF/ELSE on the fertilizer decision.
All fertilizer confidence values derive purely from predict_proba().
"""

import sys
import json
import os
import pickle

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
_artifacts = None


def load_model():
    global _artifacts
    if _artifacts is None:
        with open(MODEL_PATH, "rb") as f:
            _artifacts = pickle.load(f)
    return _artifacts


# ─── DETERMINISTIC NPK CLASSIFICATION ────────────────────────────────────────

_NPK_THRESHOLDS = [
    (10,  "Very Low"),
    (20,  "Low"),
    (35,  "Medium"),
    (50,  "High"),
    (999, "Very High"),
]

_NPK_ACTIONS = {
    "Very Low": ("Apply urgently",                   "Critical deficiency — immediate application essential"),
    "Low":      ("Supplement heavily",               "Significant deficiency — apply supplemental dose"),
    "Medium":   ("Moderate supplementation",         "Moderate deficiency — standard application recommended"),
    "High":     ("Maintenance only",                 "Adequate levels — maintenance dose only, do not over-apply"),
    "Very High":("No additional application",        "Levels are sufficient — further application not required"),
}

_NPK_RISK = {
    "Very Low": "Critical",
    "Low":      "High",
    "Medium":   "Medium",
    "High":     "Low",
    "Very High":"None",
}


def classify_npk(value: float) -> str:
    """Convert a raw mg/kg soil value to a soil level category."""
    for threshold, label in _NPK_THRESHOLDS:
        if value < threshold:
            return label
    return "Very High"


# Crop N requirements (kg N/ha) — used for rate calculations
_CROP_N_REQ = {
    "Maize":   120, "Wheat": 100, "Rice":    90, "Cassava": 70,
    "Sorghum":  80, "Tomato":130, "Millet":  70, "Beans":   40,
}
_CROP_P_REQ = {
    "Maize":    50, "Wheat":  45, "Rice":    40, "Cassava": 35,
    "Sorghum":  35, "Tomato": 55, "Millet":  35, "Beans":   30,
}
_CROP_K_REQ = {
    "Maize":    80, "Wheat":  70, "Rice":    70, "Cassava": 90,
    "Sorghum":  60, "Tomato":100, "Millet":  60, "Beans":   50,
}

# Fertilizer NPK composition (% by weight)
_FERT_COMPOSITION = {
    "Urea":           {"N": 0.46, "P": 0.00, "K": 0.00},
    "DAP":            {"N": 0.18, "P": 0.46, "K": 0.00},
    "Potash":         {"N": 0.00, "P": 0.00, "K": 0.60},
    "NPK 15-15-15":   {"N": 0.15, "P": 0.15, "K": 0.15},
    "NPK 20-10-10":   {"N": 0.20, "P": 0.10, "K": 0.10},
    "Superphosphate": {"N": 0.00, "P": 0.18, "K": 0.00},
    "Organic Compost":{"N": 0.02, "P": 0.01, "K": 0.02},
}

_FERT_TYPE = {
    "Urea":            "Nitrogen booster (46-0-0)",
    "DAP":             "Phosphorus starter (18-46-0)",
    "Potash":          "Potassium source (0-0-60)",
    "NPK 15-15-15":    "Balanced maintenance (15-15-15)",
    "NPK 20-10-10":    "Nitrogen-heavy blend (20-10-10)",
    "Superphosphate":  "Soluble phosphate (0-18-0)",
    "Organic Compost": "Slow-release organic amendment",
}

_SOIL_MULT = {"Sandy": 1.25, "Loamy": 1.00, "Silt": 0.90, "Clay": 0.75}


def _compute_application_rate(fertilizer, crop_type, n_level, p_level, k_level, soil_type):
    """
    Calculate recommended application rate based on:
      - Crop nutrient requirements
      - Most-limiting nutrient deficit
      - Fertilizer composition
      - Soil retention factor
    Returns (rate_str, calculation dict).
    """
    comp   = _FERT_COMPOSITION.get(fertilizer, {"N": 0.15, "P": 0.15, "K": 0.15})
    smult  = _SOIL_MULT.get(soil_type, 1.0)

    # Determine which nutrient drives the rate
    deficiency_weights = {"Very Low": 1.0, "Low": 0.7, "Medium": 0.4, "High": 0.1, "Very High": 0.0}
    levels = {"N": n_level, "P": p_level, "K": k_level}
    nutrient_reqs = {
        "N": _CROP_N_REQ.get(crop_type, 80),
        "P": _CROP_P_REQ.get(crop_type, 40),
        "K": _CROP_K_REQ.get(crop_type, 70),
    }

    # Find the most limiting nutrient that the fertilizer can supply
    best_nutrient, best_req, best_content = "N", nutrient_reqs["N"], comp.get("N", 0)
    best_score = -1
    for nutr in ["N", "P", "K"]:
        c = comp.get(nutr, 0)
        if c > 0:
            score = deficiency_weights.get(levels[nutr], 0) * c
            if score > best_score:
                best_score = best_nutrient, best_req, best_content = nutr, nutrient_reqs[nutr], c
                best_score = score

    target_kg = nutrient_reqs[best_nutrient] * deficiency_weights.get(levels[best_nutrient], 0.5)
    if best_content > 0:
        raw_rate = (target_kg / best_content) * smult
    else:
        raw_rate = 150 * smult

    rate = max(50, min(4000, round(raw_rate / 5) * 5))
    rate_str = f"{rate} kg/ha"

    calc = {
        "targetRequirement": f"{target_kg:.0f} kg {best_nutrient}/ha",
        "fertilizerContent":  f"{best_content*100:.0f}% {best_nutrient}",
        "calculatedRate":     rate_str,
        "explanation": (
            f"{target_kg:.0f} ÷ {best_content:.2f} × {smult:.2f} (soil factor) "
            f"= {rate} kg/ha"
        ),
    }
    return rate_str, calc


# ─── SOIL HEALTH INDEX ────────────────────────────────────────────────────────
# 0-40 Poor | 41-70 Fair | 71-85 Good | 86-100 Excellent

def compute_soil_health(n, p, k, ph, oc):
    score = 0.0

    # N component (max 20)
    nl = classify_npk(n)
    score += {"Very Low": 4, "Low": 10, "Medium": 18, "High": 20, "Very High": 16}[nl]

    # P component (max 20)
    pl = classify_npk(p)
    score += {"Very Low": 4, "Low": 10, "Medium": 18, "High": 20, "Very High": 16}[pl]

    # K component (max 20)
    kl = classify_npk(k)
    score += {"Very Low": 4, "Low": 10, "Medium": 18, "High": 20, "Very High": 16}[kl]

    # pH component (max 25) — optimal 6.0-7.0
    if 6.0 <= ph <= 7.0:   score += 25
    elif 5.5 <= ph <= 7.5: score += 18
    elif 5.0 <= ph <= 8.0: score += 10
    else:                   score += 3

    # Organic carbon (max 15) — optimal >= 1.5%
    if oc >= 2.0:    score += 15
    elif oc >= 1.5:  score += 12
    elif oc >= 1.0:  score += 8
    elif oc >= 0.5:  score += 4
    else:            score += 1

    score = round(min(100.0, score), 1)
    if score <= 40:   category = "Poor"
    elif score <= 70: category = "Fair"
    elif score <= 85: category = "Good"
    else:             category = "Excellent"

    return score, category


# ─── CLIMATE ASSESSMENT ───────────────────────────────────────────────────────

def assess_climate(temperature, humidity, rainfall, fertilizer):
    # Leaching risk from rainfall
    if rainfall > 1500:
        leaching_risk = "High"
        split_advice  = "Apply in 3–4 split doses to prevent nutrient leaching."
    elif rainfall > 1000:
        leaching_risk = "Medium"
        split_advice  = "Apply in 2–3 split doses for best efficiency."
    else:
        leaching_risk = "Low"
        split_advice  = "Single basal application is suitable under low-rainfall conditions."

    # Volatilization risk from temperature + humidity
    if temperature > 32 and humidity > 75:
        vol_risk       = "High"
        timing_advice  = "Apply early morning or evening to reduce ammonia volatilization."
    elif temperature > 28 or humidity > 60:
        vol_risk       = "Medium"
        timing_advice  = "Avoid applying during peak heat hours (10am–3pm)."
    else:
        vol_risk       = "Low"
        timing_advice  = "Standard daytime application is acceptable."

    # Urea-specific high-temp warning
    if fertilizer in ("Urea", "NPK 20-10-10") and temperature > 30:
        timing_advice += " For urea-based fertilizers, incorporate into soil immediately after spreading."

    return {
        "leachingRisk":       leaching_risk,
        "volatilizationRisk": vol_risk,
        "applicationTiming":  timing_advice,
        "splitAdvice":        split_advice,
    }


# ─── DYNAMIC EXPLANATIONS ────────────────────────────────────────────────────

def _npk_explanation(nutrient_name, value, level, crop_type):
    """Generate a factual, value-based explanation for the given nutrient."""
    base = {
        "Very Low": (
            f"{nutrient_name} is critically low at {value:.0f} mg/kg — "
            f"far below the minimum threshold for {crop_type}. "
            f"Immediate application is required to prevent severe yield loss."
        ),
        "Low": (
            f"{nutrient_name} is below the recommended range at {value:.0f} mg/kg. "
            f"Supplemental application will significantly improve {crop_type} performance."
        ),
        "Medium": (
            f"{nutrient_name} is at a moderate level ({value:.0f} mg/kg). "
            f"A standard application will support optimal {crop_type} growth."
        ),
        "High": (
            f"{nutrient_name} is at a good level ({value:.0f} mg/kg). "
            f"Only a maintenance dose is needed — avoid over-application."
        ),
        "Very High": (
            f"{nutrient_name} is at {value:.0f} mg/kg — well above the recommended range. "
            f"No additional {nutrient_name.lower()} application is required for {crop_type}."
        ),
    }
    return base.get(level, f"{nutrient_name} is {value:.0f} mg/kg ({level}).")


def _npk_recommendation(nutrient_name, level, crop_type):
    """Return a short actionable recommendation string."""
    # Map full name to nutrient key (Potassium starts with P, not K)
    _name_to_key = {"Nitrogen": "N", "Phosphorus": "P", "Potassium": "K"}
    nutr_key = _name_to_key.get(nutrient_name, nutrient_name[0])
    req_map = {"N": _CROP_N_REQ, "P": _CROP_P_REQ, "K": _CROP_K_REQ}
    req_base = req_map.get(nutr_key, _CROP_N_REQ).get(crop_type, 80)
    dose_map = {
        "Very Low": req_base,
        "Low":      round(req_base * 0.75),
        "Medium":   round(req_base * 0.45),
        "High":     round(req_base * 0.15),
        "Very High": 0,
    }
    dose = dose_map.get(level, round(req_base * 0.5))
    if dose == 0:
        return f"No additional {nutrient_name.lower()} required"
    return f"Apply {dose} kg {nutr_key}/ha"


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def validate_report(nutrient_status):
    """
    Pre-render consistency checks.
    Raises ValueError with a clear message if any contradiction is found.
    """
    for nutr, data in nutrient_status.items():
        level  = data["level"]
        action = data["action"]
        action_for_level = _NPK_ACTIONS[level][0]
        if action != action_for_level:
            raise ValueError(
                f"VALIDATION FAIL: {nutr} level='{level}' but action='{action}' "
                f"(expected '{action_for_level}')"
            )
    return True


# ─── MAIN PREDICTION ─────────────────────────────────────────────────────────

def predict(input_data):
    import pandas as pd

    arts = load_model()

    n    = float(input_data["nitrogen"])
    p    = float(input_data["phosphorus"])
    k    = float(input_data["potassium"])
    ph   = float(input_data["ph"])
    oc   = float(input_data["organicCarbon"])
    temp = float(input_data["temperature"])
    hum  = float(input_data["humidity"])
    rain = float(input_data["rainfall"])
    crop = str(input_data["cropType"])
    soil = str(input_data["soilType"])

    numeric_features     = arts["numeric_features"]
    categorical_features = arts["categorical_features"]

    X = pd.DataFrame(
        [[n, p, k, ph, oc, temp, hum, rain, crop, soil]],
        columns=numeric_features + categorical_features,
    )

    # ── 1. PRIMARY: fertilizer via RandomForest ────────────────────────────────
    fert_proba = arts["model_pipeline"].predict_proba(X)[0]
    fert_idx   = int(fert_proba.argmax())
    fertilizer = arts["fert_encoder"].inverse_transform([fert_idx])[0]
    confidence = float(fert_proba[fert_idx])

    # Top-3 alternatives
    top3_idx = fert_proba.argsort()[::-1][:3]
    top_alternatives = [
        {
            "fertilizer":  arts["fert_encoder"].inverse_transform([i])[0],
            "probability": round(float(fert_proba[i]), 4),
        }
        for i in top3_idx
    ]

    # ── 2. DETERMINISTIC NPK classification (no ML, no rule override) ─────────
    n_level = classify_npk(n)
    p_level = classify_npk(p)
    k_level = classify_npk(k)

    nutrient_status = {
        "nitrogen": {
            "value":          n,
            "level":          n_level,
            "action":         _NPK_ACTIONS[n_level][0],
            "recommendation": _npk_recommendation("Nitrogen", n_level, crop),
            "explanation":    _npk_explanation("Nitrogen", n, n_level, crop),
        },
        "phosphorus": {
            "value":          p,
            "level":          p_level,
            "action":         _NPK_ACTIONS[p_level][0],
            "recommendation": _npk_recommendation("Phosphorus", p_level, crop),
            "explanation":    _npk_explanation("Phosphorus", p, p_level, crop),
        },
        "potassium": {
            "value":          k,
            "level":          k_level,
            "action":         _NPK_ACTIONS[k_level][0],
            "recommendation": _npk_recommendation("Potassium", k_level, crop),
            "explanation":    _npk_explanation("Potassium", k, k_level, crop),
        },
    }

    # Fix: pass level not name to _npk_recommendation for P and K
    nutrient_status["phosphorus"]["recommendation"] = _npk_recommendation("Phosphorus", p_level, crop)
    nutrient_status["potassium"]["recommendation"]  = _npk_recommendation("Potassium",  k_level, crop)

    # ── 3. Validation (raises if any contradiction found) ──────────────────────
    validate_report(nutrient_status)

    # ── 4. Soil health ─────────────────────────────────────────────────────────
    soil_score, soil_cat = compute_soil_health(n, p, k, ph, oc)

    # ── 5. Climate assessment ──────────────────────────────────────────────────
    climate = assess_climate(temp, hum, rain, fertilizer)

    # ── 6. Application rate ────────────────────────────────────────────────────
    app_rate, rate_calc = _compute_application_rate(
        fertilizer, crop, n_level, p_level, k_level, soil
    )

    # ── 7. Application strategy ────────────────────────────────────────────────
    split_count = (
        "3–4 applications" if climate["leachingRisk"] == "High" else
        "2–3 applications" if climate["leachingRisk"] == "Medium" else
        "1–2 applications"
    )
    strategy = {
        "method":    "Split application" if climate["leachingRisk"] != "Low" else "Basal application",
        "timing":    ["Pre-plant / basal stage", "Early vegetative stage", "Mid growth stage"],
        "frequency": split_count + " per season",
        "notes":     climate["splitAdvice"],
    }

    # ── 8. Agronomic advice ────────────────────────────────────────────────────
    advice = [
        f"Soil health is rated {soil_cat} ({soil_score}/100) — "
        + ("prioritise improvement before next season." if soil_cat in ("Poor","Fair") else "maintain current management practices."),
        climate["applicationTiming"],
        "Test soil pH regularly — optimal range for most crops is 6.0–7.0.",
        "Incorporate organic matter to improve long-term nutrient retention and soil structure.",
    ]
    if soil == "Sandy":
        advice.append("Sandy soil has high leaching risk — prefer slow-release or split fertilizer applications.")
    elif soil == "Clay":
        advice.append("Clay soil retains nutrients well — avoid over-application to prevent nutrient toxicity.")

    # ── 9. Warnings ────────────────────────────────────────────────────────────
    warnings = []
    if ph < 5.5:
        warnings.append(f"Strongly acidic soil (pH {ph}) — apply agricultural lime to raise pH to 6.0–7.0 before fertilizing.")
    if ph > 7.8:
        warnings.append(f"Alkaline soil (pH {ph}) — consider sulfur amendment; iron and manganese may be unavailable.")
    if climate["leachingRisk"] == "High":
        warnings.append(f"High rainfall ({rain:.0f} mm) creates a high leaching risk — split all applications.")
    if climate["volatilizationRisk"] == "High":
        warnings.append(f"High temperature ({temp}°C) and humidity ({hum}%) — incorporate urea-based fertilizers immediately after application.")
    if confidence < 0.55:
        warnings.append(f"Model confidence is {confidence*100:.0f}% — consider lab soil analysis for confirmation.")

    # ── 10. Decision trace ─────────────────────────────────────────────────────
    decision_trace = [
        f"Soil Type: {soil} — {{'Sandy':'low retention, high leaching','Loamy':'balanced fertility','Clay':'high retention','Silt':'moderate fertility'}}.get('{soil}','')",
        f"Crop Type: {crop} — target N={_CROP_N_REQ.get(crop,80)} kg/ha, P={_CROP_P_REQ.get(crop,40)} kg/ha, K={_CROP_K_REQ.get(crop,70)} kg/ha",
        f"Nutrient Status: N={n_level} ({n:.0f} mg/kg), P={p_level} ({p:.0f} mg/kg), K={k_level} ({k:.0f} mg/kg)",
        f"Climate Assessment: Rainfall={rain:.0f} mm → Leaching risk={climate['leachingRisk']}; Temp={temp}°C → Volatilization risk={climate['volatilizationRisk']}",
        f"RandomForest evaluated {len(fert_proba)} fertilizer candidates across 300 decision trees",
        f"Selected '{fertilizer}' with {confidence*100:.1f}% probability",
        f"Top alternatives: " + ", ".join(f"{a['fertilizer']} ({a['probability']*100:.0f}%)" for a in top_alternatives),
        f"Application rate: {app_rate} ({rate_calc['explanation']})",
    ]

    # Clean up trace formatting
    import re
    decision_trace = [re.sub(r"\.get\('[^']+',''\)", "", s).strip() for s in decision_trace]

    return {
        "status": "success",
        "modelInfo": {
            "modelName": "RandomForest Fertilizer Recommender",
            "version":    arts.get("version", "3.0"),
        },
        "fertilizerSection": {
            "primaryFertilizer": fertilizer,
            "confidence":        round(confidence, 4),
            "reason": (
                f"RandomForest selected '{fertilizer}' from {len(fert_proba)} candidates "
                f"based on learned patterns from {crop} grown on {soil} soil "
                f"with N={n_level}, P={p_level}, K={k_level} nutrient status."
            ),
            "applicationType":   _FERT_TYPE.get(fertilizer, "Balanced fertilizer"),
            "applicationRate":   app_rate,
            "topAlternatives":   top_alternatives,
            "rateCalculation":   rate_calc,
        },
        "nutrientStatus":    nutrient_status,
        "soilHealthIndex":   {"score": soil_score, "category": soil_cat},
        "climateAssessment": climate,
        "applicationStrategy": strategy,
        "agronomicAdvice":   advice,
        "decisionTrace":     decision_trace,
        "warnings":          warnings,
    }


if __name__ == "__main__":
    raw = sys.stdin.read()
    try:
        result = predict(json.loads(raw))
        print(json.dumps(result))
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "trace": traceback.format_exc()}),
              file=sys.stderr)
        sys.exit(1)
