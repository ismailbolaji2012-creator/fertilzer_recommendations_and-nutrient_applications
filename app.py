"""
FertilizerAI — Streamlit App (Standalone)
Run: streamlit run app.py
"""

import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ML_DIR   = BASE_DIR / "ml"
DB_PATH  = BASE_DIR / "fertilizer.db"

sys.path.insert(0, str(ML_DIR))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FertilizerAI — Crop Nutrient Advisor",
    page_icon="🌱",
    layout="wide",
)

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    return sqlite3.connect(str(DB_PATH))


def ensure_table():
    with get_conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_type         TEXT,
                soil_type         TEXT,
                nitrogen          REAL,
                phosphorus        REAL,
                potassium         REAL,
                ph                REAL,
                organic_carbon    REAL,
                temperature       REAL,
                humidity          REAL,
                rainfall          REAL,
                fertilizer        TEXT,
                n_needed          TEXT,
                p_needed          TEXT,
                k_needed          TEXT,
                confidence        REAL,
                soil_health_score REAL,
                created_at        TEXT
            )
        """)

ensure_table()


def save_prediction(inp, result):
    fert_sec   = result["fertilizerSection"]
    nutr       = result["nutrientStatus"]
    health     = result["soilHealthIndex"]
    with get_conn() as con:
        con.execute("""
            INSERT INTO predictions
              (crop_type, soil_type, nitrogen, phosphorus, potassium, ph,
               organic_carbon, temperature, humidity, rainfall,
               fertilizer, n_needed, p_needed, k_needed, confidence,
               soil_health_score, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            inp["cropType"], inp["soilType"],
            inp["nitrogen"], inp["phosphorus"], inp["potassium"],
            inp["ph"], inp["organicCarbon"],
            inp["temperature"], inp["humidity"], inp["rainfall"],
            fert_sec["primaryFertilizer"],
            nutr["nitrogen"]["level"],
            nutr["phosphorus"]["level"],
            nutr["potassium"]["level"],
            fert_sec["confidence"],
            health["score"],
            datetime.utcnow().isoformat(),
        ))


def load_history(limit=50):
    try:
        with get_conn() as con:
            cur = con.execute("""
                SELECT id, crop_type, soil_type, nitrogen, phosphorus, potassium,
                       ph, temperature, humidity, rainfall,
                       fertilizer, n_needed, p_needed, k_needed,
                       confidence, soil_health_score, created_at
                FROM predictions
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


def load_stats():
    try:
        with get_conn() as con:
            total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            if total == 0:
                return None
            avg_conf  = con.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0]
            top_fert  = con.execute("SELECT fertilizer, COUNT(*) c FROM predictions GROUP BY fertilizer ORDER BY c DESC LIMIT 1").fetchone()
            top_crop  = con.execute("SELECT crop_type, COUNT(*) c FROM predictions GROUP BY crop_type ORDER BY c DESC LIMIT 1").fetchone()
            fert_rows = con.execute("SELECT fertilizer, COUNT(*) c FROM predictions GROUP BY fertilizer ORDER BY c DESC").fetchall()
            crop_rows = con.execute("SELECT crop_type, COUNT(*) c FROM predictions GROUP BY crop_type ORDER BY c DESC").fetchall()
        return {
            "total": total,
            "avg_confidence": avg_conf or 0,
            "top_fertilizer": top_fert[0] if top_fert else "—",
            "top_crop":       top_crop[0] if top_crop else "—",
            "fertilizer_breakdown": [{"name": r[0], "count": r[1]} for r in fert_rows],
            "crop_breakdown":       [{"name": r[0], "count": r[1]} for r in crop_rows],
        }
    except Exception:
        return None


# ── ML helper ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    from predict import load_model as _load
    return _load()


def run_prediction(inp):
    from predict import predict
    return predict(inp)


# ── Level badge helper ────────────────────────────────────────────────────────

LEVEL_COLORS = {
    "Very Low":  "🔴",
    "Low":       "🟠",
    "Medium":    "🟡",
    "High":      "🟢",
    "Very High": "🔵",
}

def level_icon(level):
    return LEVEL_COLORS.get(level, "⚪") + " " + level


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🌱 FertilizerAI")
st.sidebar.caption("ML-powered Crop Nutrient Advisor")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Predict", "History", "Statistics"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.info(
    "Enter soil lab data and climate conditions to get an AI-powered fertilizer recommendation."
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — PREDICT
# ─────────────────────────────────────────────────────────────────────────────

if page == "Predict":
    st.title("Analysis & Prediction")
    st.caption("Enter soil sample data and climate conditions to generate a full ML-driven recommendation.")

    col_form, col_result = st.columns([1, 1.4], gap="large")

    with col_form:
        with st.form("soil_form"):
            st.subheader("🧪 Macronutrients (mg/kg)")
            c1, c2, c3 = st.columns(3)
            nitrogen      = c1.number_input("N", 0.0, 60.0, 20.0, 0.1)
            phosphorus    = c2.number_input("P", 0.0, 60.0, 20.0, 0.1)
            potassium     = c3.number_input("K", 0.0, 60.0, 20.0, 0.1)

            st.subheader("🔬 Physical Properties")
            ph             = st.number_input("pH Level", 4.0, 8.5, 6.5, 0.1)
            organic_carbon = st.number_input("Organic Carbon (%)", 0.1, 3.0, 1.5, 0.01)

            st.subheader("🌡️ Climate Conditions")
            temperature = st.number_input("Temperature (°C)", 10.0, 50.0, 28.0, 0.5)
            humidity    = st.number_input("Humidity (%)", 20.0, 100.0, 72.0, 1.0)
            rainfall    = st.number_input("Rainfall (mm/yr)", 100.0, 3000.0, 1100.0, 10.0)

            st.subheader("🌾 Field Context")
            crop_type = st.selectbox(
                "Target Crop",
                ["Maize", "Wheat", "Rice", "Cassava", "Sorghum", "Tomato", "Millet", "Beans"],
            )
            soil_type = st.selectbox(
                "Soil Type",
                ["Loamy", "Sandy", "Clay", "Silt"],
            )

            submitted = st.form_submit_button(
                "Generate Recommendation", use_container_width=True, type="primary"
            )

    with col_result:
        if submitted:
            inp = {
                "nitrogen":      nitrogen,
                "phosphorus":    phosphorus,
                "potassium":     potassium,
                "ph":            ph,
                "organicCarbon": organic_carbon,
                "temperature":   temperature,
                "humidity":      humidity,
                "rainfall":      rainfall,
                "cropType":      crop_type,
                "soilType":      soil_type,
            }

            with st.spinner("Running RandomForest model…"):
                try:
                    result = run_prediction(inp)
                    try:
                        save_prediction(inp, result)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Model error: {e}")
                    st.stop()

            fert     = result["fertilizerSection"]
            nutr     = result["nutrientStatus"]
            health   = result["soilHealthIndex"]
            climate  = result["climateAssessment"]
            strategy = result["applicationStrategy"]

            # ── Section 1 ────────────────────────────────────────────────────
            st.markdown("### Section 1 — Fertilizer Recommendation")
            conf_pct = fert["confidence"] * 100
            st.success(
                f"**Recommended Fertilizer: {fert['primaryFertilizer']}**  \n"
                f"{fert['applicationType']} — **{conf_pct:.1f}% confidence**  \n"
                f"{fert['reason']}"
            )
            col_rate, col_type = st.columns(2)
            col_rate.metric("Application Rate", fert["applicationRate"])
            col_type.metric("Fertilizer Type",  fert["applicationType"])

            with st.expander("Top Alternatives"):
                for i, alt in enumerate(fert["topAlternatives"]):
                    pct = alt["probability"] * 100
                    st.write(f"**{i+1}. {alt['fertilizer']}** — {pct:.0f}%")
                    st.progress(pct / 100)

            with st.expander("Application Rate Calculation"):
                rc = fert["rateCalculation"]
                cc1, cc2 = st.columns(2)
                cc1.metric("Target Requirement", rc["targetRequirement"])
                cc2.metric("Fertilizer Content",  rc["fertilizerContent"])
                st.info(f"**{rc['calculatedRate']}** — {rc['explanation']}")

            # ── Section 2 ────────────────────────────────────────────────────
            st.markdown("### Section 2 — Nutrient Application Requirements")
            for key, label in [("nitrogen","Nitrogen (N)"), ("phosphorus","Phosphorus (P)"), ("potassium","Potassium (K)")]:
                n = nutr[key]
                with st.container(border=True):
                    h1, h2 = st.columns([3, 1])
                    h1.write(f"**{label}** — {n['value']:.0f} mg/kg")
                    h2.write(level_icon(n["level"]))
                    st.write(f"**Action:** {n['action']}")
                    st.write(f"**Recommendation:** `{n['recommendation']}`")
                    st.caption(n["explanation"])

            # ── Health + confidence ───────────────────────────────────────────
            st.markdown("### Soil Health & Model Confidence")
            hc1, hc2 = st.columns(2)
            with hc1:
                h_score = health["score"]
                st.metric("Soil Health Index", f"{h_score}/100", delta=health["category"])
                st.progress(h_score / 100)
                st.caption("0–40 Poor · 41–70 Fair · 71–85 Good · 86–100 Excellent")
            with hc2:
                st.metric("Model Confidence", f"{conf_pct:.1f}%")
                st.progress(conf_pct / 100)
                mi = result["modelInfo"]
                st.caption(f"{mi['modelName']} v{mi['version']}")

            # ── Climate ───────────────────────────────────────────────────────
            st.markdown("### Climate Assessment")
            cl1, cl2 = st.columns(2)
            cl1.metric("Leaching Risk",       climate["leachingRisk"])
            cl2.metric("Volatilization Risk", climate["volatilizationRisk"])
            st.write(f"**Timing:** {climate['applicationTiming']}")
            st.write(f"**Split advice:** {climate['splitAdvice']}")

            # ── Application strategy ──────────────────────────────────────────
            st.markdown("### Application Strategy")
            sc1, sc2 = st.columns(2)
            sc1.metric("Method",    strategy["method"])
            sc2.metric("Frequency", strategy["frequency"])
            st.write("**Timing:** " + " → ".join(strategy["timing"]))
            st.caption(strategy["notes"])

            # ── Agronomic advice ──────────────────────────────────────────────
            if result.get("agronomicAdvice"):
                st.markdown("### Agronomic Advice")
                for tip in result["agronomicAdvice"]:
                    st.write(f"• {tip}")

            # ── Warnings ─────────────────────────────────────────────────────
            if result.get("warnings"):
                st.markdown("### Alerts")
                for w in result["warnings"]:
                    st.warning(w)

            # ── Decision trace ────────────────────────────────────────────────
            with st.expander("Model Decision Trace"):
                for i, step in enumerate(result.get("decisionTrace", []), 1):
                    st.code(f"{i:02d}. {step}", language=None)

        else:
            st.info("Fill in the form and click **Generate Recommendation** to run the ML model.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — HISTORY
# ─────────────────────────────────────────────────────────────────────────────

elif page == "History":
    st.title("Prediction History")
    st.caption("Review all past fertilizer recommendations and soil assessments.")

    records = load_history(50)
    if not records:
        st.info("No predictions yet. Run your first soil analysis on the Predict page.")
    else:
        import pandas as pd
        st.write(f"**{len(records)}** recent analyses")
        df = pd.DataFrame(records).rename(columns={
            "crop_type":         "Crop",
            "soil_type":         "Soil",
            "nitrogen":          "N",
            "phosphorus":        "P",
            "potassium":         "K",
            "ph":                "pH",
            "temperature":       "Temp °C",
            "humidity":          "Humidity %",
            "rainfall":          "Rainfall mm",
            "fertilizer":        "Fertilizer",
            "n_needed":          "N Level",
            "p_needed":          "P Level",
            "k_needed":          "K Level",
            "confidence":        "Confidence",
            "soil_health_score": "Health",
            "created_at":        "Date",
        })
        df["Confidence"] = (df["Confidence"] * 100).round(1).astype(str) + "%"
        df["Health"]     = df["Health"].round(1).astype(str) + "/100"
        df["Date"]       = pd.to_datetime(df["Date"]).dt.strftime("%d %b %Y %H:%M")
        display_cols = ["Date","Crop","Soil","N Level","P Level","K Level",
                        "pH","Fertilizer","Confidence","Health"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Statistics":
    st.title("Statistics")
    st.caption("Aggregate analytics across all fertilizer recommendations.")

    stats = load_stats()
    if not stats:
        st.info("No predictions yet. Run some analyses on the Predict page first.")
    else:
        import plotly.express as px
        import pandas as pd

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Analyses",  stats["total"])
        m2.metric("Top Fertilizer",  stats["top_fertilizer"])
        m3.metric("Top Crop",        stats["top_crop"])
        m4.metric("Avg Confidence",  f"{stats['avg_confidence']*100:.1f}%")

        st.divider()
        ch1, ch2 = st.columns(2)

        with ch1:
            st.subheader("Fertilizer Distribution")
            fert_df = pd.DataFrame(stats["fertilizer_breakdown"])
            fig = px.bar(fert_df, x="count", y="name", orientation="h",
                         labels={"count": "Predictions", "name": ""},
                         color="count", color_continuous_scale="Greens")
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            st.subheader("Crop Distribution")
            crop_df = pd.DataFrame(stats["crop_breakdown"])
            fig2 = px.pie(crop_df, values="count", names="name",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig2, use_container_width=True)
