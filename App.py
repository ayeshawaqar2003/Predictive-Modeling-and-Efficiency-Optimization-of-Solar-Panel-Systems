import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="SolarCleanse Dashboard", layout="wide")

# =====================================================================
# CONSTANTS
# =====================================================================

OUTCOME_LABEL_MAP = {
    0: "Very Clean — No Cleaning Required",
    1: "Slight Dust — Monitor Only",
    2: "Moderate Dirt — Cleaning Required",
    3: "Heavy Dirt — Immediate Cleaning Required"
}

OUTCOME_COLOR_MAP = {
    0: "success",
    1: "info",
    2: "warning",
    3: "error"
}

MODEL_PATH  = "saved_models/best_model.h5"
SCALER_PATH = "saved_models/scaler.pkl"
CONFIG_PATH = "saved_models/best_config.pkl"
FEATURE_PATH = "saved_models/feature_cols.pkl"

# =====================================================================
# LOAD SAVED ASSETS
# =====================================================================

@st.cache_resource
def load_assets():
    """Load trained model, scaler, and config. Returns None if not found."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH, FEATURE_PATH]):
        return None, None, None, None
    model       = load_model(MODEL_PATH)
    scaler      = joblib.load(SCALER_PATH)
    config      = joblib.load(CONFIG_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, scaler, config, feature_cols

@st.cache_data
def load_baseline(file_path="AllInOne.xlsx"):
    """Load and preprocess baseline performance logs."""
    if not os.path.exists(file_path):
        return None
    df = pd.read_excel(file_path)
    df["difference"] = df["Expected_kWh"] - df["Actual_kWh"]
    q1, q2, q3 = df["difference"].quantile([0.25, 0.50, 0.75])
    conditions = [
        (df["difference"] < q1),
        (df["difference"] >= q1) & (df["difference"] < q2),
        (df["difference"] >= q2) & (df["difference"] < q3),
        (df["difference"] >= q3)
    ]
    df["Outcome"] = np.select(conditions, [0, 1, 2, 3], default=1)
    df["Status"] = df["Outcome"].map(OUTCOME_LABEL_MAP)
    return df

# =====================================================================
# UI
# =====================================================================

st.title("☀️ SolarCleanse: Smart Cleaning Scheduler")
st.caption("ML-powered predictive maintenance for large-scale solar panel arrays.")

# Load model assets
model, scaler, config, feature_cols = load_assets()

# Model status banner
if model is not None:
    st.success(f"✅ Model loaded — {config['architecture']} | Timestep window: {config['timesteps']} days")
else:
    st.warning("⚠️ No trained model found. Run `model.py` first to train and save the model, then restart the app.")

st.divider()

# =====================================================================
# SECTION 1 — BASELINE OVERVIEW
# =====================================================================

st.subheader("📊 Baseline Performance Logs")
base_df = load_baseline("AllInOne.xlsx")

if base_df is not None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(base_df))
    col2.metric("Avg Expected kWh", f"{base_df['Expected_kWh'].mean():.2f}")
    col3.metric("Avg Actual kWh",   f"{base_df['Actual_kWh'].mean():.2f}")
    col4.metric("Avg Performance Gap", f"{base_df['difference'].mean():.2f}")

    st.dataframe(
        base_df[["Expected_kWh", "Actual_kWh", "difference", "Outcome", "Status"]].sample(10),
        use_container_width=True
    )
else:
    st.info("No baseline file found. Place AllInOne.xlsx in the project root to view historical logs.")

st.divider()

# =====================================================================
# SECTION 2 — LIVE PREDICTION
# =====================================================================

st.subheader("🤖 Run Maintenance Prediction")
st.caption("Upload fresh SCADA records to get a real-time cleaning recommendation.")

uploaded_file = st.file_uploader("📂 Upload Solar SCADA Records", type=["xlsx", "csv"])

if uploaded_file:
    df_new = (
        pd.read_excel(uploaded_file) 
        if uploaded_file.name.endswith(".xlsx") 
        else pd.read_csv(uploaded_file)
    )
    st.success(f"File uploaded: {len(df_new)} records found.")
    st.dataframe(df_new.head(), use_container_width=True)

    if st.button("⚡ Run Prediction", type="primary"):
        if model is None:
            st.error("No trained model available. Please run model.py first.")
        else:
            with st.spinner("Analyzing sequence patterns..."):
                try:
                    # Feature engineering
                    df_new["difference"] = df_new["Expected_kWh"] - df_new["Actual_kWh"]

                    # Use only the columns the model was trained on
                    available_cols = [c for c in feature_cols if c in df_new.columns]
                    missing_cols   = [c for c in feature_cols if c not in df_new.columns]

                    if missing_cols:
                        st.warning(f"Missing columns from training data: {missing_cols}. Filling with 0.")
                        for col in missing_cols:
                            df_new[col] = 0

                    required_ts = config["timesteps"]

                    if len(df_new) < required_ts:
                        st.error(f"Not enough data. Model requires at least {required_ts} consecutive records. Uploaded file has {len(df_new)}.")
                    else:
                        # Scale using the saved scaler from training
                        feats_scaled   = scaler.transform(df_new[feature_cols].fillna(0).values)

                        # Extract the most recent sequence window
                        last_sequence  = feats_scaled[-required_ts:].reshape(1, required_ts, feats_scaled.shape[1])

                        # Real model prediction
                        probabilities  = model.predict(last_sequence, verbose=0)[0]
                        predicted_class = int(np.argmax(probabilities))
                        confidence      = float(np.max(probabilities)) * 100

                        # Results display
                        st.markdown("### 📌 Maintenance Recommendation")

                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Status", OUTCOME_LABEL_MAP[predicted_class])
                        col2.metric("Model Confidence", f"{confidence:.1f}%")

                        # Probability breakdown
                        st.markdown("**Class Probability Breakdown**")
                        prob_df = pd.DataFrame({
                            "Category": list(OUTCOME_LABEL_MAP.values()),
                            "Probability": [f"{p*100:.1f}%" for p in probabilities]
                        })
                        st.dataframe(prob_df, use_container_width=True)

                        # Action recommendation
                        if predicted_class >= 2:
                            st.error(f"🚨 ACTION REQUIRED: {OUTCOME_LABEL_MAP[predicted_class]} — Schedule cleaning crews immediately.")
                        elif predicted_class == 1:
                            st.info(f"👁️ MONITOR: {OUTCOME_LABEL_MAP[predicted_class]} — No action needed yet. Re-evaluate in 7 days.")
                        else:
                            st.success(f"✅ ALL CLEAR: {OUTCOME_LABEL_MAP[predicted_class]} — Performance nominal. No cleaning required.")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
