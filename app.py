# ================================
# Diabetes Prediction System
# CatBoost + SMOTE (joblib)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
h1 { color: #1f7764; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL (JOBLIB ONLY) ----------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("diabetes_catboost_smote_v1.pkl")
    except Exception as e:
        st.error("❌ Model loading failed")
        st.exception(e)
        st.stop()

model = load_model()

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.title("🧑‍⚕️ Patient Information")

st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age (years)", 18, 100, 30)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Transgender"]
)

# Pregnancy ONLY if Female
if gender == "Female":
    pregnancies = st.sidebar.number_input(
        "Pregnancies", min_value=0, max_value=20, value=0
    )
else:
    pregnancies = 0

st.sidebar.subheader("Medical Measurements")
bmi = st.sidebar.number_input("BMI (kg/m²)", 10.0, 70.0, 25.0)
waist_hip = st.sidebar.number_input("Waist–Hip Ratio", 0.5, 2.0, 0.9)
glucose = st.sidebar.slider("Blood Glucose (mg/dL)", 50, 400, 120)
hba1c = st.sidebar.slider("HbA1c (%)", 3.0, 15.0, 6.5)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 50, 200, 80)
insulin = st.sidebar.slider("Insulin (µU/mL)", 0, 900, 80)
skin = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
trig = st.sidebar.slider("Triglycerides (mg/dL)", 50, 500, 150)
hr = st.sidebar.slider("Resting Heart Rate (bpm)", 40, 150, 75)

st.sidebar.subheader("Lifestyle & History")
alcohol = st.sidebar.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"])
smoking = st.sidebar.selectbox("Smoking History", ["Never", "Former", "Current"])
activity = st.sidebar.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

hypertension = st.sidebar.radio("Hypertension (0/1)", [0, 1])
heart_disease = st.sidebar.radio("Heart Disease (0/1)", [0, 1])
family = st.sidebar.radio("Family History (0/1)", [0, 1])

metabolic = st.sidebar.slider("Metabolic Score (0–4)", 0, 4, 2)
obesity_risk = st.sidebar.number_input("Obesity Risk Index", value=1200)
sugar_load = st.sidebar.number_input("Chronic Sugar Load", value=1000)

predict_btn = st.sidebar.button("🔍 Predict Diabetes Risk", use_container_width=True)

# ================================
# MAIN UI
# ================================
st.title("🩺 Diabetes Prediction System")
st.markdown("### AI-Powered Diabetes Risk Assessment Tool")

st.markdown("""
This system uses a **CatBoost model trained with SMOTE balancing**
to estimate diabetes risk based on **21 clinical attributes**.
""")

# ---------------- BEFORE PREDICTION ----------------
if not predict_btn:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "CatBoost + SMOTE")
    c2.metric("Accuracy", "~89%")
    c3.metric("Features", "21 Attributes")
    st.info("👈 Enter details in the sidebar and click **Predict**.")

# ================================
# PREDICTION
# ================================
if predict_btn:

    input_df = pd.DataFrame([{
        "Age (years)": age,
        "Gender": gender,
        "BMI (kg/m2)": bmi,
        "WaistToHipRatio": waist_hip,
        "BloodGlucoseLevel (mg/dL)": glucose,
        "HbA1cLevel (%)": hba1c,
        "BloodPressure (mmHg)": bp,
        "Insulin (µU/mL)": insulin,
        "SkinThickness (mm)": skin,
        "Triglycerides (mg/dL)": trig,
        "RestingHeartRate (bpm)": hr,
        "AlcoholConsumption": alcohol,
        "SmokingHistory": smoking,
        "PhysicalActivityLevel": activity,
        "Hypertension (0/1)": hypertension,
        "HeartDisease (0/1)": heart_disease,
        "FamilyHistory (0/1)": family,
        "Pregnancies": pregnancies,
        "Metabolic_Score (count_0-4)": metabolic,
        "Obesity_Risk (kg/m2 * years)": obesity_risk,
        "Chronic_Sugar_Load (mg/dL * %)": sugar_load
    }])

    # 🔑 FIXED PROBABILITY (NO 100% BUG)
    raw_score = model.predict(input_df, prediction_type="RawFormulaVal")[0]
    prob = 1 / (1 + np.exp(-raw_score)) * 100

    # ---------------- RESULT ----------------
    st.markdown("---")
    st.header("📊 Prediction Result")

    col1, col2 = st.columns([2, 1])

    with col1:
        if prob < 35:
            st.success("✅ LOW RISK of Diabetes")
        elif prob < 65:
            st.warning("⚠️ MODERATE RISK of Diabetes")
        else:
            st.error("❌ HIGH RISK of Diabetes")

        st.metric("Diabetes Probability", f"{prob:.2f}%")

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "%"},
            title={"text": "Risk Level"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 35], "color": "lightgreen"},
                    {"range": [35, 65], "color": "yellow"},
                    {"range": [65, 100], "color": "red"}
                ],
                "bar": {"color": "darkblue"}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.warning("""
**Medical Disclaimer:**  
This tool is for educational purposes only and does NOT replace professional medical advice.
""")
