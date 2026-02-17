# ================================
# Diabetes Prediction System
# CatBoost + SMOTE (Streamlit)
# ================================

import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
h1 { color: #1f7764; }
.sidebar .sidebar-content { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL (CORRECT WAY) ----------------
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("diabetes_catboost_smote_v1.pkl")
    return model

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

# Pregnancy → ONLY if Female
if gender == "Female":
    pregnancies = st.sidebar.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )
else:
    pregnancies = 0

st.sidebar.subheader("Medical Measurements")
glucose = st.sidebar.slider("Blood Glucose (mg/dL)", 50, 400, 120)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 50, 200, 80)
skin = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.sidebar.slider("Insulin (µU/mL)", 0, 900, 80)
bmi = st.sidebar.number_input("BMI (kg/m²)", 10.0, 70.0, 25.0)
hba1c = st.sidebar.slider("HbA1c (%)", 3.0, 15.0, 6.5)
trig = st.sidebar.slider("Triglycerides (mg/dL)", 50, 500, 150)
hr = st.sidebar.slider("Resting Heart Rate (bpm)", 40, 150, 75)
waist_hip = st.sidebar.number_input("Waist–Hip Ratio", 0.5, 2.0, 0.9)

st.sidebar.subheader("Lifestyle & History")
smoking = st.sidebar.selectbox("Smoking History", ["Never", "Former", "Current"])
alcohol = st.sidebar.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"])
activity = st.sidebar.selectbox("Physical Activity", ["Low", "Moderate", "High"])

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
This application uses a **CatBoost Machine Learning model trained with SMOTE**
to estimate the probability of diabetes using clinical and lifestyle parameters.
""")

# ---------------- BEFORE PREDICTION ----------------
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "CatBoost + SMOTE")
    col2.metric("Accuracy", "~89%")
    col3.metric("Features", "21 Clinical Attributes")

    st.info("👈 Enter patient details in the sidebar and click **Predict**.")

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

    prob = model.predict_proba(input_df)[0][1] * 100

    # ---------------- RESULT ----------------
    st.markdown("---")
    st.header("📊 Prediction Result")

    col1, col2 = st.columns([2, 1])

    with col1:
        if prob < 30:
            st.success("✅ LOW RISK of Diabetes")
        elif prob < 70:
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
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ],
                "bar": {"color": "darkblue"}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.warning("""
**Medical Disclaimer:**  
This tool is for educational purposes only and does not replace professional medical advice.
""")
