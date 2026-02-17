import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_catboost_smote_v1.pkl")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧍 Patient Information")

age = st.sidebar.slider("Age (years)", 1, 120, 30)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Transgender"]
)

# Pregnancy only for Female
pregnancies = 0
if gender == "Female":
    pregnancies = st.sidebar.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

st.sidebar.subheader("🩺 Medical Measurements")

glucose = st.sidebar.slider("Blood Glucose Level (mg/dL)", 50, 400, 120)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 60, 200, 80)
skin = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.sidebar.slider("Insulin (µU/mL)", 0, 300, 80)
bmi = st.sidebar.slider("BMI (kg/m²)", 10.0, 60.0, 25.0)
waist_to_hip = st.sidebar.slider("Waist to Hip Ratio", 0.5, 1.5, 0.9)
hba1c = st.sidebar.slider("HbA1c Level (%)", 3.0, 15.0, 6.5)
trig = st.sidebar.slider("Triglycerides (mg/dL)", 50, 500, 150)
hr = st.sidebar.slider("Resting Heart Rate (bpm)", 40, 150, 75)

alcohol = st.sidebar.selectbox(
    "Alcohol Consumption",
    ["None", "Low", "Moderate", "High"]
)

smoking = st.sidebar.selectbox(
    "Smoking History",
    ["Never", "Former", "Current"]
)

activity = st.sidebar.selectbox(
    "Physical Activity Level",
    ["Low", "Moderate", "High"]
)

hypertension = st.sidebar.radio("Hypertension", [0, 1])
heart_disease = st.sidebar.radio("Heart Disease", [0, 1])
family = st.sidebar.radio("Family History", [0, 1])

metabolic = st.sidebar.slider("Metabolic Score (0–4)", 0, 4, 2)
obesity_risk = st.sidebar.number_input(
    "Obesity Risk (kg/m² × years)", value=1200
)
sugar_load = st.sidebar.number_input(
    "Chronic Sugar Load (mg/dL × %)", value=1080
)

predict_btn = st.sidebar.button("🔍 Predict")

# ---------------- MAIN UI ----------------
st.title("🩺 Diabetes Prediction System")
st.caption("AI-powered risk assessment using CatBoost + SMOTE")

st.markdown("""
### 📌 About This System
This system predicts **diabetes risk probability** using a **machine learning pipeline**
trained on **21 clinical and lifestyle features**.

- Handles class imbalance using **SMOTE**
- Uses **CatBoost Classifier**
- Suitable for clinical risk screening (educational purpose)
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Model Type", "CatBoost + SMOTE")
col2.metric("Model Accuracy", "~88%")
col3.metric("Total Features", "21")

st.markdown("---")

# ---------------- PREDICTION ----------------
if predict_btn:
    input_df = pd.DataFrame([{
        "Age (years)": age,
        "Gender": gender,
        "BMI (kg/m2)": bmi,
        "WaistToHipRatio": waist_to_hip,
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

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prob >= 0.7:
        st.error(f"🔴 High Diabetes Risk\n\nProbability: **{prob:.3f}**")
    elif prob >= 0.4:
        st.warning(f"🟠 Moderate Diabetes Risk\n\nProbability: **{prob:.3f}**")
    else:
        st.success(f"🟢 Low Diabetes Risk\n\nProbability: **{prob:.3f}**")
