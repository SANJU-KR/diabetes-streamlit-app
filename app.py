import streamlit as st
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# ---------------- LOAD MODEL (FIXED) ----------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_catboost_smote_v1.pkl")

model = load_model()

# ---------------- UI ----------------
st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes risk")

# -------- Inputs --------
age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Transgender"]
)

bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.5)
waist_to_hip = st.number_input("Waist to Hip Ratio", value=0.95)
glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=160)
hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=6.8)
bp = st.number_input("Blood Pressure (mmHg)", value=130)
insulin = st.number_input("Insulin (µU/mL)", value=15)
skin = st.number_input("Skin Thickness (mm)", value=22)
trig = st.number_input("Triglycerides (mg/dL)", value=180)
hr = st.number_input("Resting Heart Rate (bpm)", value=78)

alcohol = st.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"])
smoking = st.selectbox("Smoking History", ["Never", "Former", "Current"])
activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

hypertension = st.radio("Hypertension (0/1)", [0, 1])
heart_disease = st.radio("Heart Disease (0/1)", [0, 1])
family = st.radio("Family History (0/1)", [0, 1])

# -------- Pregnancy logic (IMPORTANT) --------
if gender == "Female":
    preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
else:
    preg = 0  # Auto set for Male / Other

metabolic = st.slider("Metabolic Score (0–4)", 0, 4, 2)
obesity_risk = st.number_input("Obesity Risk (kg/m² × years)", value=1200)
sugar_load = st.number_input("Chronic Sugar Load (mg/dL × %)", value=1080)

# -------- Prediction --------
if st.button("Predict Diabetes Risk"):
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
        "Pregnancies": preg,
        "Metabolic_Score (count_0-4)": metabolic,
        "Obesity_Risk (kg/m2 * years)": obesity_risk,
        "Chronic_Sugar_Load (mg/dL * %)": sugar_load
    }])

    prob = model.predict_proba(input_df)[0][1]
    st.success(f"🟢 Diabetes Probability: **{round(prob, 3)}**")
