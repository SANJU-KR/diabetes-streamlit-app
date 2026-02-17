# =========================================
# Diabetes Prediction App (PIPELINE FIXED)
# =========================================

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# -----------------------------------------
# LOAD PIPELINE
# -----------------------------------------
@st.cache_resource
def load_pipeline():
    pipe = joblib.load("diabetes_catboost_smote_v1.pkl")
    return pipe

pipe = load_pipeline()

# 🔑 IMPORTANT: split pipeline
preprocessor = pipe.named_steps["prep"]
model = pipe.named_steps["model"]

# -----------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------
st.sidebar.title("Patient Information")

age = st.sidebar.number_input("Age", 1, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Transgender"])

if gender == "Female":
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0)
else:
    pregnancies = 0

glucose = st.sidebar.slider("Glucose (mg/dL)", 50, 400, 120)
bp = st.sidebar.slider("Blood Pressure", 50, 200, 80)
bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 25.0)
hba1c = st.sidebar.slider("HbA1c (%)", 3.0, 15.0, 6.5)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
trig = st.sidebar.slider("Triglycerides", 50, 500, 150)
hr = st.sidebar.slider("Heart Rate", 40, 150, 75)
waist_hip = st.sidebar.number_input("Waist-Hip Ratio", 0.5, 2.0, 0.9)

smoking = st.sidebar.selectbox("Smoking", ["Never", "Former", "Current"])
alcohol = st.sidebar.selectbox("Alcohol", ["None", "Low", "Moderate", "High"])
activity = st.sidebar.selectbox("Activity", ["Low", "Moderate", "High"])

hypertension = st.sidebar.radio("Hypertension", [0, 1])
heart_disease = st.sidebar.radio("Heart Disease", [0, 1])
family = st.sidebar.radio("Family History", [0, 1])

metabolic = st.sidebar.slider("Metabolic Score", 0, 4, 2)
obesity_risk = st.sidebar.number_input("Obesity Risk", value=1200)
sugar_load = st.sidebar.number_input("Sugar Load", value=1000)

predict_btn = st.sidebar.button("Predict")

# -----------------------------------------
# MAIN
# -----------------------------------------
st.title("🩺 Diabetes Prediction System")

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

    # 🔥 MANUAL TRANSFORM (NO sklearn check)
    X = preprocessor.transform(input_df)

    # 🔥 DIRECT CATBOOST CALL
    prob = model.predict_proba(X)[0][1] * 100

    # ---------------------------------
    st.subheader("Result")

    if prob < 30:
        st.success(f"Low Risk ({prob:.2f}%)")
    elif prob < 70:
        st.warning(f"Moderate Risk ({prob:.2f}%)")
    else:
        st.error(f"High Risk ({prob:.2f}%)")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.warning("Educational purpose only. Consult a doctor.")
