import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

st.title("❤️ Cardiovascular Disease Prediction (Neural Network)")

BASE_DIR = Path(__file__).resolve().parent.parent

# โหลด model + scaler
model = load_model(BASE_DIR / "models/nn/nn_model.keras")
scaler = joblib.load(BASE_DIR / "models/nn/scaler.pkl")

# รับ input
age = st.number_input("Age (years)", value=50)
gender = st.selectbox("Gender", ["Female", "Male"])
height = st.number_input("Height (cm)", value=170)
weight = st.number_input("Weight (kg)", value=70)
ap_hi = st.number_input("Systolic BP (ap_hi)", value=120)
ap_lo = st.number_input("Diastolic BP (ap_lo)", value=80)
cholesterol = st.selectbox("Cholesterol (1=Normal, 2=Above Normal, 3=High)", [1,2,3])
gluc = st.selectbox("Glucose (1=Normal, 2=Above Normal, 3=High)", [1,2,3])
smoke = st.selectbox("Smoke", [0,1])
alco = st.selectbox("Alcohol", [0,1])
active = st.selectbox("Physically Active", [0,1])

if st.button("Predict"):

    # แปลง gender ให้เหมือนตอน train
    gender_val = 1 if gender == "Male" else 0

    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender_val,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    # scale
    input_scaled = scaler.transform(input_df)

    # predict
    prediction = model.predict(input_scaled)[0][0]

    if prediction > 0.5:
        st.error(f"⚠️ High Risk of Cardiovascular Disease ({prediction:.2f})")
    else:
        st.success(f"✅ Low Risk ({prediction:.2f})")