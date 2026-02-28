import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent

model = tf.keras.models.load_model(
    BASE_DIR / "models/nn/nn_model.keras"
)

scaler = joblib.load(
    BASE_DIR / "models/nn/scaler.pkl"
)

st.set_page_config(
    page_title="Heart Disease Prediction (Neural Network)",
    layout="centered"
)

st.title("Heart Disease Prediction (Neural Network)")
st.write("Fill in patient information below:")

age = st.number_input("Age", min_value=1, max_value=120, value=55)

sex_dict = {
    "Female": 0,
    "Male": 1
}
sex_label = st.selectbox("Sex", list(sex_dict.keys()))
sex = sex_dict[sex_label]

cp_dict = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp_label = st.selectbox("Chest Pain Type", list(cp_dict.keys()))
cp = cp_dict[cp_label]

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=130)
chol = st.number_input("Cholesterol (mg/dl)", value=250)

fbs_dict = {
    "Normal (≤120 mg/dl)": 0,
    "High (>120 mg/dl)": 1
}
fbs_label = st.selectbox("Fasting Blood Sugar", list(fbs_dict.keys()))
fbs = fbs_dict[fbs_label]

restecg_dict = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg_label = st.selectbox("Rest ECG", list(restecg_dict.keys()))
restecg = restecg_dict[restecg_label]

thalach = st.number_input("Maximum Heart Rate Achieved", value=150)

exang_dict = {
    "No": 0,
    "Yes": 1
}
exang_label = st.selectbox("Exercise Induced Angina", list(exang_dict.keys()))
exang = exang_dict[exang_label]

oldpeak = st.number_input("Oldpeak (ST Depression)", value=1.0)

slope_dict = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
slope_label = st.selectbox("Slope of Peak Exercise ST Segment", list(slope_dict.keys()))
slope = slope_dict[slope_label]

ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

thal_dict = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2,
    "Unknown": 3
}
thal_label = st.selectbox("Thalassemia", list(thal_dict.keys()))
thal = thal_dict[thal_label]

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    input_scaled = scaler.transform(input_df)
    probability = model.predict(input_scaled)[0][0]


    if probability >= 0.5:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")