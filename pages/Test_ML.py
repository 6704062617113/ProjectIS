import streamlit as st
import numpy as np
import pickle
from pathlib import Path
st.title("Diabetes Prediction (Ensemble ML)")
BASE_DIR = Path(__file__).resolve().parent.parent
model = pickle.load(open(BASE_DIR / "models/ml/ml_model.pkl", "rb"))
scaler = pickle.load(open(BASE_DIR / "models/ml/scaler.pkl", "rb"))
preg = st.number_input("Pregnancies", value=1)
glucose = st.number_input("Glucose", value=100)
bp = st.number_input("Blood Pressure", value=70)
skin = st.number_input("Skin Thickness", value=20)
insulin = st.number_input("Insulin", value=80)
bmi = st.number_input("BMI", value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
age = st.number_input("Age", value=30)
if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("High Risk of Diabetes Detected")
    else:
        st.success("Low Risk of Diabetes Detected")