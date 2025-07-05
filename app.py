import streamlit as st
import numpy as np
import joblib

st.title("Diabetes Predictor")

# Load the trained model and scaler
model = joblib.load('svm_diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')  # Ensure scaler.pkl is in your project directory

# Streamlit form for input
with st.form(key='diabetes_form'):
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucose', min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin', min_value=0.0, max_value=1000.0, value=79.0)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=20.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age', min_value=1, max_value=120, value=33)
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    features_scaled = scaler.transform([features])
    result = model.predict(features_scaled)[0]
    prediction = 'Diabetic' if result == 1 else 'Not Diabetic'
    st.success(f"Prediction: {prediction}")
