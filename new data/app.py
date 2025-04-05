import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("lstm_anomaly_model.h5")
scaler = joblib.load("feature_scaler.pkl")

st.set_page_config(page_title="Pet Step Anomaly Detection", layout="centered")
st.title("🐾 Pet Step Anomaly Detector")

st.markdown("Enter the step details below to detect if it's an **anomalous activity**.")

# Input fields for selected features
steps = st.number_input("Steps", min_value=0, value=100)
activity_duration = st.number_input("Activity Duration (in seconds)", min_value=0, value=300)
step_frequency = st.number_input("Step Frequency", min_value=0.0, value=0.5, step=0.01)
rest_period = st.number_input("Rest Period (in seconds)", min_value=0, value=120)
noise_flag = st.selectbox("Noise Flag", options=[0, 1])

# Predict Button
if st.button("Predict Anomaly"):
    input_data = np.array([[steps, activity_duration, step_frequency, rest_period, noise_flag]])
    
    # Scale and reshape for LSTM
    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))

    prediction = model.predict(input_scaled)[0][0]
    is_anomaly = prediction > 0.5

    st.subheader("🔍 Prediction Result:")
    if is_anomaly:
        st.error("⚠️ This activity is classified as an **Anomaly**.")
    else:
        st.success("✅ Normal activity detected.")
