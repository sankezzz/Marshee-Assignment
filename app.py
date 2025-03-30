import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# 📌 Load Saved Model & Scalers
model = tf.keras.models.load_model("lstm_calories_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# 🎨 Streamlit UI
st.title("🔥 Calories Prediction using LSTM")
st.write("Enter your activity details to predict calories burned.")

# 📌 User Inputs
TotalSteps = st.number_input("Total Steps", min_value=0, max_value=50000, value=10000)
TotalDistance = st.number_input("Total Distance (km)", min_value=0.0, max_value=50.0, value=5.0)
VeryActiveDistance = st.number_input("Very Active Distance (km)", min_value=0.0, max_value=20.0, value=1.5)
LightActiveDistance = st.number_input("Light Active Distance (km)", min_value=0.0, max_value=30.0, value=3.5)
VeryActiveMinutes = st.number_input("Very Active Minutes", min_value=0, max_value=1440, value=60)
SedentaryMinutes = st.number_input("Sedentary Minutes", min_value=0, max_value=1440, value=600)

# 📌 Predict Button
if st.button("Predict Calories"):
    # Format Input for Model
    user_input = np.array([[TotalSteps, TotalDistance, VeryActiveDistance, LightActiveDistance, VeryActiveMinutes, SedentaryMinutes]])
    user_input_scaled = scaler_X.transform(user_input)
    user_input_reshaped = user_input_scaled.reshape((1, 1, user_input_scaled.shape[1]))

    # 📌 Model Prediction
    predicted_scaled = model.predict(user_input_reshaped)
    predicted_calories = scaler_y.inverse_transform(predicted_scaled)[0][0]

    # 🎯 Display Result
    st.success(f"🔥 Estimated Calories Burned: **{predicted_calories*1000:.2f} cal**")
