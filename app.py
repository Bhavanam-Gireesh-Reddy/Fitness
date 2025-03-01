import streamlit as st
import joblib
import pandas as pd

# Load the trained model
loaded_model = joblib.load("model.pkl")

# Define the feature columns used during training
feature_columns = [
    "age", "height_cm", "weight_kg", "duration_minutes", "intensity", "calories_burned", 
    "avg_heart_rate", "hours_sleep", "stress_level", "daily_steps", "hydration_level", 
    "bmi", "resting_heart_rate", "blood_pressure_systolic", "blood_pressure_diastolic"
]

# Streamlit UI
st.title("🏋️‍♂️ FitLife360 - Health Condition Prediction")
st.write("Enter your details to predict your health condition!")

# Create input fields for user
input_data = {}
for feature in feature_columns:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Predict Button
if st.button("Predict Health Condition"):
    # Convert input data to DataFrame
    sample_input = pd.DataFrame([input_data])
    
    # Ensure correct column order
    sample_input = sample_input[feature_columns]
    
    # Make prediction
    prediction = loaded_model.predict(sample_input)

    # Display result
    st.success(f"🩺 Predicted Health Condition: **{prediction[0]}**")