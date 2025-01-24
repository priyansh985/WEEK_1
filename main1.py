import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the model and scaler
model = joblib.load('Random.pkl')
scaler = joblib.load('std_scaler.pkl')

# Title of the app
st.title("Crop Recommendation System")

# Subtitle
st.subheader("Enter the required parameters to get crop recommendations")
# Input fields
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
N = st.number_input("Nitrogen (N) content in soil", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
P = st.number_input("Phosphorous (P) content in soil", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
K = st.number_input("Potassium (K) content in soil", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

# Button to get recommendations
if st.button("Recommend Crop"):
    try:
        # Prepare the input data with feature names
        input_data = pd.DataFrame([[temperature, humidity, pH, rainfall]], columns=['temperature', 'humidity', 'ph', 'rainfall'])
        scaled_features = scaler.transform(input_data)
        final_data = np.hstack([scaled_features, [[N, P, K]]])
        
        # Log final data shape
        st.write("Final Data Shape:", final_data.shape)
        
        # Predict the crop
        recommendation = model.predict(final_data)
        
        # Display recommendation
        st.success(f"Recommended Crop for You: {recommendation[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")