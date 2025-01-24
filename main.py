import streamlit as st
import numpy as np
# Removed unused pandas import
import pickle

# Load the model and scaler
model_path = "Random.pkl"
scaler_path = "std_scaler.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title of the app
st.title("Crop Recommendation System")

# Subtitle
st.subheader("Enter the required parameters to get crop recommendations")

# Input fields for user inputs
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=43.675493, value=25.0, step=0.1)
    pH = st.number_input("Soil pH Level", min_value=0.0, max_value=9.935091, value=7.0, step=0.1)
    N = st.number_input("Nitrogen (N) content in soil", min_value=0.0, max_value=140.0, value=50.0, step=1.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=99.981876, value=50.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=298.560117, value=100.0, step=1.0)
    P = st.number_input("Phosphorous (P) content in soil", min_value=0.0, max_value=145.0, value=30.0, step=1.0)

# Potassium input
K = st.number_input("Potassium (K) content in soil", min_value=0.0, max_value=205.0, value=20.0, step=1.0)

# Button to get recommendations
if st.button("Recommend Crop"):
    try:
        # Prepare the input data
        scaled_input = scaler.transform(np.array([[temperature, humidity, pH, rainfall]]))
        non_scaled_input = np.array([[N, P, K]])
        final_data = np.concatenate([scaled_input, non_scaled_input], axis=1)
        
        # Log final data shape
        st.write("Final Data Shape:", final_data.shape)
        
        # Predict the crop
        recommendation = model.predict(final_data)
        
        # Display recommendation
        st.success(f"Recommended Crop: {recommendation}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
