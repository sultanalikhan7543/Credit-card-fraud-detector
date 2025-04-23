import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("💳 Credit Card Fraud Detection")

st.markdown("Enter the transaction details below:")

# Collect user input
input_data = []

# Add 'Time' input
time = st.number_input("Time", value=0.0)
input_data.append(time)

# Add V1 to V28 inputs
for i in range(1, 29):
    input_data.append(st.number_input(f"V{i}", value=0.0))

# Add 'Amount' input
amount = st.number_input("Amount", value=0.0)
input_data.append(amount)

# Convert to NumPy array
input_data = np.array(input_data).reshape(1, -1)

# Predict button
if st.button("Predict"):
    try:
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = "🛑 Fraud" if prediction[0] == 1 else "✅ Not Fraud"
        st.success(f"Prediction: {result}")
    except ValueError as e:
        st.error(f"Input error: {e}")
