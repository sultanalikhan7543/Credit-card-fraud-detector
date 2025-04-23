import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('fraud_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("💳 Credit Card Fraud Detection")

# Input fields for features
input_data = []
for i in range(1, 29):  # V1 to V28
    value = st.number_input(f"V{i}", step=0.01)
    input_data.append(value)

amount = st.number_input("Amount", step=0.01)
input_data.append(amount)

# Predict
if st.button("Predict"):
    features = scaler.transform([input_data])
    result = model.predict(features)[0]
    if result == 1:
        st.error("⚠️ Fraud Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
