import streamlit as st
import requests
import json
from PIL import Image

# Streamlit UI Setup
st.title("🔍 Predictive Audit Risk Scoring")
st.write("Enter the financial/audit details below to get a risk score.")

# Input Fields
feature1 = st.number_input("Feature 1", value=1.5)
feature2 = st.number_input("Feature 2", value=2.3)
feature3 = st.number_input("Feature 3", value=3.7)

# API Endpoint
API_URL = "http://127.0.0.1:8000/predict/"

if st.button("Predict Risk Score"):
    input_data = {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3
    }
    
    # Send request to FastAPI backend
    response = requests.post(API_URL, data=json.dumps(input_data), headers={"Content-Type": "application/json"})
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Risk Score: {result['risk_score']:.2f}")
    else:
        st.error("Failed to get prediction. Check API connection.")
