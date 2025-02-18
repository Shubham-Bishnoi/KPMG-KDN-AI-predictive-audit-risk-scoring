import streamlit as st
import requests
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://127.0.0.1:8000/predict"
PLOT_DIR = "../backend/plots"  # Adjust the path based on your file structure

st.title("🔍 Predictive Audit Risk Scoring")
st.write("Enter the company's financial details to predict audit risk.")

# Creating input fields for all 20 features with descriptions
feature_inputs = {
    "Sector Score": st.number_input("Risk level of the business sector", value=1.5),
    "Financial Score": st.number_input("A score based on financial stability", value=2.3),
    "Money Value": st.number_input("Total financial transactions in a company", value=3.7),
    "Total Loss": st.number_input("Losses recorded in financial statements", value=0.9),
    "Past Fraud History": st.number_input("Previous fraud reports in audits", value=1.1),
    "Transaction Size": st.number_input("Large transactions are riskier", value=2.2),
    "Operational Efficiency": st.number_input("Efficiency in handling funds", value=1.8),
    "Loan Repayment Rate": st.number_input("Higher delays in payments → higher risk", value=0.5),
    "Revenue Growth": st.number_input("Declining revenue may indicate financial instability", value=3.1),
    "Tax Compliance Score": st.number_input("If a company evades taxes, risk is high", value=2.9),
    "Business Age": st.number_input("New businesses are riskier than older, established ones", value=0.3),
    "Financial Leverage": st.number_input("High debt levels increase financial risk", value=1.6),
    "Cash Flow Stability": st.number_input("Inconsistent cash flows → higher risk", value=2.7),
    "Audit Score": st.number_input("Past audit performance score", value=1.2),
    "Company Size": st.number_input("Smaller businesses tend to have higher fraud risk", value=3.3),
    "Regulatory Compliance": st.number_input("If regulations are frequently broken, risk is high", value=0.8),
    "Board Independence": st.number_input("Independent boards reduce fraud risk", value=2.0),
    "Profit Margin Stability": st.number_input("Declining profits → financial distress", value=1.4),
    "Supplier Risk Score": st.number_input("If suppliers are unreliable, risk is higher", value=2.8),
    "Investment Risk": st.number_input("High-risk investments increase financial exposure", value=1.0),
}

import matplotlib.pyplot as plt
import numpy as np

# Predict Button
if st.button("🔎 Predict Risk Score"):
    response = requests.post(API_URL, json=feature_inputs)
    if response.status_code == 200:
        risk_score = response.json().get("risk_score", "Error fetching risk score")

        # Display Risk Score
        st.success(f"Predicted Risk Score: {risk_score}")

        # Categorize Risk & Provide Descriptions
        if risk_score < 100:
            risk_category = "🔵 Low Risk"
            description = "✅ The company has a strong financial position with no major risk factors."
            color = "blue"
        elif risk_score < 400:
            risk_category = "🟠 Moderate Risk"
            description = "⚠️ The company has some financial concerns but is not in immediate danger."
            color = "orange"
        else:
            risk_category = "🔴 High Risk"
            description = "🚨 The company has severe financial risks, regulatory issues, or fraud history!"
            color = "red"

        # Display Risk Category
        st.markdown(f"### {risk_category}")
        st.write(description)

        # **Risk Scale Visualization**
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 1)

        # Color-coded segments
        ax.fill_betweenx([0, 1], 0, 100, color="blue", alpha=0.3, label="Low Risk (0-100)")
        ax.fill_betweenx([0, 1], 100, 400, color="orange", alpha=0.3, label="Moderate Risk (100-400)")
        ax.fill_betweenx([0, 1], 400, 600, color="red", alpha=0.3, label="High Risk (400-600)")

        # Risk Score Pointer
        ax.plot([risk_score, risk_score], [0, 1], color=color, linewidth=3, marker="o", markersize=8)

        # Labels & Aesthetics
        ax.set_xticks([0, 100, 400, 600])
        ax.set_xticklabels(["0", "100", "400", "600"])
        ax.set_yticks([])
        ax.set_xlabel("Risk Score Scale")

        # Display Risk Scale in Streamlit
        st.pyplot(fig)

    else:
        st.error("❌ Failed to fetch risk score. Check API connection.")

# Option to Show More Plots
st.sidebar.header("📊 Explore Data Insights")
selected_plot = st.sidebar.selectbox(
    "Choose a visualization to view:",
    [
        "Feature Correlation Heatmap",
        "Audit Risk Score Distribution",
        "Risk Level Distribution",
        "Integer Features Histogram",
        "Float Features Histogram",
        "Pairplot of Selected Features",
        "Gradient Boosting Score vs Audit Risk",
        "XGBoost GridSearch Heatmap"
    ]
)

plot_files = {
    "Feature Correlation Heatmap": "correlation_heatmap_after.png",
    "Audit Risk Score Distribution": "audit_risk_distribution.png",
    "Risk Level Distribution": "count_of_risk_levels.png",
    "Integer Features Histogram": "integer_features_histogram.png",
    "Float Features Histogram": "float_features_histogram.png",
    "Pairplot of Selected Features": "pairplot_features.png",
    "Gradient Boosting Score vs Audit Risk": "gradient_boosting_score_vs_audit_risk.png",
    "XGBoost GridSearch Heatmap": "xgboost_gridsearch_heatmap.png",
}




# Footer with Project Details
st.markdown("---")
st.markdown(
    """
    **🔗 Project Details**  
    *This project is developed as part of KPMG's Predictive Audit Risk Scoring initiative. It utilizes advanced machine learning techniques to predict financial risks in businesses based on key financial parameters.*  
    **🔍 Technologies Used:** Python, Streamlit, Flask API, XGBoost, SHAP, Seaborn.  
    📌 **GitHub Repository:** [KPMG Audit Risk Scoring](https://github.com/Shubham-Bishnoi/KPMG-KDN-AI-predictive-audit-risk-scoring.git)
    """
)
st.markdown("<p style='text-align: center;'>© 2025 KPMG - All Rights Reserved</p>", unsafe_allow_html=True)