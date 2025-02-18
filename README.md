#  Predictive Audit Risk Scoring

##  Project Overview
Predictive Audit Risk Scoring is a **machine learning-powered** system that assesses financial and audit risk based on historical data. Using **XGBoost & Flask**, it predicts risk scores and provides insights into financial anomalies, helping auditors and financial institutions prevent fraud.

##  Purpose of This Repository
This repository is dedicated to developing a **predictive audit risk scoring system**, making it:
- **Accurate**  - Uses ML models for risk prediction.
- **Explainable**  - Includes SHAP visualizations.
- **Scalable**  - Deployable on AWS, Azure, or local servers.
- **Interactive**  - Frontend UI with Streamlit for user-friendly interactions.

---

##  **Project Structure**
```bash
task1/
│── backend/                     # Backend logic & model training
│   │── data/                     # Raw dataset files
│   │── plots/                    # Stored visualization images
│   │── api.py                     # FastAPI backend for predictions
│   │── model.py                   # Model training & evaluation script
│   │── xgb_model.pkl               # Trained machine learning model
│── frontend/                     # Frontend UI (Streamlit)
│   │── assets/                     # Static images/logos
│   │── app.py                      # Streamlit frontend for risk prediction
│── myenv/                         # Virtual environment (not committed)
│── README.md                      # Project documentation
│── requirements.txt                # Dependencies list

---

##  Key Features
-  **Flask API** - Backend for risk prediction.
-  **XGBoost Model** - Trained ML model for accuracy.
-  **SHAP Explainability** - Feature importance visualization.
-  **Streamlit UI** - Interactive user dashboard.
-  **Cloud Deployment Ready** - Scalable architecture.

---

##  Installation Guide
###  Prerequisites
- Python 3.10+
- Git
- Virtual Environment (Recommended)

###  Setup Instructions
```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/predictive-audit-risk-scoring.git
cd predictive-audit-risk-scoring

# Create a virtual environment
python -m venv myenv
source myenv/bin/activate  # For macOS/Linux
myenv\Scripts\activate  # For Windows

# Install dependencies
pip install -r requirements.txt

# Run the backend API
python backend/api.py

# Run the frontend (Streamlit)
streamlit run frontend/app.py
```

---

##  Contribution Guidelines
###  How to Contribute
1. **Fork the repository** 
2. **Clone your fork** 
   ```bash
   git clone https://github.com/YOUR-USERNAME/predictive-audit-risk-scoring.git
   cd predictive-audit-risk-scoring
   ```
3. **Create a new branch** 
   ```bash
   git checkout -b feature-branch
   ```
4. **Make changes & commit** 
   ```bash
   git add .
   git commit -m "Added new feature"
   ```
5. **Push changes** 
   ```bash
   git push origin feature-branch
   ```
6. **Submit a Pull Request (PR)** 

###  Code Formatting
- Follow **PEP8** for Python.
- Use meaningful commit messages.

---

##  License
This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---
##  Code of Conduct
We follow a **friendly and inclusive** environment. Check out [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for guidelines.

---

##  Demo & Usage
###  API Demo
Test the API with `curl`:

```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' -d '{
  "feature1": 1.5,
  "feature2": 2.3,
  "feature3": 3.7,
  "feature4": 0.9,
  "feature5": 1.1,
  "feature6": 2.2,
  "feature7": 1.8,
  "feature8": 0.5,
  "feature9": 3.1,
  "feature10": 2.9,
  "feature11": 0.3,
  "feature12": 1.6,
  "feature13": 2.7,
  "feature14": 1.2,
  "feature15": 3.3,
  "feature16": 0.8,
  "feature17": 2.0,
  "feature18": 1.4,
  "feature19": 2.8,
  "feature20": 1.0
}'
```

###  UI Demo

```bash
# Start Streamlit App
streamlit run frontend/app.py
```

Open `http://localhost:8501/` in your browser.

---
#  Risk Prediction Categories

| **Risk Score** | **Category**    | **Description** |
|---------------|----------------|----------------|
| **0 - 100**   | 🔵 Low Risk     | Strong financial health |
| **101 - 400** | 🟠 Moderate Risk | Some financial concerns but not critical |
| **401 - 600** | 🔴 High Risk     | Severe financial instability or fraud risks |

---

#  Features Used in Model

| **Feature**                 | **Description** |
|-----------------------------|----------------|
| **Sector Score**            | Risk level of the business sector |
| **Financial Score**         | A score based on financial stability |
| **Money Value**             | Total financial transactions in a company |
| **Total Loss**              | Losses recorded in financial statements |
| **Past Fraud History**      | Previous fraud reports in audits |
| **Transaction Size**        | Large transactions are riskier |
| **Operational Efficiency**  | Efficiency in handling funds |
| **Loan Repayment Rate**     | Higher delays in payments → higher risk |
| **Revenue Growth**          | Declining revenue may indicate financial instability |
| **Tax Compliance Score**    | If a company evades taxes, risk is high |
| **Business Age**            | New businesses are riskier than older, established ones |
| **Financial Leverage**      | High debt levels increase financial risk |
| **Cash Flow Stability**     | Inconsistent cash flows → higher risk |
| **Audit Score**             | Past audit performance score |
| **Company Size**            | Smaller businesses tend to have higher fraud risk |
| **Regulatory Compliance**   | If regulations are frequently broken, risk is high |
| **Board Independence**      | Independent boards reduce fraud risk |
| **Profit Margin Stability** | Declining profits → financial distress |
| **Supplier Risk Score**     | If suppliers are unreliable, risk is higher |
| **Investment Risk**         | High-risk investments increase financial exposure |

---

 **Usage:**  
- This file can be **linked in your main `README.md`** file.
- Store this as **`risk_categories.md`** in your documentation folder.

---

 **Next Steps:**
 **Create the file:**  
```bash
touch risk_categories.md


##  Future Improvements

```markdown
- [ ] **Real-time Risk Monitoring**
- [ ] **Deploy to AWS/GCP**
- [ ] **Enhance SHAP Explainability**
- [ ] **Integrate Database Storage**
```

##  Model Training & SHAP Analysis

###  Trained Model
The trained XGBoost model has been successfully saved at:

```bash
/Users/shubhambishnoi/Desktop/task1/backend/xgb_model.pkl
```

### Best Hyperparameters:
```bash
{'learning_rate': 0.1, 'n_estimators': 400}
```

### Model Performance:
```yaml
Train Score: 0.9982882729944789
Test Score: 0.9450530284419184
```


###  Generated Plots
All plots generated during training and SHAP analysis are saved in the **plots/** directory:

|  Plot Name | Description |
|-------------|-------------|
| `adaboost_decision_tree_heatmap.png` | AdaBoost hyperparameter heatmap |
| `adaboost_decision_tree_score_vs_audit_risk.png` | AdaBoost prediction vs actual risk |
| `adaboost_linear_svr_heatmap.png` | AdaBoost Linear SVR hyperparameter heatmap |
| `adaboost_linear_svr_score_vs_audit_risk.png` | AdaBoost Linear SVR prediction plot |
| `audit_risk_distribution.png` | Audit Risk distribution |
| `bagging_decision_tree_heatmap.png` | Bagging Decision Tree hyperparameter heatmap |
| `bagging_knn_score_vs_audit_risk.png` | Bagging KNN prediction vs audit risk |
| `bagging_hyperparameter_tuning_heatmap.png` | Bagging model hyperparameter tuning heatmap |
| `bagging_score_vs_audit_risk.png` | Bagging model prediction plot |
| `correlation_heatmap_after.png` | Correlation heatmap after feature selection |
| `correlation_heatmap_before.png` | Correlation heatmap before feature selection |
| `count_of_risk_levels.png` | Count of different risk levels |
| `float_features_histogram.png` | Histogram of float-type features |
| `gradient_boosting_heatmap.png` | Gradient Boosting hyperparameter heatmap |
| `gradient_boosting_score_vs_audit_risk.png` | Gradient Boosting prediction vs audit risk |
| `gridsearchcv_validation_heatmap.png` | GridSearchCV validation heatmap |
| `integer_features_histogram.png` | Histogram of integer-type features |
| `pasting_decision_tree_heatmap.png` | Pasting Decision Tree hyperparameter heatmap |
| `pasting_decision_tree_score_vs_audit_risk.png` | Pasting Decision Tree prediction plot |
| `pasting_knn_heatmap.png` | Pasting KNN hyperparameter heatmap |
| `pasting_knn_score_vs_audit_risk.png` | Pasting KNN prediction plot |
| `scaled_features_boxplot.png` | Scaled feature distributions |
| `shap_feature_importance_bar_chart.png` | SHAP feature importance (bar plot) |
| `shap_feature_interaction.png` | SHAP feature interaction plot |
| `shap_global_feature_importance.png` | SHAP global feature importance |
| `xgboost_gridsearch_heatmap.png` | XGBoost GridSearchCV hyperparameter heatmap |

---

###  SHAP Force Plot (Interactive)
SHAP force plots cannot be saved as images, so they are saved as an **interactive HTML file**:

####  File Location:
```bash
/plots/shap_force_plot.html
```

Open this file in any browser to explore individual prediction explanations.

---

###  Next Steps:
- Improve the model by tuning additional hyperparameters.
- Experiment with feature selection techniques.
- Compare XGBoost performance with other ensemble models.

**Have suggestions? Feel free to open an issue!**