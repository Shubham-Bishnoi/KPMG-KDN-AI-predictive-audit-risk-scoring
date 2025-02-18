from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model_path = "backend/xgb_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(" Model file not found. Train and save the model first!")

model = joblib.load(model_path)

# Get the expected number of features
expected_feature_count = model.n_features_in_

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Predictive Audit Risk Scoring API! Use /predict to get risk scores."})

@app.route("/predict", methods=["POST"])
def predict_risk():
    try:
        data = request.get_json()
        
        # Ensure correct number of features
        if len(data) != expected_feature_count:
            return jsonify({"error": f"Feature count mismatch. Expected: {expected_feature_count}, Got: {len(data)}"}), 400

        # Convert input data into a DataFrame
        df = pd.DataFrame([data.values()])  # Avoid using feature names since model does not have them

        # Make prediction
        risk_score = model.predict(df)[0]

        return jsonify({"risk_score": float(risk_score)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
