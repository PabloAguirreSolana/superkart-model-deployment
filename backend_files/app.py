
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "superkart_model.joblib"
)

model = joblib.load(MODEL_PATH)

FEATURES = list(model.feature_names_in_)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "SuperKart Sales Prediction API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not isinstance(data, (dict, list)):
            return jsonify({
                "error": "Expected a JSON object or list of objects"
            }), 400

        # Support single and batch predictions
        if isinstance(data, dict):
            data = [data]

        input_df = pd.DataFrame(data)

        # Validate expected features
        missing = [
            col for col in FEATURES
            if col not in input_df.columns
        ]

        if missing:
            return jsonify({
                "error": f"Missing features: {missing}"
            }), 400

        input_df = input_df[FEATURES]

        if input_df.isnull().any().any():
            return jsonify({
                "error": "Input contains missing values"
            }), 400

        predictions = model.predict(input_df)

        return jsonify({
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
