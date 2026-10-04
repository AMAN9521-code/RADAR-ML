from flask import Flask, request, jsonify
import joblib
import pandas as pd

# =========================================================
# RADAR ML SERVICE
# =========================================================

app = Flask(__name__)

MODEL_FILE = "radar_fraud_model.joblib"

model = joblib.load(MODEL_FILE)

print("RADAR Random Forest model loaded successfully.")


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "OK",
        "service": "RADAR ML Fraud Detection",
        "model": "Random Forest"
    })


# =========================================================
# FRAUD PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        amount = float(data["amount"])
        merchant = str(data["merchant"])
        location = str(data["location"])

        transaction = pd.DataFrame([
            {
                "Amount": amount,
                "Merchant": merchant,
                "Location": location
            }
        ])

        # Fraud probability
        probabilities = model.predict_proba(transaction)[0]

        fraud_probability = float(probabilities[1])

        legitimate_probability = float(probabilities[0])

        prediction = int(
            model.predict(transaction)[0]
        )

        return jsonify({
            "prediction": prediction,
            "fraud_probability": round(
                fraud_probability * 100,
                2
            ),
            "legitimate_probability": round(
                legitimate_probability * 100,
                2
            ),
            "model": "Random Forest"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("RADAR ML SERVICE")
    print("========================================")
    print("Running on http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )