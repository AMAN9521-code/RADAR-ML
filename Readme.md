# RADAR ML: Fraud Prediction Service

The machine-learning service for the **RADAR (Risk Analysis and Detection of Anonymous Response)** fraud detection system. It trains a **Random Forest** classifier on transaction data and serves predictions through a small Flask REST API.

**Main application repository:** https://github.com/AMAN9521-code/FraudDetectionSystem
*(Java, Servlets, Tomcat, MySQL. This repository is the ML companion to that project.)*

---

## What it does

Given a transaction's **amount, merchant and location**, the service returns:

- a prediction (`0` = legitimate, `1` = fraud)
- the fraud probability (percentage)
- the legitimate probability (percentage)

The main RADAR application calculates a rule-based risk score (amount, velocity, location and z-score rules). This service adds a trained ML model that can be combined with that score.

---

## Repository contents

| File | Purpose |
|---|---|
| `train_model.py` | Loads the dataset, trains the Random Forest pipeline, prints accuracy and a classification report, and saves the model |
| `radar_ml_training_dataset.csv` | Training dataset, 1,500 rows (columns: `Date`, `Amount`, `Merchant`, `Location`, `Transaction ID`, `Fraud`) |
| `radar_fraud_model.joblib` | The trained model (a full scikit-learn pipeline) |
| `ml_service.py` | Flask API that loads the model and serves predictions |
| `requirements.txt` | Python dependencies |
| `Procfile` | Start command for Railway: `gunicorn ml_service:app --bind 0.0.0.0:$PORT` |
| `railway.toml` | Railway deployment configuration |

---

## How the model works

**Features used:** `Amount` (number), `Merchant` (text category), `Location` (text category)
**Target:** `Fraud` (0 = legitimate, 1 = fraud)
**Not used for training:** `Date` and `Transaction ID` (present in the dataset but ignored by the model)

**Pipeline** (`train_model.py`):

1. **Preprocessing:** `Merchant` and `Location` are converted to numbers with one-hot encoding (unknown values at prediction time are ignored safely). `Amount` is passed through unchanged.
2. **Classifier:** `RandomForestClassifier` with 200 trees, maximum depth 12, `class_weight="balanced"` (so the rare fraud class is not ignored) and `random_state=42`.
3. **Evaluation:** 80% of the data (1,200 rows) is used for training and 20% (300 rows) for testing, split with stratification so both sets keep the same fraud ratio. The script prints accuracy and a precision/recall report for both classes.
4. **Saving:** the whole pipeline (encoder and classifier together) is saved as `radar_fraud_model.joblib`, so the API only needs the raw three fields.

### Model results

Evaluated on the held-out test set of 300 transactions (261 legitimate, 39 fraud):

| Metric | Value |
|---|---|
| Accuracy | 99.67% (299 of 300 correct) |
| Fraud precision | 0.97 |
| Fraud recall | 1.00 (all 39 fraud cases caught) |
| Fraud F1-score | 0.99 |
| Legitimate precision / recall | 1.00 / 1.00 |

| | Predicted legitimate | Predicted fraud |
|---|---|---|
| **Actually legitimate** | 260 | 1 |
| **Actually fraud** | 0 | 39 |

**How to read this:** the model missed no fraud and raised one false alarm on a legitimate transaction. Fraud is the rarer class (13% of the test set), so recall and precision for fraud matter more than overall accuracy.

**Caveat:** these numbers come from a single train/test split on a small dataset prepared for this project, not from real banking data. Scores this high suggest the fraud pattern in the dataset is easy to separate, so real-world performance would be lower. See *Limitations* and *Future scope*.

---

## Requirements

- Python 3.10 or newer
- Packages in `requirements.txt`: Flask, pandas, numpy, scikit-learn, joblib, gunicorn

---

## Setup and run locally

```bash
git clone https://github.com/AMAN9521-code/RADAR-ML.git
cd RADAR-ML

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

### 1. (Optional) Retrain the model

```bash
python train_model.py
```

This reads `radar_ml_training_dataset.csv`, prints the evaluation report, and overwrites `radar_fraud_model.joblib`.

> If loading the saved model fails with a version warning or error, your scikit-learn version differs from the one used to train it. Run `python train_model.py` to create a model that matches your installed version.

### 2. Start the API

```bash
python ml_service.py
```

The service runs at **http://127.0.0.1:5000**.

---

## API reference

### `GET /health`

Checks that the service and model are running.

```bash
curl http://127.0.0.1:5000/health
```

```json
{
  "status": "OK",
  "service": "RADAR ML Fraud Detection",
  "model": "Random Forest"
}
```

### `POST /predict`

Request body (JSON), all three fields required:

| Field | Type | Example |
|---|---|---|
| `amount` | number | `95000` |
| `merchant` | string | `"Electronics"` |
| `location` | string | `"Delhi"` |

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 95000, "merchant": "Electronics", "location": "Delhi"}'
```

Response (example format):

```json
{
  "prediction": 1,
  "fraud_probability": 87.5,
  "legitimate_probability": 12.5,
  "model": "Random Forest"
}
```

- `prediction`: `1` means fraud, `0` means legitimate
- probabilities are percentages and add up to 100
- a missing or invalid field returns HTTP 400 with an `{"error": "..."}` message

> On Windows PowerShell, use `Invoke-RestMethod` instead of `curl`:
> ```powershell
> Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body '{"amount":95000,"merchant":"Electronics","location":"Delhi"}'
> ```

---

## Deployment (Railway)

The repository includes a `Procfile` and `railway.toml`.

1. Create a new Railway service from this GitHub repository.
2. Railway installs `requirements.txt` and starts the app with `gunicorn ml_service:app --bind 0.0.0.0:$PORT`.
3. Generate a public domain, or use Railway's private networking so only the main RADAR application can reach it.
4. Test with `GET /health` on the deployed address.

---

## Connecting to the main RADAR application

**Integration status:** Integrated and verified on the hosted deployment. The Java application (`MLFraudService`) calls this service's `POST /predict` endpoint, which runs as a separate Railway service, and combines the returned fraud probability with its rule-based score (final risk = 40% rule-based + 60% ML). The service address is set through the `ML_API_URL` environment variable and defaults to `http://127.0.0.1:5000/predict` for local runs. For example, a transaction of ₹97,354 received a rule-based score of 53.14% and an ML score of 90.41%, giving a final risk of 75.50%, and a fraud alert was raised with the ML score recorded in the alert. If the ML service is unreachable, the application logs the error and falls back to the rule-based score, so transactions are still processed.

---

## Limitations

- The model uses only three features (amount, merchant, location). It does not see transaction history, velocity or time of day, which the rule engine in the main application covers.
- The dataset is small (1,500 rows) and was prepared for this project. It is not real banking data.
- The reported scores come from one 80/20 split. The near-perfect result likely reflects an easily separable synthetic dataset and should not be read as real-world accuracy.
- The API has no authentication, so do not expose it publicly without protection.

## Future scope

- Add features such as transaction time, user history, velocity and device or IP information
- Retrain automatically from admin-reviewed alerts (confirmed fraud and false positives) in the main application
- Add API authentication and request validation
- Compare other models (Gradient Boosting, XGBoost) and report cross-validated scores

---
