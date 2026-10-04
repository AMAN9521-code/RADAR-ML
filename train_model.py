import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

DATA_FILE = "radar_ml_training_dataset.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# =========================================================
# 2. SELECT FEATURES AND TARGET
# =========================================================

features = [
    "Amount",
    "Merchant",
    "Location"
]

target = "Fraud"

X = df[features]
y = df[target]


# =========================================================
# 3. SPLIT TRAINING AND TEST DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 4. PREPROCESS CATEGORICAL FEATURES
# =========================================================

categorical_features = [
    "Merchant",
    "Location"
]

numeric_features = [
    "Amount"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# =========================================================
# 5. CREATE RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)


# =========================================================
# 6. CREATE COMPLETE ML PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# =========================================================
# 7. TRAIN MODEL
# =========================================================

print()
print("Training Random Forest model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# =========================================================
# 8. TEST MODEL
# =========================================================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print("========================================")
print("MODEL EVALUATION")
print("========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print()
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Fraud"
        ]
    )
)


# =========================================================
# 9. SAVE TRAINED MODEL
# =========================================================

MODEL_FILE = "radar_fraud_model.joblib"

joblib.dump(
    pipeline,
    MODEL_FILE
)

print()
print("========================================")
print("MODEL SAVED")
print("========================================")
print(
    f"Saved as: {MODEL_FILE}"
)