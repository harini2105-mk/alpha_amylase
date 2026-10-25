import os
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# ============================================================
# AI-DRIVEN NOVEL BIO-ENZYMES
# Alpha-Amylase ML Prediction Model
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

TRAIN_FILE = "data/processed/alpha_amylase_train.csv"
TEST_FILE = "data/processed/alpha_amylase_test.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "alpha_amylase_model.pkl"
)

PREDICTION_FILE = os.path.join(
    MODEL_DIR,
    "test_predictions.csv"
)


# ------------------------------------------------------------
# 2. Create model folder
# ------------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------------------------------------------
# 3. Load training and test data
# ------------------------------------------------------------

print("=" * 60)
print("ALPHA-AMYLASE ML MODEL")
print("=" * 60)

print("\nLoading training dataset...")

train_df = pd.read_csv(TRAIN_FILE)

print("Training rows:", len(train_df))


print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print("Test rows:", len(test_df))


# ------------------------------------------------------------
# 4. Prepare input and target
# ------------------------------------------------------------

X_train = train_df["sequence"].astype(str)
y_train = train_df["target"].astype(float)

X_test = test_df["sequence"].astype(str)
y_test = test_df["target"].astype(float)


# ------------------------------------------------------------
# 5. Create ML pipeline
# ------------------------------------------------------------

print("\nCreating ML model...")

model = Pipeline([
    
    (
        "features",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),
            lowercase=False,
            min_df=2,
            max_features=10000
        )
    ),

    (
        "regressor",
        Ridge(alpha=1.0)
    )
])


# ------------------------------------------------------------
# 6. Train the model
# ------------------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ------------------------------------------------------------
# 7. Predict test data
# ------------------------------------------------------------

print("\nPredicting test dataset...")

predictions = model.predict(X_test)


# ------------------------------------------------------------
# 8. Calculate evaluation metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# ------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")


# ------------------------------------------------------------
# 10. Save predictions
# ------------------------------------------------------------

results = test_df[
    ["sequence", "target", "set", "validation"]
].copy()

results["predicted_target"] = predictions

results["absolute_error"] = (
    abs(
        results["target"]
        - results["predicted_target"]
    )
)

results.to_csv(
    PREDICTION_FILE,
    index=False
)


# ------------------------------------------------------------
# 11. Save trained model
# ------------------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)


# ------------------------------------------------------------
# 12. Display sample predictions
# ------------------------------------------------------------

print()
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    results[
        [
            "target",
            "predicted_target",
            "absolute_error"
        ]
    ].head(10).to_string(index=False)
)


# ------------------------------------------------------------
# 13. Final information
# ------------------------------------------------------------

print()
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print()
print("Trained model saved at:")
print(MODEL_FILE)

print()
print("Test predictions saved at:")
print(PREDICTION_FILE)

print()
print("Training samples:", len(train_df))
print("Testing samples :", len(test_df))

print()
print("The model is ready for the Streamlit application.")
print("=" * 60)