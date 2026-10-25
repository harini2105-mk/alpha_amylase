import pandas as pd
import numpy as np
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = "data/processed/alpha_amylase_train.csv"

VALIDATION_FILE = "data/validation/experimental_results.csv"

TEST_FILE = "data/processed/alpha_amylase_test.csv"

MODEL_DIR = "models"

UPDATED_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "alpha_amylase_updated_model.pkl"
)

UPDATED_PREDICTION_FILE = os.path.join(
    MODEL_DIR,
    "updated_test_predictions.csv"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD ORIGINAL TRAINING DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)


# ============================================================
# LOAD EXPERIMENTAL VALIDATION DATA
# ============================================================

if not os.path.exists(
    VALIDATION_FILE
):

    print(
        "Experimental validation file not found."
    )

    exit()


validation_df = pd.read_csv(
    VALIDATION_FILE
)


# ============================================================
# CHECK VALIDATION DATA
# ============================================================

required_columns = [
    "mutation",
    "predicted_target",
    "experimental_target"
]


for column in required_columns:

    if column not in validation_df.columns:

        print(
            f"Missing required column: {column}"
        )

        exit()


# ============================================================
# CHECK WHETHER EXPERIMENTAL DATA EXISTS
# ============================================================

if len(validation_df) == 0:

    print(
        "No experimental results are available."
    )

    print(
        "Add real experimental measurements before retraining."
    )

    exit()


# ============================================================
# REMOVE INVALID EXPERIMENTAL VALUES
# ============================================================

validation_df[
    "experimental_target"
] = pd.to_numeric(
    validation_df[
        "experimental_target"
    ],
    errors="coerce"
)


validation_df = validation_df.dropna(
    subset=[
        "experimental_target"
    ]
).copy()


if len(validation_df) == 0:

    print(
        "No valid experimental target values were found."
    )

    exit()


# ============================================================
# FIND MUTATION SEQUENCES
# ============================================================

mutation_df = pd.read_csv(
    "data/processed/alpha_amylase_mutation_screen.csv"
)


# Keep only required columns

mutation_sequences = mutation_df[
    [
        "mutation",
        "sequence"
    ]
].drop_duplicates(
    subset=["mutation"]
)


# ============================================================
# MERGE EXPERIMENTAL RESULTS WITH SEQUENCES
# ============================================================

validation_with_sequences = validation_df.merge(
    mutation_sequences,
    on="mutation",
    how="left"
)


# ============================================================
# REMOVE MISSING SEQUENCES
# ============================================================

validation_with_sequences = (
    validation_with_sequences
    .dropna(
        subset=[
            "sequence",
            "experimental_target"
        ]
    )
    .copy()
)


if len(validation_with_sequences) == 0:

    print(
        "No matching mutation sequences were found."
    )

    print(
        "Make sure the mutation names match the "
        "mutation screening file."
    )

    exit()


# ============================================================
# PREPARE ORIGINAL TRAINING DATA
# ============================================================

X_train_original = (
    train_df[
        "sequence"
    ]
    .astype(str)
)


y_train_original = (
    train_df[
        "target"
    ]
    .astype(float)
)


# ============================================================
# PREPARE EXPERIMENTAL FEEDBACK DATA
# ============================================================

X_feedback = (
    validation_with_sequences[
        "sequence"
    ]
    .astype(str)
)


y_feedback = (
    validation_with_sequences[
        "experimental_target"
    ]
    .astype(float)
)


# ============================================================
# COMBINE ORIGINAL + EXPERIMENTAL DATA
# ============================================================

X_combined = pd.concat(
    [
        X_train_original,
        X_feedback
    ],
    ignore_index=True
)


y_combined = pd.concat(
    [
        y_train_original,
        y_feedback
    ],
    ignore_index=True
)


print(
    "Original training samples:",
    len(X_train_original)
)


print(
    "Experimental feedback samples:",
    len(X_feedback)
)


print(
    "Combined training samples:",
    len(X_combined)
)


# ============================================================
# BUILD UPDATED MODEL
# ============================================================

updated_model = Pipeline(
    [
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
            Ridge(
                alpha=1.0
            )
        )
    ]
)


# ============================================================
# TRAIN UPDATED MODEL
# ============================================================

updated_model.fit(
    X_combined,
    y_combined
)


# ============================================================
# TEST UPDATED MODEL
# ============================================================

test_df = pd.read_csv(
    TEST_FILE
)


X_test = (
    test_df[
        "sequence"
    ]
    .astype(str)
)


y_test = (
    test_df[
        "target"
    ]
    .astype(float)
)


updated_predictions = updated_model.predict(
    X_test
)


# ============================================================
# CALCULATE PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_test,
    updated_predictions
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        updated_predictions
    )
)


r2 = r2_score(
    y_test,
    updated_predictions
)


# ============================================================
# SAVE UPDATED PREDICTIONS
# ============================================================

updated_results = test_df[
    [
        "sequence",
        "target",
        "set",
        "validation"
    ]
].copy()


updated_results[
    "predicted_target"
] = updated_predictions


updated_results[
    "absolute_error"
] = abs(
    updated_results[
        "target"
    ]
    -
    updated_results[
        "predicted_target"
    ]
)


updated_results.to_csv(
    UPDATED_PREDICTION_FILE,
    index=False
)


# ============================================================
# SAVE UPDATED MODEL
# ============================================================

joblib.dump(
    updated_model,
    UPDATED_MODEL_FILE
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print(
    "============================================"
)

print(
    "UPDATED MODEL TRAINING COMPLETED"
)

print(
    "============================================"
)

print(
    f"Original training samples : "
    f"{len(X_train_original)}"
)

print(
    f"Experimental feedback     : "
    f"{len(X_feedback)}"
)

print(
    f"Combined training samples : "
    f"{len(X_combined)}"
)

print()

print(
    f"Updated Model MAE  : {mae:.6f}"
)

print(
    f"Updated Model RMSE : {rmse:.6f}"
)

print(
    f"Updated Model R²   : {r2:.6f}"
)

print()

print(
    "Updated model saved to:"
)

print(
    UPDATED_MODEL_FILE
)

print()

print(
    "Updated predictions saved to:"
)

print(
    UPDATED_PREDICTION_FILE
)