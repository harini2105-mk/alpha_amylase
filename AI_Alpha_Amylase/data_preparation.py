import pandas as pd
import os

# ============================================================
# AI-DRIVEN NOVEL BIO-ENZYMES
# Alpha-Amylase Dataset Preparation
# ============================================================

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

INPUT_FILE = "data/by_mutation.csv.gz"

OUTPUT_DIR = "data/processed"

MASTER_FILE = os.path.join(
    OUTPUT_DIR,
    "alpha_amylase_clean.csv"
)

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "alpha_amylase_train.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "alpha_amylase_test.csv"
)


# ------------------------------------------------------------
# 2. Create output folder
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 3. Load the original dataset
# ------------------------------------------------------------

print("=" * 60)
print("LOADING ALPHA-AMYLASE DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)
print("Original columns:", df.columns.tolist())


# ------------------------------------------------------------
# 4. Check required columns
# ------------------------------------------------------------

required_columns = [
    "sequence",
    "target",
    "set",
    "validation"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from the dataset."
        )


# ------------------------------------------------------------
# 5. Keep only required columns
# ------------------------------------------------------------

df = df[required_columns].copy()


# ------------------------------------------------------------
# 6. Remove rows with missing values
# ------------------------------------------------------------

before_missing = len(df)

df = df.dropna(
    subset=["sequence", "target", "set"]
)

after_missing = len(df)

print()
print("Rows removed because of missing values:",
      before_missing - after_missing)


# ------------------------------------------------------------
# 7. Clean protein sequences
# ------------------------------------------------------------

df["sequence"] = (
    df["sequence"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ------------------------------------------------------------
# 8. Keep valid amino-acid sequences
# ------------------------------------------------------------

valid_amino_acids = set(
    "ACDEFGHIKLMNPQRSTVWY"
)

def valid_sequence(sequence):
    return (
        len(sequence) > 0
        and all(
            amino_acid in valid_amino_acids
            for amino_acid in sequence
        )
    )


df = df[
    df["sequence"].apply(valid_sequence)
].copy()


# ------------------------------------------------------------
# 9. Convert target to numeric
# ------------------------------------------------------------

df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce"
)

df = df.dropna(
    subset=["target"]
).copy()


# ------------------------------------------------------------
# 10. Remove duplicate sequences
# ------------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["sequence"],
    keep="first"
).copy()

after_duplicates = len(df)

print(
    "Duplicate sequences removed:",
    before_duplicates - after_duplicates
)


# ------------------------------------------------------------
# 11. Add sequence length
# ------------------------------------------------------------

df["sequence_length"] = (
    df["sequence"].str.len()
)


# ------------------------------------------------------------
# 12. Standardize dataset split names
# ------------------------------------------------------------

df["set"] = (
    df["set"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ------------------------------------------------------------
# 13. Display final dataset information
# ------------------------------------------------------------

print()
print("=" * 60)
print("CLEAN DATASET INFORMATION")
print("=" * 60)

print("Final dataset shape:", df.shape)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Dataset split:")
print(df["set"].value_counts())

print()
print("Target summary:")
print(df["target"].describe())

print()
print("Sequence length summary:")
print(df["sequence_length"].describe())


# ------------------------------------------------------------
# 14. Save cleaned master dataset
# ------------------------------------------------------------

df.to_csv(
    MASTER_FILE,
    index=False
)


# ------------------------------------------------------------
# 15. Create training dataset
# ------------------------------------------------------------

train_df = df[
    df["set"] == "train"
].copy()


# ------------------------------------------------------------
# 16. Create test dataset
# ------------------------------------------------------------

test_df = df[
    df["set"] == "test"
].copy()


# ------------------------------------------------------------
# 17. Save train and test datasets
# ------------------------------------------------------------

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ------------------------------------------------------------
# 18. Final report
# ------------------------------------------------------------

print()
print("=" * 60)
print("DATA PREPARATION COMPLETED")
print("=" * 60)

print()
print("Clean master dataset:")
print(MASTER_FILE)

print()
print("Training dataset:")
print(TRAIN_FILE)

print("Training rows:", len(train_df))

print()
print("Test dataset:")
print(TEST_FILE)

print("Test rows:", len(test_df))

print()
print("Total cleaned rows:", len(df))

print()
print("Sample cleaned data:")
print(
    df.head(5).to_string(index=False)
)

print()
print("=" * 60)
print("READY FOR MODEL TRAINING")
print("=" * 60)