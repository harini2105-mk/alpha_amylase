import pandas as pd
import joblib
import os

# ============================================================
# AI-DRIVEN NOVEL BIO-ENZYMES
# Alpha-Amylase Single Mutation Screening
# ============================================================

MODEL_FILE = "models/alpha_amylase_model.pkl"

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "alpha_amylase_mutation_screen.csv"
)

# ============================================================
# 1. Load trained model
# ============================================================

print("=" * 60)
print("ALPHA-AMYLASE MUTATION SCREENING")
print("=" * 60)

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# 2. Load reference sequence
# ============================================================
#
# We use the longest sequence in the experimental dataset
# as the reference sequence for this prototype.
# ============================================================

print("\nLoading experimental dataset...")

df = pd.read_csv(
    "data/processed/alpha_amylase_clean.csv"
)

reference_sequence = (
    df.loc[df["sequence_length"].idxmax(), "sequence"]
)

print(
    "Reference sequence length:",
    len(reference_sequence)
)


# ============================================================
# 3. Amino acids
# ============================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# ============================================================
# 4. Generate all single mutations
# ============================================================

print("\nGenerating single mutations...")

mutations = []

for position, original_amino_acid in enumerate(
    reference_sequence,
    start=1
):

    for new_amino_acid in AMINO_ACIDS:

        # Do not mutate an amino acid into itself
        if new_amino_acid == original_amino_acid:
            continue

        mutated_sequence = (
            reference_sequence[:position - 1]
            + new_amino_acid
            + reference_sequence[position:]
        )

        mutations.append({
            "mutation": (
                f"{original_amino_acid}"
                f"{position}"
                f"{new_amino_acid}"
            ),
            "position": position,
            "wild_type": original_amino_acid,
            "mutant": new_amino_acid,
            "sequence": mutated_sequence
        })


# ============================================================
# 5. Convert to DataFrame
# ============================================================

mutation_df = pd.DataFrame(mutations)

print(
    "Total mutations generated:",
    len(mutation_df)
)


# ============================================================
# 6. Predict target for every mutation
# ============================================================

print("\nPredicting experimental target...")

mutation_df["predicted_target"] = model.predict(
    mutation_df["sequence"]
)


# ============================================================
# 7. Rank candidates
# ============================================================

mutation_df = mutation_df.sort_values(
    by="predicted_target",
    ascending=False
).reset_index(drop=True)


mutation_df["rank"] = (
    mutation_df.index + 1
)


# ============================================================
# 8. Save results
# ============================================================

mutation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 9. Display top candidates
# ============================================================

print()
print("=" * 60)
print("TOP 20 PREDICTED CANDIDATES")
print("=" * 60)

print(
    mutation_df[
        [
            "rank",
            "mutation",
            "position",
            "wild_type",
            "mutant",
            "predicted_target"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 10. Final information
# ============================================================

print()
print("=" * 60)
print("MUTATION SCREENING COMPLETED")
print("=" * 60)

print(
    "\nTotal single mutations screened:",
    len(mutation_df)
)

print(
    "\nResults saved to:"
)

print(OUTPUT_FILE)

print()
print("The ranked mutation list is ready for the Streamlit UI.")

print("=" * 60)