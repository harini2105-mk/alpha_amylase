
import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# AI-DRIVEN NOVEL BIO-ENZYMES: ALPHA-AMYLASE
# =========================================================

st.set_page_config(
    page_title="AI-Driven Novel Bio-Enzymes",
    page_icon="🧬",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLEAN_PATH = os.path.join(DATA_DIR, "alpha_amylase_clean.csv")
MUTATION_PATH = os.path.join(DATA_DIR, "alpha_amylase_mutation_screen.csv")
PREDICTIONS_PATH = os.path.join(MODEL_DIR, "test_predictions.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "alpha_amylase_model.pkl")

PDB_URL = "https://files.rcsb.org/download/1BAG.pdb"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


clean_df = load_csv(CLEAN_PATH)
mutation_df = load_csv(MUTATION_PATH)
test_predictions = load_csv(PREDICTIONS_PATH)
saved_model = load_model(MODEL_PATH)


def find_column(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def get_model_parts(obj):
    if obj is None:
        return None, None

    if hasattr(obj, "predict"):
        return obj, None

    if isinstance(obj, dict):
        for key in ("pipeline", "model", "best_model", "estimator"):
            candidate = obj.get(key)
            if hasattr(candidate, "predict"):
                return candidate, None

        vectorizer = None
        regressor = None

        for key in ("vectorizer", "tfidf", "tfidf_vectorizer"):
            if hasattr(obj.get(key), "transform"):
                vectorizer = obj[key]
                break

        for key in ("regressor", "ridge", "regression_model"):
            if hasattr(obj.get(key), "predict"):
                regressor = obj[key]
                break

        if vectorizer is not None and regressor is not None:
            return regressor, vectorizer

    return None, None


model, vectorizer = get_model_parts(saved_model)


def clean_sequence(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    return "".join(aa for aa in value if aa in AMINO_ACIDS)


def predict_sequences(sequences):
    if model is None:
        raise RuntimeError(
            "The saved model could not be loaded. Check models/"
            "alpha_amylase_model.pkl."
        )

    if vectorizer is None:
        values = model.predict(list(sequences))
    else:
        values = model.predict(vectorizer.transform(list(sequences)))

    return np.asarray(values, dtype=float).reshape(-1)


# =========================================================
# REFERENCE SEQUENCE
# =========================================================

sequence_col = find_column(clean_df, ["sequence", "mutated_sequence"])

if sequence_col and not clean_df.empty:
    clean_df[sequence_col] = clean_df[sequence_col].apply(clean_sequence)
    usable = clean_df[clean_df[sequence_col].str.len() > 0]

    if not usable.empty:
        reference_sequence = usable.loc[
            usable[sequence_col].str.len().idxmax(), sequence_col
        ]
    else:
        reference_sequence = ""
else:
    reference_sequence = ""

reference_length = len(reference_sequence)


# =========================================================
# PREPARE MUTATION DATA
# =========================================================

def prepare_mutations(df):
    df = df.copy()

    if df.empty:
        return df

    rename = {}
    aliases = {
        "mutation": ["mutation", "mutant_name"],
        "sequence": ["sequence", "mutated_sequence"],
        "position": ["position", "mutation_position"],
        "predicted_target": [
            "predicted_target", "prediction", "predicted_score"
        ]
    }

    for target, alternatives in aliases.items():
        source = find_column(df, alternatives)
        if source and source != target:
            rename[source] = target

    df = df.rename(columns=rename)

    if "sequence" in df.columns:
        df["sequence"] = df["sequence"].apply(clean_sequence)

    if "mutation" not in df.columns:
        return df

    df["mutation"] = df["mutation"].astype(str).str.strip().str.upper()

    parsed = df["mutation"].str.extract(
        r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)"
        r"([ACDEFGHIKLMNPQRSTVWY])$"
    )

    if "position" not in df.columns:
        df["position"] = pd.to_numeric(parsed[1], errors="coerce")

    if "predicted_target" in df.columns:
        df["predicted_target"] = pd.to_numeric(
            df["predicted_target"], errors="coerce"
        )

    # If no scores exist, predict the available mutant sequences.
    if (
        "sequence" in df.columns
        and model is not None
        and (
            "predicted_target" not in df.columns
            or df["predicted_target"].isna().all()
        )
    ):
        try:
            df["predicted_target"] = predict_sequences(
                df["sequence"].tolist()
            )
        except Exception:
            pass

    return df


mutation_df = prepare_mutations(mutation_df)


# =========================================================
# MUTATION GENERATION
# =========================================================

def generate_single_mutations(reference):
    rows = []

    for position, original_aa in enumerate(reference, start=1):
        for replacement_aa in AMINO_ACIDS:
            if replacement_aa == original_aa:
                continue

            mutated_sequence = (
                reference[:position - 1]
                + replacement_aa
                + reference[position:]
            )

            rows.append({
                "mutation": f"{original_aa}{position}{replacement_aa}",
                "position": position,
                "wild_type": original_aa,
                "replacement": replacement_aa,
                "sequence": mutated_sequence
            })

    return pd.DataFrame(rows)


def get_scored_candidates(use_generated):
    if use_generated:
        candidates = generate_single_mutations(reference_sequence)
    else:
        if mutation_df.empty or "mutation" not in mutation_df.columns:
            return pd.DataFrame()

        candidates = mutation_df.copy()

        if "sequence" not in candidates.columns:
            candidates["sequence"] = ""

        # Fill missing sequences from the reference when possible.
        for index, row in candidates.iterrows():
            if not row["sequence"]:
                match = re.fullmatch(
                    r"([ACDEFGHIKLMNPQRSTVWY])(\d+)"
                    r"([ACDEFGHIKLMNPQRSTVWY])",
                    str(row["mutation"])
                )

                if match:
                    old_aa, pos_text, new_aa = match.groups()
                    pos = int(pos_text)

                    if (
                        1 <= pos <= len(reference_sequence)
                        and reference_sequence[pos - 1] == old_aa
                    ):
                        candidates.at[index, "sequence"] = (
                            reference_sequence[:pos - 1]
                            + new_aa
                            + reference_sequence[pos:]
                        )

    if candidates.empty:
        return candidates

    if model is None:
        return pd.DataFrame()

    # Recalculate predictions with the currently loaded model.
    candidates = candidates[
        candidates["sequence"].astype(str).str.len() > 0
    ].copy()

    if candidates.empty:
        return candidates

    try:
        candidates["predicted_target"] = predict_sequences(
            candidates["sequence"].tolist()
        )
    except Exception:
        return pd.DataFrame()

    return candidates


# =========================================================
# RANKING AND RECOMMENDATION
# =========================================================

def rank_candidates(df, objective):
    if df.empty or "predicted_target" not in df.columns:
        return df

    ascending = objective == "Lower is better"

    result = df.sort_values(
        "predicted_target", ascending=ascending
    ).reset_index(drop=True)

    result["Rank"] = np.arange(1, len(result) + 1)
    return result


def recommend_mutation(score, reference_score, rank, total, objective):
    if objective == "Higher is better":
        improvement = score - reference_score
    else:
        improvement = reference_score - score

    if total > 0:
        percentile = 100 * (1 - (rank - 1) / total)
    else:
        percentile = 0

    if improvement > 0 and percentile >= 75:
        return (
            "Promising computational candidate",
            "Predicted score improves in the selected direction and "
            "the candidate is in the top quarter of the ranked screen."
        )

    if improvement > 0:
        return (
            "Consider for further testing",
            "Predicted score improves in the selected direction. "
            "Experimental validation is required."
        )

    if improvement < 0:
        return (
            "Lower priority",
            "Predicted score moves away from the selected objective."
        )

    return (
        "Uncertain",
        "No predicted improvement over the reference was detected."
    )


# =========================================================
# 3D REFERENCE STRUCTURE
# =========================================================

def render_structure(position=None, mutation_name="Reference structure"):
    positions = [int(position)] if position else []
    html = """
    <div style="font-family:Arial,sans-serif">
      <h4 id="protein-title"></h4>
      <p style="font-size:13px;color:#555">
        Reference structure PDB 1BAG. Highlighting indicates a reference
        residue only, not a predicted mutant-specific structure.
      </p>
      <div id="protein-viewer"
           style="height:470px;width:100%;border:1px solid #ddd;
                  border-radius:10px"></div>
      <p id="protein-status"></p>
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      const pdbUrl = "https://files.rcsb.org/download/1BAG.pdb";
      const selectedPositions = __POSITIONS__;
      document.getElementById("protein-title").textContent = __TITLE__;
      const status = document.getElementById("protein-status");
      const element = document.getElementById("protein-viewer");

      if (typeof $3Dmol === "undefined") {
        status.textContent = "3Dmol.js could not load. Check your internet.";
      } else {
        const viewer = $3Dmol.createViewer(
          element, {backgroundColor: "white"}
        );

        fetch(pdbUrl)
          .then(response => {
            if (!response.ok) throw new Error("Could not load PDB structure.");
            return response.text();
          })
          .then(data => {
            viewer.addModel(data, "pdb");
            viewer.setStyle({}, {cartoon: {color: "spectrum"}});

            selectedPositions.forEach(position => {
              viewer.setStyle(
                {chain: "A", resi: position},
                {cartoon: {color: "orange"},
                 stick: {color: "orange", radius: 0.25}}
              );
            });

            viewer.zoomTo();
            viewer.render();
            viewer.resize();

            status.textContent = selectedPositions.length
              ? "Highlighted reference residue: " + selectedPositions.join(", ")
              : "Reference structure loaded.";
          })
          .catch(error => {
            status.textContent = "Structure error: " + error.message;
          });
      }
    </script>
    """

    html = html.replace("__POSITIONS__", json.dumps(positions))
    html = html.replace(
        "__TITLE__", json.dumps(mutation_name)
    )

    components.html(html, height=550, scrolling=False)


# =========================================================
# HEADER AND SIDEBAR
# =========================================================

st.title("🧬 AI-Driven Novel Bio-Enzymes")
st.caption("Alpha-amylase prototype | Condition inputs | Mutation ranking")

st.info(
    "Temperature and pH are provided as screening conditions. The current "
    "sequence-only model does not learn their effects unless it was trained "
    "on condition-specific experimental data."
)

page = st.sidebar.radio(
    "Choose a page",
    [
        "Project Overview",
        "Condition-Based Mutation Screening",
        "Dataset Mutation Selection",
        "Custom Mutation Evaluation",
        "Mutation Analysis",
        "Model Evaluation",
        "Experimental Validation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("### Files")
st.sidebar.write(
    f"Reference sequence: {'Loaded' if reference_sequence else 'Missing'}"
)
st.sidebar.write(
    f"Mutation dataset: {'Loaded' if not mutation_df.empty else 'Missing'}"
)
st.sidebar.write(
    f"Model: {'Loaded' if model is not None else 'Missing'}"
)


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Project Overview":
    st.header("Project Overview")

    st.write(
        "This project explores how machine learning can help prioritize "
        "enzyme sequence variants for experimental testing. The current "
        "prototype focuses on alpha-amylase."
    )

    st.subheader("Workflow")
    steps = [
        "Sequence Data",
        "Preprocessing",
        "TF-IDF Features",
        "Ridge Regression",
        "Model Evaluation",
        "Mutation Generation",
        "Prediction",
        "Ranking",
        "Recommendation",
        "3D Visualization",
        "Experimental Validation"
    ]

    cols = st.columns(3)
    for i, step in enumerate(steps):
        with cols[i % 3]:
            st.container(border=True).markdown(f"**{i + 1}. {step}**")

    a, b, c = st.columns(3)
    a.metric("Dataset rows", len(clean_df))
    b.metric("Saved mutations", len(mutation_df))
    c.metric("Reference length", reference_length)

    st.warning(
        "The reference is currently the longest sequence in the cleaned "
        "dataset. Verify that it is the biological wild-type sequence."
    )


# =========================================================
# PAGE 2: CONDITION-BASED SCREENING
# =========================================================

elif page == "Condition-Based Mutation Screening":
    st.header("Condition-Based Mutation Screening")
    st.write(
        "Choose your desired experimental conditions and objective. "
        "The app generates candidates, predicts their sequence-based scores, "
        "ranks them, and produces computational recommendations."
    )

    with st.form("condition_form"):
        a, b = st.columns(2)

        with a:
            temperature = st.slider(
                "Temperature (°C)", 20, 100, 70, 5
            )
            ph = st.slider(
                "pH", 3.0, 11.0, 5.5, 0.5
            )
            substrate = st.selectbox(
                "Substrate",
                ["Starch", "Soluble starch", "Amylose",
                 "Amylopectin", "Other / unspecified"]
            )

        with b:
            solvent = st.selectbox(
                "Solvent condition",
                ["Aqueous / buffer", "Ethanol-containing",
                 "Organic solvent", "Other / unspecified"]
            )
            cofactor = st.selectbox(
                "Cofactor condition",
                ["Not specified", "Calcium present",
                 "Calcium absent", "Other / unspecified"]
            )
            objective = st.radio(
                "Optimization objective",
                ["Higher is better", "Lower is better"]
            )

        source = st.radio(
            "Candidate source",
            [
                "Use saved mutation dataset",
                "Generate all single mutations from reference"
            ]
        )

        max_candidates = st.selectbox(
            "Maximum candidates to rank",
            [10, 25, 50, 100, 250, 500],
            index=2
        )

        run_screen = st.form_submit_button("Generate and Rank Mutations")

    if run_screen:
        st.session_state["condition_settings"] = {
            "temperature": temperature,
            "pH": ph,
            "substrate": substrate,
            "solvent": solvent,
            "cofactor": cofactor,
            "objective": objective,
            "source": source
        }

        use_generated = (
            source == "Generate all single mutations from reference"
        )

        with st.spinner("Generating and scoring mutation candidates..."):
            candidates = get_scored_candidates(use_generated)

        if candidates.empty:
            st.error(
                "No candidate predictions could be produced. Check the "
                "saved model, reference sequence, and mutation dataset."
            )
        else:
            ranked = rank_candidates(candidates, objective)
            ranked = ranked.head(max_candidates).copy()

            reference_score = None
            if reference_sequence:
                try:
                    reference_score = float(
                        predict_sequences([reference_sequence])[0]
                    )
                except Exception:
                    pass

            if reference_score is not None:
                recommendations = []
                explanations = []

                for _, row in ranked.iterrows():
                    rec, explanation = recommend_mutation(
                        float(row["predicted_target"]),
                        reference_score,
                        int(row["Rank"]),
                        len(candidates),
                        objective
                    )
                    recommendations.append(rec)
                    explanations.append(explanation)

                ranked["Recommendation"] = recommendations
                ranked["Reason"] = explanations

            # Save results and the user's selected conditions for display.
            st.session_state["ranked_candidates"] = ranked
            st.session_state["screen_reference_score"] = reference_score

    if "condition_settings" in st.session_state:
        settings = st.session_state["condition_settings"]

        st.subheader("Selected screening conditions")
        a, b, c = st.columns(3)
        a.metric("Temperature", f"{settings['temperature']} °C")
        b.metric("pH", f"{settings['pH']:.1f}")
        c.metric("Objective", settings["objective"])

        st.write(
            f"**Substrate:** {settings['substrate']}  \n"
            f"**Solvent:** {settings['solvent']}  \n"
            f"**Cofactor:** {settings['cofactor']}"
        )

        st.warning(
            "These conditions are recorded in the screening workflow, but "
            "the current model predicts from amino-acid sequence alone. "
            "Changing temperature or pH does not change the model's score "
            "unless the model was trained with these condition features or "
            "condition-specific measurements."
        )

    if "ranked_candidates" in st.session_state:
        ranked = st.session_state["ranked_candidates"]

        if not ranked.empty:
            st.subheader("Ranked Mutation Candidates")

            display_cols = [
                c for c in [
                    "Rank", "mutation", "position", "predicted_target",
                    "Recommendation", "Reason"
                ]
                if c in ranked.columns
            ]

            st.dataframe(
                ranked[display_cols],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download ranked mutations CSV",
                data=ranked.to_csv(index=False).encode("utf-8"),
                file_name="ranked_alpha_amylase_mutations.csv",
                mime="text/csv"
            )

            st.subheader("Inspect a ranked mutation")
            chosen = st.selectbox(
                "Choose mutation to inspect",
                ranked["mutation"].astype(str).tolist(),
                key="inspect_ranked_mutation"
            )

            row = ranked[
                ranked["mutation"].astype(str) == chosen
            ].iloc[0]

            a, b, c = st.columns(3)
            a.metric("Rank", int(row["Rank"]))
            b.metric("Mutation", chosen)
            c.metric(
                "Predicted target",
                f"{float(row['predicted_target']):.5f}"
            )

            if "Recommendation" in row.index:
                st.markdown(f"**Recommendation:** {row['Recommendation']}")
                st.write(row["Reason"])

            match = re.fullmatch(
                r"([ACDEFGHIKLMNPQRSTVWY])(\d+)"
                r"([ACDEFGHIKLMNPQRSTVWY])",
                chosen
            )

            position = int(match.group(2)) if match else None

            st.subheader("3D reference structure")
            render_structure(position, f"{chosen} reference position")

            st.caption(
                "This is a reference-structure view with the mutation site "
                "highlighted. It is not a mutant-specific structural prediction."
            )


# =========================================================
# PAGE 3: SELECT MUTATION FROM SAVED DATASET
# =========================================================

elif page == "Dataset Mutation Selection":
    st.header("Dataset Mutation Selection")

    if mutation_df.empty or "mutation" not in mutation_df.columns:
        st.error("Mutation dataset is missing or has no mutation column.")
    elif model is None:
        st.error("The saved model could not be loaded.")
    else:
        objective = st.radio(
            "Ranking objective",
            ["Higher is better", "Lower is better"],
            horizontal=True,
            key="dataset_objective"
        )

        candidates = get_scored_candidates(False)
        ranked = rank_candidates(candidates, objective)

        if ranked.empty:
            st.error("No usable mutation predictions were found.")
        else:
            selected_mutation = st.selectbox(
                "Select mutation",
                ranked["mutation"].astype(str).tolist()
            )

            row = ranked[
                ranked["mutation"].astype(str) == selected_mutation
            ].iloc[0]

            rank = int(row["Rank"])
            score = float(row["predicted_target"])
            total = len(ranked)

            reference_score = float(
                predict_sequences([reference_sequence])[0]
            ) if reference_sequence else None

            st.subheader("Mutation results")
            a, b, c = st.columns(3)
            a.metric("Mutation", selected_mutation)
            b.metric("Rank", f"{rank} / {total}")
            c.metric("Predicted target", f"{score:.5f}")

            if reference_score is not None:
                rec, explanation = recommend_mutation(
                    score, reference_score, rank, total, objective
                )
                st.markdown(f"**Recommendation:** {rec}")
                st.write(explanation)
                st.write(f"Reference predicted target: {reference_score:.5f}")
                st.write(f"Mutant minus reference: {score-reference_score:+.5f}")

            match = re.fullmatch(
                r"([ACDEFGHIKLMNPQRSTVWY])(\d+)"
                r"([ACDEFGHIKLMNPQRSTVWY])",
                selected_mutation
            )

            position = int(match.group(2)) if match else None
            render_structure(position, f"{selected_mutation} reference position")

            report = {
                "mutation": selected_mutation,
                "rank": rank,
                "total_candidates": total,
                "predicted_target": score,
                "reference_predicted_target": reference_score,
                "recommendation": rec if reference_score is not None else None,
                "temperature_pH_note": (
                    "Scores are sequence-based unless condition-specific "
                    "training data were used."
                )
            }

            st.download_button(
                "Download mutation report JSON",
                data=json.dumps(report, indent=2),
                file_name=f"{selected_mutation}_report.json",
                mime="application/json"
            )


# =========================================================
# PAGE 4: CUSTOM MUTATION
# =========================================================

elif page == "Custom Mutation Evaluation":
    st.header("Custom Mutation Evaluation")

    if not reference_sequence:
        st.error("Reference sequence is unavailable.")
    elif model is None:
        st.error("The saved model could not be loaded.")
    else:
        st.write(
            "Enter a single substitution such as T125V. The first letter "
            "must match the reference residue at the stated position."
        )

        with st.form("custom_form"):
            mutation_text = st.text_input(
                "Mutation", placeholder="Example: T125V"
            )
            objective = st.radio(
                "Optimization objective",
                ["Higher is better", "Lower is better"],
                horizontal=True
            )
            submitted = st.form_submit_button("Evaluate Mutation")

        if submitted:
            text = mutation_text.strip().upper()
            match = re.fullmatch(
                r"([ACDEFGHIKLMNPQRSTVWY])(\d+)"
                r"([ACDEFGHIKLMNPQRSTVWY])",
                text
            )

            if not match:
                st.error(
                    "Invalid format. Use a mutation such as T125V. "
                    "X is not a valid replacement amino acid."
                )
            else:
                old_aa, pos_text, new_aa = match.groups()
                position = int(pos_text)

                if position < 1 or position > reference_length:
                    st.error(
                        f"Position must be between 1 and {reference_length}."
                    )
                elif reference_sequence[position - 1] != old_aa:
                    actual = reference_sequence[position - 1]
                    st.error(
                        f"Reference mismatch: position {position} contains "
                        f"{actual}, not {old_aa}. Try {actual}{position}{new_aa}."
                    )
                elif old_aa == new_aa:
                    st.error("Choose a replacement different from the original.")
                else:
                    mutated = (
                        reference_sequence[:position - 1]
                        + new_aa
                        + reference_sequence[position:]
                    )

                    try:
                        reference_score = float(
                            predict_sequences([reference_sequence])[0]
                        )
                        mutant_score = float(
                            predict_sequences([mutated])[0]
                        )

                        candidates = get_scored_candidates(False)
                        ranked = rank_candidates(candidates, objective)

                        rank = None
                        if not ranked.empty:
                            matched = ranked[
                                ranked["mutation"].astype(str) == text
                            ]
                            if not matched.empty:
                                rank = int(matched.iloc[0]["Rank"])

                        st.subheader("Prediction results")
                        a, b, c = st.columns(3)
                        a.metric("Mutation", text)
                        b.metric("Reference score", f"{reference_score:.5f}")
                        c.metric("Mutant score", f"{mutant_score:.5f}")

                        delta = mutant_score - reference_score
                        st.write(f"**Mutant minus reference:** {delta:+.5f}")

                        if rank is not None:
                            st.metric("Saved mutation-screen rank", rank)
                        else:
                            st.info(
                                "This custom mutation has no rank in the "
                                "saved mutation screen. Its model score is shown."
                            )

                        rec, explanation = recommend_mutation(
                            mutant_score,
                            reference_score,
                            rank if rank is not None else 1,
                            len(ranked) if not ranked.empty else 1,
                            objective
                        )

                        st.markdown(f"**Recommendation:** {rec}")
                        st.write(explanation)

                        st.subheader("3D reference structure")
                        render_structure(position, f"{text} reference position")

                        report = {
                            "mutation": text,
                            "position": position,
                            "original_amino_acid": old_aa,
                            "replacement_amino_acid": new_aa,
                            "reference_score": reference_score,
                            "mutant_score": mutant_score,
                            "delta": delta,
                            "rank_in_saved_screen": rank,
                            "recommendation": rec
                        }

                        st.download_button(
                            "Download custom mutation report",
                            data=json.dumps(report, indent=2),
                            file_name=f"{text}_custom_report.json",
                            mime="application/json"
                        )

                    except Exception as exc:
                        st.error(f"Could not evaluate mutation: {exc}")


# =========================================================
# PAGE 5: MUTATION ANALYSIS
# =========================================================

elif page == "Mutation Analysis":
    st.header("Mutation Analysis")

    if not reference_sequence:
        st.error("Reference sequence is unavailable.")

    elif model is None:
        st.error("The saved model could not be loaded.")

    else:
        with st.form("mutation_analysis_form"):
            mutation_text = st.text_input(
                "Enter mutation",
                placeholder="Example: T125V"
            )

            submitted = st.form_submit_button("Analyze Mutation")

        if submitted:
            text = mutation_text.strip().upper()

            match = re.fullmatch(
                r"([ACDEFGHIKLMNPQRSTVWY])(\d+)"
                r"([ACDEFGHIKLMNPQRSTVWY])",
                text
            )

            if not match:
                st.error("Invalid mutation format. Example: T125V.")

            else:
                old_aa, pos_text, new_aa = match.groups()
                position = int(pos_text)

                if not 1 <= position <= reference_length:
                    st.error(
                        f"Position must be between 1 and {reference_length}."
                    )

                elif reference_sequence[position - 1] != old_aa:
                    actual = reference_sequence[position - 1]
                    st.error(
                        f"Position {position} contains {actual}. "
                        f"Use {actual}{position}{new_aa} instead."
                    )

                elif old_aa == new_aa:
                    st.error(
                        "Choose a replacement different from the original."
                    )

                else:
                    mutated_sequence = (
                        reference_sequence[:position - 1]
                        + new_aa
                        + reference_sequence[position:]
                    )

                    try:
                        reference_score = float(
                            predict_sequences([reference_sequence])[0]
                        )

                        mutant_score = float(
                            predict_sequences([mutated_sequence])[0]
                        )

                        st.subheader("Prediction Graph")

                        graph_data = pd.DataFrame({
                            "Sequence": ["Reference", text],
                            "Predicted Score": [
                                reference_score,
                                mutant_score
                            ]
                        })

                        st.bar_chart(
                            graph_data.set_index("Sequence")
                        )

                        st.subheader("Mutated Sequence")

                        st.text_area(
                            "Amino-acid sequence",
                            value=mutated_sequence,
                            height=150,
                            disabled=True
                        )

                    except Exception as exc:
                        st.error(
                            f"Could not analyze mutation: {exc}"
                        )
# =========================================================
# PAGE 6: MODEL EVALUATION
# =========================================================

elif page == "Model Evaluation":
    st.header("Model Evaluation")

    if model is None:
        st.error("The model could not be loaded.")
    else:
        st.success("Saved model loaded.")

    if test_predictions.empty:
        st.warning(
            "No saved test predictions found at models/test_predictions.csv."
        )
    else:
        st.dataframe(test_predictions.head(100), use_container_width=True)

        actual_col = find_column(
            test_predictions, ["actual", "y_true", "target", "true_target"]
        )
        pred_col = find_column(
            test_predictions, ["predicted", "y_pred", "prediction"]
        )

        if actual_col and pred_col:
            evaluation = test_predictions[[actual_col, pred_col]].copy()
            evaluation[actual_col] = pd.to_numeric(
                evaluation[actual_col], errors="coerce"
            )
            evaluation[pred_col] = pd.to_numeric(
                evaluation[pred_col], errors="coerce"
            )
            evaluation = evaluation.dropna()

            if len(evaluation) >= 2:
                actual = evaluation[actual_col]
                predicted = evaluation[pred_col]

                a, b, c = st.columns(3)
                a.metric(
                    "MAE",
                    f"{mean_absolute_error(actual, predicted):.4f}"
                )
                b.metric(
                    "RMSE",
                    f"{np.sqrt(mean_squared_error(actual, predicted)):.4f}"
                )
                c.metric("R²", f"{r2_score(actual, predicted):.4f}")
            else:
                st.info("Not enough valid records to calculate metrics.")
        else:
            st.info(
                "Could not identify actual and predicted columns. "
                "Check the columns in test_predictions.csv."
            )


# =========================================================
# PAGE 6: EXPERIMENTAL VALIDATION
# =========================================================

elif page == "Experimental Validation":
    st.header("Experimental Validation")
    st.write(
        "Record planned or completed experiments. These entries are "
        "downloadable records; this prototype does not automatically retrain "
        "the model from the submitted measurements."
    )

    with st.form("experiment_form"):
        candidate = st.text_input("Candidate mutation or ID")
        temperature = st.number_input(
            "Experimental temperature (°C)", 0.0, 150.0, 70.0
        )
        ph = st.number_input(
            "Experimental pH", 0.0, 14.0, 5.5, step=0.5
        )
        substrate = st.text_input("Substrate", value="Starch")
        replicates = st.number_input(
            "Number of replicates", 1, 100, 3
        )
        measurement = st.selectbox(
            "Measurement",
            [
                "Enzyme activity",
                "Thermal stability",
                "pH stability",
                "Substrate range",
                "Solvent tolerance",
                "Selectivity",
                "Manufacturing robustness",
                "Other"
            ]
        )
        status = st.selectbox(
            "Status",
            ["Not started", "Planned", "In progress", "Completed"]
        )
        result_notes = st.text_area(
            "Measured results and notes",
            placeholder="Include units, controls, replicates and observations."
        )

        submitted = st.form_submit_button("Prepare Validation Record")

    if submitted:
        record = {
            "candidate": candidate,
            "temperature_celsius": temperature,
            "pH": ph,
            "substrate": substrate,
            "replicates": int(replicates),
            "measurement": measurement,
            "status": status,
            "result_notes": result_notes
        }

        st.success("Validation record prepared.")
        st.json(record)

        st.download_button(
            "Download validation record",
            data=json.dumps(record, indent=2),
            file_name="alpha_amylase_validation.json",
            mime="application/json"
        )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.markdown("---")
st.sidebar.caption(
    "AI-Driven Novel Bio-Enzymes | Alpha-amylase prototype"
)
