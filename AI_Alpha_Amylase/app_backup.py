import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Driven Novel Bio-Enzymes",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_FILE = "data/processed/alpha_amylase_clean.csv"
MUTATION_FILE = "data/processed/alpha_amylase_mutation_screen.csv"
MODEL_FILE = "models/alpha_amylase_model.pkl"
PREDICTION_FILE = "models/test_predictions.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)
    mutations = pd.read_csv(MUTATION_FILE)

    if os.path.exists(PREDICTION_FILE):
        predictions = pd.read_csv(PREDICTION_FILE)
    else:
        predictions = pd.DataFrame()

    return data, mutations, predictions


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


df, mutation_df, prediction_df = load_data()
model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🧬 AI-Driven Novel Bio-Enzymes")

st.subheader("AI-Based Alpha-Amylase Discovery and Engineering")

st.write(
    "A computational prototype that uses experimental alpha-amylase "
    "sequence data and machine learning to predict experimental targets "
    "and screen possible single-mutation candidates."
)

st.info(
    "⚠️ Computational screening only. Predicted candidates require "
    "experimental validation before biological or industrial use."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Mutation Screening",
        "Candidate Details",
        "Candidate Comparison",
        "Single Mutation Prediction",
        "Dataset"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Project Dashboard")

    st.write(
        "The dashboard summarizes the experimental dataset, machine-learning "
        "model, and computational mutation screening results."
    )

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Experimental Variants",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Training Variants",
            f"{len(df[df['set'] == 'train']):,}"
        )

    with col3:
        st.metric(
            "Test Variants",
            f"{len(df[df['set'] == 'test']):,}"
        )

    with col4:
        st.metric(
            "Screened Mutations",
            f"{len(mutation_df):,}"
        )

    st.divider()

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.subheader("📋 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Total experimental variants**")
        st.write(f"{len(df):,}")

    with col2:
        st.write("**Minimum experimental target**")
        st.write(f"{df['target'].min():.6f}")

    with col3:
        st.write("**Maximum experimental target**")
        st.write(f"{df['target'].max():.6f}")

    st.divider()

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📈 Experimental Target Distribution")

    fig_target = px.histogram(
        df,
        x="target",
        nbins=30,
        title="Distribution of Experimental Target Values"
    )

    fig_target.update_layout(
        xaxis_title="Experimental Target",
        yaxis_title="Number of Variants"
    )

    st.plotly_chart(
        fig_target,
        use_container_width=True
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    if not prediction_df.empty:

        st.subheader("🤖 Machine Learning Model Performance")

        actual = prediction_df["target"]
        predicted = prediction_df["predicted_target"]

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "MAE",
                f"{mae:.6f}"
            )

        with col2:
            st.metric(
                "RMSE",
                f"{rmse:.6f}"
            )

        with col3:
            st.metric(
                "R²",
                f"{r2:.4f}"
            )

        fig_prediction = px.scatter(
            prediction_df,
            x="target",
            y="predicted_target",
            title="Experimental vs Predicted Target",
            labels={
                "target": "Experimental Target",
                "predicted_target": "Predicted Target"
            }
        )

        st.plotly_chart(
            fig_prediction,
            use_container_width=True
        )


# ============================================================
# MUTATION SCREENING
# ============================================================

elif page == "Mutation Screening":

    st.header("🔬 Alpha-Amylase Mutation Screening")

    st.write(
        "The trained machine-learning model predicts the experimental "
        "target for computationally generated single-mutation candidates."
    )

    st.metric(
        "Total Screened Candidates",
        f"{len(mutation_df):,}"
    )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader("🔎 Candidate Filters")

    col1, col2, col3 = st.columns(3)

    with col1:

        number = st.selectbox(
            "Number of candidates",
            [10, 25, 50, 100],
            index=0
        )

    with col2:

        wild_types = sorted(
            mutation_df["wild_type"].unique().tolist()
        )

        selected_wild = st.selectbox(
            "Wild-type amino acid",
            ["All"] + wild_types
        )

    with col3:

        mutants = sorted(
            mutation_df["mutant"].unique().tolist()
        )

        selected_mutant = st.selectbox(
            "Mutant amino acid",
            ["All"] + mutants
        )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered_df = mutation_df.copy()

    if selected_wild != "All":
        filtered_df = filtered_df[
            filtered_df["wild_type"] == selected_wild
        ]

    if selected_mutant != "All":
        filtered_df = filtered_df[
            filtered_df["mutant"] == selected_mutant
        ]

    filtered_df = filtered_df.sort_values(
        "predicted_target",
        ascending=False
    )

    top_candidates = filtered_df.head(number)

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader("🏆 Ranked Mutation Candidates")

    st.dataframe(
        top_candidates[
            [
                "rank",
                "mutation",
                "position",
                "wild_type",
                "mutant",
                "predicted_target"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # TOP CANDIDATE CHART
    # --------------------------------------------------------

    if len(top_candidates) > 0:

        st.subheader("📊 Top Candidate Predictions")

        chart_df = top_candidates.copy()

        chart_df["candidate"] = (
            chart_df["mutation"]
            + "  (Rank "
            + chart_df["rank"].astype(str)
            + ")"
        )

        fig_candidates = px.bar(
            chart_df,
            x="predicted_target",
            y="candidate",
            orientation="h",
            title="Predicted Target of Selected Candidates"
        )

        fig_candidates.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Predicted Experimental Target",
            yaxis_title="Mutation"
        )

        st.plotly_chart(
            fig_candidates,
            use_container_width=True
        )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Ranked Mutation Results",
        data=csv_data,
        file_name="alpha_amylase_ranked_mutations.csv",
        mime="text/csv"
    )


# ============================================================
# CANDIDATE DETAILS
# ============================================================

elif page == "Candidate Details":

    st.header("🏆 Candidate Details")

    candidates_for_selection = mutation_df.head(100)

    candidate = st.selectbox(
        "Select a mutation",
        candidates_for_selection["mutation"].tolist()
    )

    selected = mutation_df[
        mutation_df["mutation"] == candidate
    ].iloc[0]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rank",
            int(selected["rank"])
        )

    with col2:
        st.metric(
            "Position",
            int(selected["position"])
        )

    with col3:
        st.metric(
            "Predicted Target",
            f"{selected['predicted_target']:.6f}"
        )

    st.divider()

    # --------------------------------------------------------
    # MUTATION INFORMATION
    # --------------------------------------------------------

    st.subheader("🧬 Mutation Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Mutation:** {selected['mutation']}"
        )

        st.write(
            f"**Wild-type amino acid:** {selected['wild_type']}"
        )

        st.write(
            f"**Mutant amino acid:** {selected['mutant']}"
        )

    with col2:

        st.write(
            f"**Position:** {int(selected['position'])}"
        )

        st.write(
            f"**Predicted experimental target:** "
            f"{selected['predicted_target']:.6f}"
        )

        st.write(
            f"**Candidate rank:** {int(selected['rank'])}"
        )

    st.divider()

    # --------------------------------------------------------
    # SEQUENCE
    # --------------------------------------------------------

    st.subheader("🧪 Mutated Sequence")

    st.code(
        selected["sequence"],
        language=None
    )

    st.download_button(
        label="📥 Download Candidate Sequence",
        data=selected["sequence"],
        file_name=f"{selected['mutation']}.fasta",
        mime="text/plain"
    )


# ============================================================
# CANDIDATE COMPARISON
# ============================================================

elif page == "Candidate Comparison":

    st.header("⚖️ Candidate Comparison")

    st.write(
        "Select multiple computational candidates to compare their "
        "mutation positions and predicted experimental targets."
    )

    candidate_list = mutation_df.head(100)["mutation"].tolist()

    selected_candidates = st.multiselect(
        "Select candidates",
        candidate_list,
        default=candidate_list[:3]
    )

    if len(selected_candidates) > 0:

        comparison = mutation_df[
            mutation_df["mutation"].isin(selected_candidates)
        ].copy()

        comparison = comparison.sort_values(
            "predicted_target",
            ascending=False
        )

        st.subheader("📋 Comparison Table")

        st.dataframe(
            comparison[
                [
                    "rank",
                    "mutation",
                    "position",
                    "wild_type",
                    "mutant",
                    "predicted_target"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("📊 Predicted Target Comparison")

        fig_compare = px.bar(
            comparison,
            x="mutation",
            y="predicted_target",
            title="Candidate Predicted Target Comparison"
        )

        fig_compare.update_layout(
            xaxis_title="Mutation",
            yaxis_title="Predicted Experimental Target"
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True
        )

    else:

        st.warning(
            "Please select at least one candidate."
        )


# ============================================================
# SINGLE MUTATION PREDICTION
# ============================================================

elif page == "Single Mutation Prediction":

    st.header("🧪 Single Mutation Predictor")

    st.write(
        "Enter an alpha-amylase protein sequence and generate a "
        "machine-learning prediction."
    )

    sequence = st.text_area(
        "Enter protein sequence",
        height=180,
        placeholder="Paste an amino-acid sequence here..."
    )

    if st.button("🔮 Predict Experimental Target"):

        if not sequence.strip():

            st.warning(
                "Please enter a protein sequence."
            )

        else:

            sequence = (
                sequence
                .upper()
                .replace(" ", "")
                .replace("\n", "")
                .replace("\r", "")
            )

            valid_amino_acids = set(
                "ACDEFGHIKLMNPQRSTVWY"
            )

            invalid = [
                aa for aa in sequence
                if aa not in valid_amino_acids
            ]

            if invalid:

                st.error(
                    "Invalid amino acid characters found: "
                    + ", ".join(sorted(set(invalid)))
                )

            else:

                prediction = model.predict(
                    [sequence]
                )[0]

                st.success(
                    f"Predicted experimental target: "
                    f"{prediction:.6f}"
                )

                st.write(
                    f"**Sequence length:** {len(sequence)} amino acids"
                )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.header("📊 Experimental Alpha-Amylase Dataset")

    st.write(
        "This table contains the cleaned experimental alpha-amylase "
        "dataset used by the computational prototype."
    )

    st.metric(
        "Number of Records",
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # DATASET TABLE
    # --------------------------------------------------------

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # TARGET SUMMARY
    # --------------------------------------------------------

    st.subheader("📈 Target Statistics")

    st.dataframe(
        df["target"].describe().to_frame(),
        use_container_width=True
    )

    # --------------------------------------------------------
    # DATASET DOWNLOAD
    # --------------------------------------------------------

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Clean Dataset",
        data=csv_data,
        file_name="alpha_amylase_clean.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Driven Novel Bio-Enzymes | "
    "Alpha-Amylase Computational Prototype"
)