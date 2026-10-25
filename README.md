# 🧬 AI-Driven Alpha-Amylase Mutation Analysis

An AI-powered bioinformatics project that uses machine learning to analyze alpha-amylase protein sequences, evaluate custom mutations, predict sequence-based scores, and support the exploration of potentially useful enzyme variants.

## 📌 Project Overview

Alpha-amylase is an enzyme that breaks down starch into smaller carbohydrate molecules. Its activity and stability are important in industries such as food processing, detergents, brewing, and biotechnology.

The goal of this project is to develop a machine-learning-based application that helps users explore alpha-amylase mutations and investigate their potential effects using protein sequence analysis and computational predictions.

The application provides an interactive interface where users can enter experimental conditions, explore available mutation candidates, evaluate custom mutations, compare predicted scores, and visualize protein structures when supported by the available structural data and visualization tools.

## 🎯 Objectives

- Analyze alpha-amylase protein sequence data using machine learning.
- Preprocess protein sequence datasets for model development.
- Extract sequence features using TF-IDF.
- Train a Ridge Regression model to predict sequence-associated scores.
- Generate and screen potential amino-acid substitutions.
- Validate custom mutation notation and residue positions against a reference sequence.
- Compare predicted scores between reference and mutated sequences.
- Visualize prediction comparisons and protein sequences.
- Provide a foundation for further structural analysis and experimental validation.

## ✨ Features

### 1. Condition-Based Mutation Screening
Explore mutation candidates using configurable parameters such as temperature, pH, substrate, solvent, cofactor, and optimization objective, according to the options implemented in the application.

### 2. Dataset Mutation Selection
Select available mutation candidates from the project's mutation dataset for computational evaluation.

### 3. Custom Mutation Evaluation
Enter a custom mutation using the format:

`OriginalAminoAcid + Position + NewAminoAcid`

**Example:** `T125V`

This represents a substitution of threonine (T) with valine (V) at position 125 in the selected reference sequence.

The application checks:

- Whether the mutation follows the required notation.
- Whether the specified position exists in the reference sequence.
- Whether the original amino acid matches the residue at that position.
- Whether the replacement amino acid differs from the original residue.

If these checks pass, the application constructs the mutated sequence for further computational analysis.

**Note:** Passing these checks establishes input and sequence consistency, not biological effectiveness or experimental validity.

### 4. Machine Learning Prediction
The application uses a trained Ridge Regression model to predict a score from protein sequence features.

The predicted score can be compared with the reference sequence score to explore how the model evaluates a proposed mutation.

The interpretation of the score depends on the target variable and training data used to develop the model.

### 5. Mutation Analysis
The mutation analysis functionality can display a comparison between predicted scores for the reference and mutated sequences, along with the resulting amino-acid sequence.

### 6. Model Evaluation
Evaluate the machine-learning model using the evaluation data and metrics implemented in the project.

### 7. Experimental Validation Workflow
Support the planning of further experimental evaluation of computationally selected mutation candidates.

Computational predictions should be treated as hypotheses that require appropriate experimental verification.

### 8. Protein Structure Visualization
The project can support three-dimensional protein structure visualization through its implemented structure-rendering functionality and available structural data.

Displaying a reference structure does not necessarily mean that a mutated structure has been generated or validated.

## 🧠 Machine Learning Methodology

The project's machine-learning workflow consists of the following stages:

1. **Data Collection:** Obtain protein sequence and associated score data from the selected dataset.
2. **Data Preprocessing:** Clean and prepare the data for model development.
3. **Feature Extraction:** Represent protein sequences using TF-IDF features.
4. **Model Training:** Train a Ridge Regression model using the training dataset.
5. **Model Evaluation:** Evaluate predictions using held-out testing data and the metrics implemented in the project.
6. **Mutation Generation:** Generate or load candidate amino-acid substitutions.
7. **Sequence Validation:** Check custom mutation notation and verify residue positions against the reference sequence.
8. **Prediction:** Use the trained model to calculate scores for reference and mutated sequences.
9. **Result Visualization:** Display prediction comparisons and relevant sequence or structural information.

### Workflow Diagram

```text
Protein Sequence Dataset
          |
          v
   Data Preprocessing
          |
          v
 TF-IDF Feature Extraction
          |
          v
   Ridge Regression
          |
          v
    Model Evaluation
          |
          v
 Mutation Candidate Selection
          |
          v
 Custom Mutation Validation
          |
          v
  Mutated Sequence Generation
          |
          v
    Model Prediction
          |
          v
 Score Comparison and Analysis
          |
          v
 Visualization and Further
 Experimental Validation
```

## 🧪 Custom Mutation Validation

A custom mutation is checked against the reference protein sequence before prediction.

For example, consider `T125V`:

- **Original amino acid:** Threonine (T)
- **Position:** 125
- **Replacement amino acid:** Valine (V)

The application verifies that the reference sequence contains threonine at position 125 before generating the substitution.

If the reference sequence contains a different amino acid at that position, the mutation is rejected or a corrected notation is suggested.

These checks prevent common input errors. They do not establish that the mutation improves enzyme activity, thermal stability, substrate specificity, or other biochemical properties.

## 📊 Dataset

The project uses protein sequence data and associated scores for machine-learning development.

The working dataset previously reported for this project contains:

- **Training records:** 3,004
- **Testing records:** 702
- **Total records:** 3,706

The dataset is organized into cleaned, training, and testing files, alongside a mutation-screening dataset.

### Expected Dataset Files

```text
data/
├── processed/
│   ├── alpha_amylase_clean.csv
│   ├── alpha_amylase_train.csv
│   ├── alpha_amylase_test.csv
│   └── alpha_amylase_mutation_screen.csv
```

The original dataset source and its experimental measurement definitions should be documented using the exact source and metadata associated with the dataset used in your local project.

Dataset availability may depend on whether the original files are included in the repository.

## 🛠️ Technologies Used

- **Python** — Main programming language
- **Streamlit** — Interactive web application
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — TF-IDF feature extraction, Ridge Regression, and model evaluation
- **Joblib** — Saving and loading the trained model
- **Regular Expressions** — Custom mutation notation validation
- **Protein Sequence Analysis** — Amino-acid sequence processing
- **Protein Structure Visualization** — Visualization functionality supported by the project's implementation and structural data

## 📂 Project Structure

```text
AI_Alpha_Amylase/
│
├── app.py
│
├── data/
│   └── processed/
│       ├── alpha_amylase_clean.csv
│       ├── alpha_amylase_train.csv
│       ├── alpha_amylase_test.csv
│       └── alpha_amylase_mutation_screen.csv
│
├── models/
│   ├── alpha_amylase_model.pkl
│   └── test_predictions.csv
│
├── requirements.txt
│
└── README.md
```

*This is an illustrative project structure. Adjust the file list to match the files actually included in your repository.*

## ⚙️ Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- A code editor such as Visual Studio Code

### Step 1: Clone the Repository

Replace `<your-github-username>` with your GitHub username.

```bash
git clone https://github.com/<your-github-username>/<your-repository-name>.git
```

### Step 2: Navigate to the Project Folder

```bash
cd <your-repository-name>
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

If the repository contains `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

Otherwise, install the packages required by the application. For the core machine-learning and Streamlit functionality, these commonly include:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

Additional packages may be required for structure visualization, depending on the implementation in `app.py`.

### Step 5: Run the Application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## 🚀 How to Use the Application

1. Start the Streamlit application.
2. Open the required page from the sidebar.
3. Select the available screening parameters or mutation candidates.
4. To evaluate a custom mutation, enter a substitution such as `T125V`.
5. Submit the mutation for validation.
6. Review the generated mutated sequence and model prediction.
7. Compare the predicted score with the reference sequence score.
8. Explore available visualizations and use the results to identify candidates for further investigation.

The mutation must match the selected reference sequence. Different alpha-amylase variants can have different amino acids at the same position.

## ⚠️ Limitations

- The model's predictions depend on the quality, coverage, and target variable of the training dataset.
- A mutation that passes sequence validation is not necessarily biologically beneficial.
- Predicted scores are not equivalent to direct laboratory measurements.
- The model should not be assumed to predict temperature-dependent activity, pH tolerance, or solvent tolerance accurately unless it has been trained and validated for those targets.
- A protein structure visualization does not by itself prove that a mutation improves stability or function.
- Structural conclusions require appropriate structural data and analysis methods.
- Experimental testing is needed to establish the real biochemical effects of promising mutations.

## 🔬 Future Enhancements

- Integrate additional experimentally characterized alpha-amylase datasets.
- Improve model evaluation and prediction uncertainty estimates.
- Add mutation-level comparisons against independent experimental results.
- Incorporate suitable protein structure prediction and structural analysis tools.
- Investigate the effects of mutations on thermal stability and enzyme activity using target-specific data.
- Add more detailed mutation reports and visualizations.
- Develop a feedback workflow to incorporate new experimental measurements into future model training.

## 🎓 Applications

Potential application areas include:

- Bioinformatics education and research
- Computational enzyme engineering
- Protein sequence analysis
- Machine-learning-assisted mutation screening
- Preliminary exploration of industrial enzyme variants

These are intended application areas; the current model's ability to support each use case depends on its training data and validation.

## 👩‍💻 Project Purpose

This project demonstrates how machine learning and bioinformatics techniques can be combined in an interactive application to explore protein sequences, evaluate proposed mutations, and investigate computational predictions relevant to enzyme engineering.

It is intended as a research and educational prototype, not as a replacement for biochemical experiments.

## 📄 License

Choose a license appropriate for your project before distributing the repository. For example, the MIT License may be suitable if you intend to permit reuse under its terms.

If you use an external dataset, verify its licensing and attribution requirements separately.

## 🙌 Acknowledgements

We acknowledge the researchers who develop and share protein datasets and machine-learning tools that support computational protein engineering.

Please add the exact dataset citation and any required software acknowledgements before publishing the repository.
