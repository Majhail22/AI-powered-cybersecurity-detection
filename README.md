# Cybersecurity Intrusion Analysis

GitHub-ready project structure based on the attached notebook `Tanish_revised.ipynb`.

## Project overview

This project analyses a cybersecurity intrusion dataset and compares multiple machine learning models for binary intrusion detection.

The workflow includes:

- data loading and quick structure checks
- basic cleaning and target preparation
- descriptive statistics and missing-value analysis
- categorical imputation comparison for `encryption_used`
- outlier detection using IQR and modified z-score
- exploratory visualisation
- mixed-type association analysis
- preprocessing with scaling and one-hot encoding
- model training and evaluation:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
  - KNN
- ROC and Precision-Recall curve plotting
- confusion matrix for the best model
- optional SHAP explainability for tree-based models

## Repository structure

```text
cybersecurity-intrusion-analysis-repo/
├── data/
│   └── README.md
├── notebooks/
│   └── Tanish_revised.ipynb
├── outputs/
├── src/
│   └── cybersecurity_intrusion_analysis.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

Place your dataset in the `data/` folder as:

```text
data/cybersecurity_intrusion_data.csv
```

If your filename is different, pass it as an argument when running the script.

## Installation

```bash
pip install -r requirements.txt
```

## Run from command line

```bash
python src/cybersecurity_intrusion_analysis.py --data data/cybersecurity_intrusion_data.csv --output outputs
```

## Outputs

The script saves figures and result tables into the `outputs/` folder, including:

- missing value bar chart
- missing-value heatmap
- count plots
- numeric distributions
- donut charts
- correlation heatmap
- model comparison CSV
- ROC curve
- Precision-Recall curve
- confusion matrix
- optional SHAP summary plot

## Notes

- The original notebook uses a hard-coded Windows path. In this repository version, the script uses a relative dataset path and command-line arguments instead.
- SHAP is optional. If it is not installed or the best model is not tree-based, the main pipeline still works.
- The original notebook is kept unchanged inside `notebooks/` for traceability.

## Suggested GitHub repo description

A reproducible Python project for cybersecurity intrusion data analysis, visualisation, and binary attack detection using classical machine learning models.
