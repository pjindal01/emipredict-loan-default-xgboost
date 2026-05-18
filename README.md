# emipredict-loan-default-xgboost
Loan default risk prediction on 404K records using XGBoost + SMOTE. Weighted F1 0.988, CV F1 0.88. SHAP explainability, MLflow experiment tracking, Streamlit deployment. Python · scikit-learn · XGBoost
# 💳 EMIPredict AI — Intelligent Financial Risk Assessment Platform

> Predicts loan default risk using XGBoost + SMOTE on 404K records. F1 0.988 on test set. Fully explained via SHAP. Tracked with MLflow.

---

## 📌 Problem Statement
Banks and NBFCs need to assess whether a loan applicant is likely to default before disbursing funds. Manual credit review is slow and inconsistent. This project builds an automated, explainable ML pipeline to classify loan risk as **Default** or **Non-Default**.

## 📊 Key Results

| Metric | Value |
|---|---|
| Dataset size | 404,800 rows × 27 columns |
| Best model | XGBoost Classifier |
| Test F1 Score (weighted) | **0.988** |
| CV mean F1 (5-fold) | **0.88** |
| Class imbalance handled | SMOTE |
| Experiment tracking | MLflow |
| Explainability | SHAP summary plots |

## 🛠️ Tech Stack
`Python` · `XGBoost` · `scikit-learn` · `imbalanced-learn (SMOTE)` · `SHAP` · `MLflow` · `Streamlit` · `Pandas` · `Matplotlib` · `Seaborn`

## 📁 Project Structure
```
EMIPredict_AI/
│
├── EMIPredict_AI.ipynb          ← Main notebook (run this)
├── emi_prediction_dataset.csv   ← Dataset (place here)
├── requirements.txt
└── README.md
```

## ▶️ How to Run
```bash
# 1. Clone or download this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Place emi_prediction_dataset.csv in this folder
# 4. Open the notebook
jupyter notebook EMIPredict_AI.ipynb

# 5. Run All Cells (Kernel → Restart & Run All)
```

## 📂 Dataset
- **Source:** [Kaggle — EMI Prediction Dataset](https://www.kaggle.com/)
- **Size:** 404,800 rows, 27 columns
- **Target:** Loan default (binary classification)

## 🔍 Key Findings
- XGBoost with SMOTE outperformed Random Forest and Gradient Boosting
- Top default risk indicators (SHAP): Credit history length, loan amount, interest rate, income
- 5-fold CV confirms results are stable, not overfitted to one split
- All runs logged in MLflow for reproducibility

## ⚙️ Pipeline Steps
1. Exploratory Data Analysis (EDA)
2. Data cleaning and feature engineering
3. SMOTE for class imbalance
4. Benchmarking: Logistic Regression → Random Forest → Gradient Boosting → XGBoost
5. Hyperparameter tuning (5-fold CV)
6. SHAP explainability
7. MLflow experiment logging
8. Streamlit app generation

---
*Project completed as part of AI/ML Internship at Labmentix Pvt. Ltd. (2025–2026)*
