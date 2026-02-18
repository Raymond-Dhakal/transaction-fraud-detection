# Credit Card Fraud Detection

## Overview

This project builds a machine learning pipeline to detect fraudulent credit card transactions under extreme class imbalance (~0.17% fraud).

The objective is to:

- Handle severe class imbalance
- Compare multiple modeling strategies
- Optimize decision threshold
- Select a robust and practical fraud detection model

---

## Dataset

The dataset contains 284,807 transactions with 492 fraudulent cases.

Features:
- V1–V28: PCA-transformed components
- Time: Seconds elapsed between transactions
- Amount: Transaction amount
- Class: Target variable (0 = Normal, 1 = Fraud)

Fraud rate ≈ 0.17%.

The dataset is publicly available on Kaggle:

[Credit Card Fraud Dataset (Kaggle)](https://www.kaggle.com/datasets/joebeachcapital/credit-card-fraud/data)

After downloading, place the file `creditcard.csv` inside:
---

## Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│
├── reports/
│   ├── figures/
│   └── results.md
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Modeling Approach

1. Exploratory Data Analysis (EDA)
2. Stratified train-test split
3. Robust scaling of `Amount`
4. Logistic Regression (baseline, SMOTE, class_weight)
5. Random Forest
6. Threshold optimization (maximize F1-score)
7. Feature importance analysis

---

## Final Model

**Random Forest with optimized threshold**

Performance:

- Precision (Fraud): 0.93  
- Recall (Fraud): 0.86  
- F1-score: 0.89  
- PR-AUC: 0.87  
- ROC-AUC: 0.95  

Threshold optimization improved recall while maintaining high precision.

---

## Key Insights

- Logistic Regression struggled under extreme imbalance.
- SMOTE did not significantly improve ranking ability.
- Random Forest captured non-linear fraud patterns effectively.
- PR-AUC is a more reliable metric than accuracy for this problem.
- Feature importance revealed V17, V12, and V14 as key predictors.

---

## How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Full Pipeline

```bash
python main.py
```

---

## Evaluation Outputs

Figures are saved in:

```
reports/figures/
```

Includes:

- Precision-Recall Curve
- ROC Curve
- Feature Importance Plot

---

## Conclusion

Random Forest with optimized threshold provides a strong balance between fraud detection and minimizing false alarms, making it suitable for real-world fraud detection systems under severe class imbalance.
