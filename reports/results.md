# Credit Card Fraud Detection - Final Results

## Problem
Detect fraudulent transactions under extreme class imbalance (~0.17% fraud).

## Best Model
Random Forest (n_estimators=100)

## Performance (Optimized Threshold)

- Precision (Fraud): 0.93
- Recall (Fraud): 0.86
- F1-score: 0.89
- PR-AUC: 0.87
- ROC-AUC: 0.95

## Key Insights

- Logistic Regression struggled under imbalance.
- SMOTE did not improve ranking ability.
- Random Forest captured non-linear fraud patterns effectively.
- Threshold optimization improved recall with minimal precision loss.
- V17, V12, and V14 were most influential features.

## Conclusion

Random Forest with optimized threshold provides a strong balance
between fraud detection and minimizing false alarms.
