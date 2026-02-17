import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)


def evaluate_model(model, X_test, y_test):
    # Get probability scores
    y_scores = model.predict_proba(X_test)[:, 1]

    # Compute PR-AUC and ROC-AUC
    pr_auc = average_precision_score(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)

    # Find optimal threshold (Max F1)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]

    # Apply optimal threshold
    y_pred_optimal = (y_scores >= best_threshold).astype(int)

    print(f"Optimal Threshold: {best_threshold:.4f}")
    print("\nClassification Report (Optimized Threshold):")
    print(classification_report(y_test, y_pred_optimal))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_optimal))
    print(f"\nPR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
