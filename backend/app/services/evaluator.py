"""
services/evaluator.py
---------------------
PHASE 3 — OUTPUT: Model Evaluation.

DecodeLabs "Output Validation" slides:
  ❌ Accuracy alone is the "Accuracy Mirage" — misleading on imbalanced data.
  ✅ We must use:
       • Confusion Matrix (TP, FP, FN, TN per class)
       • F1 Score (harmonic mean of Precision and Recall)

DecodeLabs "Strategic Trade-Offs":
  Precision → Trustworthiness (e.g. spam filters)
  Recall    → Sensitivity (e.g. medical diagnosis)
  F1 Score  → The balanced harmonic mean of both
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from app.core.config import IRIS_CLASS_NAMES


def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Full evaluation report for a classification model.

    Returns:
      accuracy        : overall accuracy (note: the Accuracy Mirage)
      f1_macro        : macro-averaged F1 (treats all classes equally)
      f1_weighted     : weighted F1 (accounts for class support)
      precision_macro : macro-averaged precision
      recall_macro    : macro-averaged recall
      confusion_matrix: 2D list (actual × predicted)
      per_class       : per-class precision, recall, F1, support
      report          : full sklearn classification report string
    """
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=IRIS_CLASS_NAMES)

    per_class = {}
    for i, cls in enumerate(IRIS_CLASS_NAMES):
        mask = y_test == i
        per_class[cls] = {
            "precision": round(float(precision_score(y_test, y_pred, labels=[i], average="macro", zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, y_pred, labels=[i], average="macro", zero_division=0)), 4),
            "f1":        round(float(f1_score(y_test, y_pred, labels=[i], average="macro", zero_division=0)), 4),
            "support":   int(mask.sum()),
        }

    return {
        "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro":         round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "f1_weighted":      round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "precision_macro":  round(float(precision_score(y_test, y_pred, average="macro")), 4),
        "recall_macro":     round(float(recall_score(y_test, y_pred, average="macro")), 4),
        "confusion_matrix": cm.tolist(),
        "class_names":      IRIS_CLASS_NAMES,
        "per_class":        per_class,
        "report":           report,
    }
