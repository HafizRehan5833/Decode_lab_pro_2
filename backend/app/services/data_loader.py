"""
services/data_loader.py
-----------------------
PHASE 1 — INPUT: Load & understand the Iris dataset.

The Iris benchmark (DecodeLabs slide "Raw Material"):
  - Samples  : 150 (balanced, 50 per class)
  - Classes  : 3  (Setosa, Versicolor, Virginica)
  - Dimensions: 4  (sepal_length, sepal_width, petal_length, petal_width)

We use sklearn's built-in loader — no CSV needed.
"""

import numpy as np
from sklearn.datasets import load_iris
from app.core.config import IRIS_CLASS_NAMES, FEATURE_NAMES


def load_dataset() -> dict:
    """
    Load the Iris dataset and return a structured summary.

    Returns a dict with:
      X         : feature matrix (150 × 4)
      y         : label vector   (150,)
      summary   : human-readable statistics dict
    """
    iris = load_iris()
    X: np.ndarray = iris.data    # shape (150, 4)
    y: np.ndarray = iris.target  # shape (150,)  — 0, 1, 2

    # ── EDA Summary ───────────────────────────────────────────────────────
    summary = {
        "total_samples": int(X.shape[0]),
        "n_features":    int(X.shape[1]),
        "n_classes":     len(IRIS_CLASS_NAMES),
        "class_names":   IRIS_CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
        "class_distribution": {
            IRIS_CLASS_NAMES[i]: int(np.sum(y == i))
            for i in range(len(IRIS_CLASS_NAMES))
        },
        "feature_stats": {
            feat: {
                "min":  round(float(X[:, i].min()), 3),
                "max":  round(float(X[:, i].max()), 3),
                "mean": round(float(X[:, i].mean()), 3),
                "std":  round(float(X[:, i].std()), 3),
            }
            for i, feat in enumerate(FEATURE_NAMES)
        },
    }

    return {"X": X, "y": y, "summary": summary}
