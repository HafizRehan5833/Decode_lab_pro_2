"""
services/trainer.py
-------------------
PHASE 2 — PROCESS (Part B): KNN Model Training.

DecodeLabs "The Algorithm: K-Nearest Neighbors":
  The Proximity Principle — similar things exist in close proximity.
  K=5 → majority vote among 5 nearest neighbors.

DecodeLabs "The Workflow: Scikit-Learn":
  model = KNeighborsClassifier(n_neighbors=5)   ← INSTANTIATE
  model.fit(X_train, y_train)                   ← FIT (memorize the map)
  predictions = model.predict(X_test)           ← PREDICT (apply logic)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from app.core.config import KNN_NEIGHBORS


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    """
    Instantiate and fit the KNN classifier.

    KNN is a lazy learner — it doesn't build an explicit model during fit.
    It memorizes all training samples and classifies new points by proximity.
    """
    model = KNeighborsClassifier(
        n_neighbors=KNN_NEIGHBORS,
        metric="euclidean",       # Euclidean distance in feature space
        weights="uniform",        # Each neighbor votes equally
    )
    model.fit(X_train, y_train)   # ← FIT: memorize the training map
    return model


def predict_single(model: KNeighborsClassifier, X_scaled: np.ndarray) -> dict:
    """
    Predict class + probability for a single scaled input.
    Returns label index, class name index, and per-class probabilities.
    """
    label    = int(model.predict(X_scaled)[0])
    proba    = model.predict_proba(X_scaled)[0].tolist()
    return {"label": label, "probabilities": [round(p, 4) for p in proba]}
