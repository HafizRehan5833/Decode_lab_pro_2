"""
services/preprocessor.py
------------------------
PHASE 2 — PROCESS (Part A): Feature Scaling + Train/Test Split.

DecodeLabs "Gatekeeper Rule: Scaling":
  Raw data is biased. StandardScaler normalizes to Mean=0, Variance=1.
  This is CRITICAL for KNN — distance-based algorithms are scale-sensitive.

DecodeLabs "Structural Integrity: The Split":
  - Training set (80%) — Pattern Recognition
  - Test set     (20%) — Validation (locked, never seen during training)
  - SHUFFLE before splitting to remove order bias
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from app.core.config import TEST_SIZE, RANDOM_STATE


def preprocess(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Full preprocessing pipeline:
      1. Shuffle + split  (80/20)
      2. Fit StandardScaler on TRAINING data only
      3. Transform both train and test sets

    Returns dict with X_train, X_test, y_train, y_test, scaler.

    IMPORTANT: The scaler is fit ONLY on X_train to prevent data leakage.
    """
    # ── Step 1: Shuffle + Split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,          # remove order bias
        stratify=y,            # keep class proportions equal in both sets
    )

    # ── Step 2: Fit scaler on TRAINING data only ───────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train)        # learn mean & std from training set only

    # ── Step 3: Transform both sets ───────────────────────────────────────
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
        "split_info": {
            "train_samples": int(X_train.shape[0]),
            "test_samples":  int(X_test.shape[0]),
            "test_ratio":    TEST_SIZE,
        },
    }


def scale_single(scaler: StandardScaler, features: list[float]) -> np.ndarray:
    """Scale a single prediction input using the already-fitted scaler."""
    return scaler.transform([features])
