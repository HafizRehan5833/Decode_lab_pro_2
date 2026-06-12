"""
core/config.py
--------------
Central configuration for DecodeLabs Project 2 — Iris Classifier.
"""

APP_NAME    = "DecodeLabs Iris Classifier — Project 2"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "Project 2 — Industrial Training Kit | Batch 2026 | Powered by DecodeLabs. "
    "KNN-based iris flower classification using the full supervised learning pipeline."
)

# ── Dataset ────────────────────────────────────────────────────────────────
IRIS_CLASS_NAMES: list[str] = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
FEATURE_NAMES:   list[str] = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

# ── Preprocessing ──────────────────────────────────────────────────────────
TEST_SIZE    = 0.20      # 80/20 split (as per DecodeLabs spec)
RANDOM_STATE = 42        # Fixed seed for reproducibility

# ── Model Hyperparameter ───────────────────────────────────────────────────
KNN_NEIGHBORS = 5        # K=5 (the proximity principle — majority vote)
