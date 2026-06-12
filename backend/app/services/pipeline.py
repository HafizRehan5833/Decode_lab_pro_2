"""
services/pipeline.py
--------------------
THE FULL IPO PIPELINE ORCHESTRATOR.

Builds and holds the trained model + scaler as a singleton so
FastAPI doesn't retrain on every request.

Pipeline:
  INPUT   → load_dataset()   → X, y
  PROCESS → preprocess()     → scaled splits
            train_model()    → fitted KNN
  OUTPUT  → evaluate_model() → metrics report
"""

from __future__ import annotations
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from app.services.data_loader  import load_dataset
from app.services.preprocessor import preprocess, scale_single
from app.services.trainer      import train_model, predict_single
from app.services.evaluator    import evaluate_model
from app.core.config           import IRIS_CLASS_NAMES


class ClassificationPipeline:
    """
    Singleton that holds the trained model, scaler, and evaluation metrics.
    Call build() once at startup; then use predict() for inference.
    """

    def __init__(self):
        self.model:       KNeighborsClassifier | None = None
        self.scaler:      StandardScaler       | None = None
        self.metrics:     dict                        = {}
        self.data_summary: dict                       = {}
        self._trained:    bool                        = False

    # ── Build (train once at startup) ─────────────────────────────────────
    def build(self) -> None:
        """Full IPO pipeline: load → preprocess → train → evaluate."""

        # INPUT ─────────────────────────────────────────────────────────────
        dataset = load_dataset()
        X, y    = dataset["X"], dataset["y"]
        self.data_summary = dataset["summary"]

        # PROCESS ───────────────────────────────────────────────────────────
        prep = preprocess(X, y)
        self.scaler = prep["scaler"]

        self.model = train_model(prep["X_train"], prep["y_train"])

        # OUTPUT ────────────────────────────────────────────────────────────
        y_pred       = self.model.predict(prep["X_test"])
        eval_results = evaluate_model(prep["y_test"], y_pred)

        self.metrics = {
            **eval_results,
            "split_info": prep["split_info"],
        }
        self._trained = True
        print(f"[Pipeline] Model trained. Accuracy={self.metrics['accuracy']:.4f}  "
              f"F1={self.metrics['f1_macro']:.4f}")

    # ── Predict ───────────────────────────────────────────────────────────
    def predict(
        self,
        sepal_length: float,
        sepal_width:  float,
        petal_length: float,
        petal_width:  float,
    ) -> dict:
        """Scale input and return class prediction with probabilities."""
        if not self._trained:
            raise RuntimeError("Pipeline not trained. Call build() first.")

        features  = [sepal_length, sepal_width, petal_length, petal_width]
        X_scaled  = scale_single(self.scaler, features)
        result    = predict_single(self.model, X_scaled)

        label_idx = result["label"]
        return {
            "predicted_class": IRIS_CLASS_NAMES[label_idx],
            "class_index":     label_idx,
            "probabilities":   {
                cls: result["probabilities"][i]
                for i, cls in enumerate(IRIS_CLASS_NAMES)
            },
            "input_features": {
                "sepal_length_cm": sepal_length,
                "sepal_width_cm":  sepal_width,
                "petal_length_cm": petal_length,
                "petal_width_cm":  petal_width,
            },
        }


# ── Module-level singleton — auto-builds on first use ─────────────────────
pipeline = ClassificationPipeline()
pipeline.build()   # build eagerly so TestClient and CLI both work without lifespan
