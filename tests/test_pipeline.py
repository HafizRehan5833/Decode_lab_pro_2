"""
tests/test_pipeline.py
----------------------
Unit + integration tests for DecodeLabs Project 2.
Run with:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
import numpy as np
from app.services.data_loader  import load_dataset
from app.services.preprocessor import preprocess, scale_single
from app.services.trainer      import train_model, predict_single
from app.services.evaluator    import evaluate_model
from app.services.pipeline     import ClassificationPipeline
from app.core.config           import IRIS_CLASS_NAMES, KNN_NEIGHBORS


# ── Data Loader ────────────────────────────────────────────────────────────
class TestDataLoader:
    def test_dataset_shape(self):
        d = load_dataset()
        assert d["X"].shape == (150, 4)
        assert d["y"].shape == (150,)

    def test_three_classes(self):
        d = load_dataset()
        assert len(np.unique(d["y"])) == 3

    def test_balanced_classes(self):
        d = load_dataset()
        for cls in [0, 1, 2]:
            assert int(np.sum(d["y"] == cls)) == 50

    def test_summary_keys(self):
        s = load_dataset()["summary"]
        for key in ["total_samples", "n_features", "n_classes", "feature_stats"]:
            assert key in s


# ── Preprocessor ──────────────────────────────────────────────────────────
class TestPreprocessor:
    def setup_method(self):
        d = load_dataset()
        self.prep = preprocess(d["X"], d["y"])

    def test_split_sizes(self):
        assert self.prep["X_train"].shape[0] == 120
        assert self.prep["X_test"].shape[0]  == 30

    def test_scaler_mean_near_zero(self):
        # After StandardScaler, training set mean should be ≈ 0
        means = self.prep["X_train"].mean(axis=0)
        assert all(abs(m) < 0.01 for m in means)

    def test_scaler_std_near_one(self):
        stds = self.prep["X_train"].std(axis=0)
        assert all(abs(s - 1.0) < 0.01 for s in stds)

    def test_scale_single(self):
        result = scale_single(self.prep["scaler"], [5.1, 3.5, 1.4, 0.2])
        assert result.shape == (1, 4)


# ── Trainer ───────────────────────────────────────────────────────────────
class TestTrainer:
    def setup_method(self):
        d    = load_dataset()
        prep = preprocess(d["X"], d["y"])
        self.model = train_model(prep["X_train"], prep["y_train"])
        self.X_test = prep["X_test"]

    def test_model_n_neighbors(self):
        assert self.model.n_neighbors == KNN_NEIGHBORS

    def test_predict_returns_valid_class(self):
        result = predict_single(self.model, self.X_test[:1])
        assert result["label"] in [0, 1, 2]

    def test_probabilities_sum_to_one(self):
        result = predict_single(self.model, self.X_test[:1])
        assert abs(sum(result["probabilities"]) - 1.0) < 0.001

    def test_probabilities_length(self):
        result = predict_single(self.model, self.X_test[:1])
        assert len(result["probabilities"]) == 3


# ── Evaluator ─────────────────────────────────────────────────────────────
class TestEvaluator:
    def setup_method(self):
        d    = load_dataset()
        prep = preprocess(d["X"], d["y"])
        model = train_model(prep["X_train"], prep["y_train"])
        y_pred = model.predict(prep["X_test"])
        self.metrics = evaluate_model(prep["y_test"], y_pred)

    def test_accuracy_above_90(self):
        # KNN on Iris should easily exceed 90% accuracy
        assert self.metrics["accuracy"] >= 0.90

    def test_f1_above_90(self):
        assert self.metrics["f1_macro"] >= 0.90

    def test_confusion_matrix_shape(self):
        cm = self.metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_confusion_matrix_sum(self):
        # Sum of all CM values should equal number of test samples (30)
        total = sum(sum(row) for row in self.metrics["confusion_matrix"])
        assert total == 30

    def test_per_class_keys(self):
        for cls in IRIS_CLASS_NAMES:
            assert cls in self.metrics["per_class"]


# ── Full Pipeline ─────────────────────────────────────────────────────────
class TestPipeline:
    def setup_method(self):
        self.pl = ClassificationPipeline()
        self.pl.build()

    def test_pipeline_trained(self):
        assert self.pl._trained is True

    def test_predict_setosa(self):
        # Classic Setosa measurements
        result = self.pl.predict(5.1, 3.5, 1.4, 0.2)
        assert result["predicted_class"] == "Iris-Setosa"

    def test_predict_virginica(self):
        # Classic Virginica measurements
        result = self.pl.predict(6.3, 3.3, 6.0, 2.5)
        assert result["predicted_class"] == "Iris-Virginica"

    def test_predict_returns_all_probabilities(self):
        result = self.pl.predict(5.1, 3.5, 1.4, 0.2)
        assert set(result["probabilities"].keys()) == set(IRIS_CLASS_NAMES)

    def test_predict_probability_sum(self):
        result = self.pl.predict(5.1, 3.5, 1.4, 0.2)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.001


# ── FastAPI Endpoints ─────────────────────────────────────────────────────
try:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    class TestAPIEndpoints:
        def test_health_check(self):
            r = client.get("/")
            assert r.status_code == 200
            assert r.json()["trained"] is True

        def test_predict_setosa(self):
            r = client.post("/api/v1/predict/", json={
                "sepal_length": 5.1, "sepal_width": 3.5,
                "petal_length": 1.4, "petal_width": 0.2
            })
            assert r.status_code == 200
            assert r.json()["predicted_class"] == "Iris-Setosa"

        def test_predict_probabilities_present(self):
            r = client.post("/api/v1/predict/", json={
                "sepal_length": 6.0, "sepal_width": 2.9,
                "petal_length": 4.5, "petal_width": 1.5
            })
            assert r.status_code == 200
            assert "probabilities" in r.json()

        def test_predict_invalid_input(self):
            # negative value should be rejected by pydantic (gt=0)
            r = client.post("/api/v1/predict/", json={
                "sepal_length": -1, "sepal_width": 3.5,
                "petal_length": 1.4, "petal_width": 0.2
            })
            assert r.status_code == 422

        def test_get_metrics(self):
            r = client.get("/api/v1/model/metrics")
            assert r.status_code == 200
            data = r.json()
            assert "confusion_matrix" in data
            assert "f1_macro" in data
            assert data["accuracy"] >= 0.90

        def test_get_dataset_info(self):
            r = client.get("/api/v1/model/dataset")
            assert r.status_code == 200
            data = r.json()
            assert data["total_samples"] == 150
            assert data["n_classes"] == 3

except ImportError:
    pass
