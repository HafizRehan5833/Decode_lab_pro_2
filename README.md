# 🌸 Iris Data Classifier — DecodeLabs Project 2

A supervised learning pipeline built with **scikit-learn**, **FastAPI**, and **Python**.
Implements the full IPO (Input → Process → Output) model from the DecodeLabs
Architecture Briefing — Project 2.

---

## 📁 Project Structure

```
iris_classifier/
│
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py              # Settings, constants, class names
│   │   │
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── endpoints/
│   │   │           ├── __init__.py
│   │   │           ├── predict.py     # POST /predict — classify a flower
│   │   │           └── model_info.py  # GET  /model  — metrics & confusion matrix
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py             # Pydantic I/O models
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py         # Load & explore the Iris dataset
│   │   │   ├── preprocessor.py        # StandardScaler + train/test split
│   │   │   ├── trainer.py             # KNN model training (fit)
│   │   │   ├── evaluator.py           # Confusion matrix, F1 score
│   │   │   └── pipeline.py            # Orchestrates the full IPO pipeline
│   │   │
│   │   ├── __init__.py
│   │   └── main.py                    # FastAPI app entry point
│   │
│   └── requirements.txt
│
├── frontend/
│   └── index.html                     # Browser-based classifier UI
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py               # Unit + integration tests
│
├── notebooks/
│   └── exploration.py                 # Standalone script for EDA
│
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Start the server
uvicorn app.main:app --reload --port 8000

# 3. Open the UI
# Open frontend/index.html in your browser
# OR visit http://localhost:8000/docs
```

---

## 🧠 Pipeline (IPO Model)

```
INPUT                  PROCESS                    OUTPUT
──────────────         ──────────────────         ──────────────────
Iris Dataset      →    StandardScaler        →    Confusion Matrix
Feature Scaling        80/20 Split                F1 Score
                        KNN Algorithm              Prediction
```

---

## ✅ Project 2 Checklist

- [x] Load and explore the Iris dataset
- [x] Feature Scaling (StandardScaler)
- [x] Train-Test Split 80/20 with shuffle
- [x] KNN Classification (n_neighbors=5)
- [x] Confusion Matrix output
- [x] F1 Score output
- [x] FastAPI REST API
- [x] Browser-based prediction UI
- [x] Unit tests
