"""
notebooks/exploration.py
------------------------
Standalone EDA script — run directly with:
    python notebooks/exploration.py

Prints a full dataset analysis without needing a server.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.data_loader  import load_dataset
from app.services.preprocessor import preprocess
from app.services.trainer      import train_model
from app.services.evaluator    import evaluate_model


def main():
    print("=" * 60)
    print("  DecodeLabs Project 2 — Iris Classifier EDA")
    print("=" * 60)

    # Load
    dataset = load_dataset()
    s = dataset["summary"]
    print(f"\n📊 Dataset: {s['total_samples']} samples, "
          f"{s['n_features']} features, {s['n_classes']} classes")
    print(f"   Classes: {s['class_names']}")
    print(f"   Distribution: {s['class_distribution']}")

    print("\n📐 Feature Statistics:")
    for feat, stats in s["feature_stats"].items():
        print(f"   {feat:22s}  min={stats['min']:.2f}  "
              f"max={stats['max']:.2f}  mean={stats['mean']:.2f}  "
              f"std={stats['std']:.2f}")

    # Preprocess
    prep = preprocess(dataset["X"], dataset["y"])
    si = prep["split_info"]
    print(f"\n✂️  Split: {si['train_samples']} train / "
          f"{si['test_samples']} test  ({int((1-si['test_ratio'])*100)}/{int(si['test_ratio']*100)})")

    # Train
    model = train_model(prep["X_train"], prep["y_train"])
    print(f"\n🤖 KNN trained  (k={model.n_neighbors}, metric=euclidean)")

    # Evaluate
    y_pred   = model.predict(prep["X_test"])
    metrics  = evaluate_model(prep["y_test"], y_pred)

    print(f"\n✅ Results:")
    print(f"   Accuracy  : {metrics['accuracy']:.4f}")
    print(f"   F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"   Precision : {metrics['precision_macro']:.4f}")
    print(f"   Recall    : {metrics['recall_macro']:.4f}")

    print(f"\n🔢 Confusion Matrix (rows=actual, cols=predicted):")
    header = "              " + "  ".join(f"{c[:10]:>10}" for c in metrics["class_names"])
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        label = metrics["class_names"][i][:12]
        print(f"  {label:14s}" + "  ".join(f"{v:>10}" for v in row))

    print(f"\n📋 Full Classification Report:\n")
    print(metrics["report"])
    print("=" * 60)


if __name__ == "__main__":
    main()
