import argparse
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from .features import prepare_dataset, LABEL_DECODING


def evaluate(dataset_path: str, model_dir: str = "ml/model"):
    print(f"Loading model from {model_dir}...")
    model = xgb.XGBClassifier()
    model.load_model(f"{model_dir}/xgboost.json")

    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    print(f"Loading dataset...")
    df = pd.read_parquet(dataset_path)
    X, y = prepare_dataset(df)
    X = X[feature_columns].fillna(0)
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    # Classification report
    print("\n" + "="*55)
    print("CLASSIFICATION REPORT")
    print("="*55)
    print(classification_report(
        y, y_pred,
        target_names=["safe", "warning", "high_risk"]
    ))

    # Confusion matrix
    print("CONFUSION MATRIX")
    print("="*55)
    cm = confusion_matrix(y, y_pred)
    labels = ["safe", "warning", "high_risk"]
    print(f"{'':12}", end="")
    for l in labels:
        print(f"{l:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{labels[i]:12}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()

    # PR-AUC per class (more meaningful than ROC for imbalanced data)
    print("\nPR-AUC PER CLASS")
    print("="*55)
    for i, label in enumerate(labels):
        binary_y = (y == i).astype(int)
        ap = average_precision_score(binary_y, y_proba[:, i])
        print(f"  {label:12}: {ap:.3f}")

    # High risk specifically — what threshold catches 90% of incidents?
    print("\nHIGH RISK THRESHOLD ANALYSIS")
    print("="*55)
    binary_y = (y == 2).astype(int)
    prec, rec, thresholds = precision_recall_curve(binary_y, y_proba[:, 2])
    print(f"{'Threshold':>12} {'Precision':>12} {'Recall':>12}")
    print("-"*38)
    for t, p, r in zip(thresholds[::10], prec[::10], rec[::10]):
        print(f"{t:>12.2f} {p:>12.3f} {r:>12.3f}")

    # Feature importance
    print("\nTOP 10 MOST IMPORTANT FEATURES")
    print("="*55)
    importance = model.feature_importances_
    feat_imp = sorted(
        zip(feature_columns, importance),
        key=lambda x: x[1], reverse=True
    )[:10]
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 200)
        print(f"  {feat:35} {imp:.4f} {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/final_dataset.parquet")
    parser.add_argument("--model-dir", default="ml/model")
    args = parser.parse_args()
    evaluate(args.dataset, args.model_dir)