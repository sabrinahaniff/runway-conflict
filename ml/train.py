import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from .features import prepare_dataset


def train(dataset_path: str, output_dir: str = "ml/model"):
    print(f"Loading {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df):,} rows")

    X, y = prepare_dataset(df)
    print(f"Features: {X.shape[1]} columns")

    # Split by scenario ID — never by row
    groups = df["scenario_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Class weights
    class_counts = np.bincount(y_train)
    total = len(y_train)
    weights = {i: total / (len(class_counts) * c)
               for i, c in enumerate(class_counts)}
    sample_weights = np.array([weights[l] for l in y_train])
    print(f"Class weights: {weights}")

    # Train
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_s, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_s, y_test)],
        verbose=50,
    )

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_model(f"{output_dir}/xgboost.json")
    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{output_dir}/feature_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"\nSaved to {output_dir}/")

    y_pred = model.predict(X_test_s)
    acc = (y_pred == y_test.values).mean()
    print(f"Test accuracy: {acc:.3f}")

    return model, scaler, X_test_s, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/final_dataset.parquet")
    parser.add_argument("--output", default="ml/model")
    args = parser.parse_args()
    train(args.dataset, args.output)