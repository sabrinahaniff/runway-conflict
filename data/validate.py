import argparse
import numpy as np
import pandas as pd

LABEL_BOUNDS = {
    "safe":      (0.50, 0.80),
    "warning":   (0.10, 0.35),
    "high_risk": (0.05, 0.25),
}


def validate(df):
    passed = 0
    failed = 0

    def ok(msg):
        nonlocal passed
        print(f"  PASS  {msg}")
        passed += 1

    def fail(msg):
        nonlocal failed
        print(f"  FAIL  {msg}")
        failed += 1

    print(f"\n{'='*50}")
    print(f"DATASET VALIDATION — {len(df):,} rows")
    print(f"{'='*50}\n")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [c for c in numeric_cols if np.isinf(df[c]).any()]
    if inf_cols:
        fail(f"Infinite values in: {inf_cols}")
    else:
        ok("No infinite values")

    null_count = df[["cpa_distance", "risk_score", "risk_level"]].isnull().sum().sum()
    if null_count > 0:
        fail(f"{null_count} null values in key columns")
    else:
        ok("No nulls in key columns")

    dist = df["risk_level"].value_counts(normalize=True)
    print(f"\nLabel distribution:")
    for label, (lo, hi) in LABEL_BOUNDS.items():
        actual = dist.get(label, 0.0)
        in_range = lo <= actual <= hi
        status = "PASS" if in_range else "FAIL"
        print(f"  {status}  {label}: {actual:.1%} (expected {lo:.0%}–{hi:.0%})")
        if in_range:
            passed += 1
        else:
            failed += 1

    if "cpa_distance" in df.columns:
        mean_cpa = df.groupby("risk_level")["cpa_distance"].mean()
        hr = mean_cpa.get("high_risk", 999)
        safe = mean_cpa.get("safe", 0)
        if hr < safe:
            ok(f"CPA correct: high_risk={hr:.0f}m < safe={safe:.0f}m")
        else:
            fail(f"CPA wrong: high_risk={hr:.0f}m >= safe={safe:.0f}m")

    print(f"\n{'='*50}")
    print(f"{passed}/{passed+failed} checks passed")
    if failed == 0:
        print("Dataset looks good — proceed to training")
    else:
        print(f"Fix {failed} issue(s) before training")
    print(f"{'='*50}\n")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.dataset)
    validate(df)