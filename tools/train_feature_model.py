from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from runtime.engines.ml_engine import FEATURE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RandomForest DMS feature-fusion model.")
    parser.add_argument("--data", default="data/features_labeled.csv")
    parser.add_argument("--output", default="models/drowsiness_rf.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in FEATURE_NAMES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training CSV: {missing}")

    df = df.dropna(subset=FEATURE_NAMES + ["label"])
    X = df[FEATURE_NAMES]
    y = df["label"].astype(str)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\n=== Confusion matrix ===")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())), index=sorted(y.unique()), columns=sorted(y.unique())))

    importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print("\n=== Feature importances ===")
    print(importances)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "labels": sorted(y.unique()),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "feature_importances": importances.to_dict(),
    }, out)
    print(f"\n[OK] Saved model to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
