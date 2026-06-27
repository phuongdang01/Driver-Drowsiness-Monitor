"""Analyze MAR thresholds and provide DynamicMAR for adaptive yawn detection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from runtime.mar_detection import DynamicMAR, DynamicMARStatus


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MAR threshold sweep using KMeans-derived labels.")
    parser.add_argument("--input-csv", default="tools/evaluation/metadata/mar_result/mar_result.csv")
    parser.add_argument("--output-csv", default="tools/evaluation/metadata/reports/mar_threshold_analysis.csv")
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-end", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    return parser


def threshold_range(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    cur = start
    while cur <= end + 1e-9:
        values.append(round(cur, 6))
        cur += step
    return values


def load_mar_values(input_csv: Path) -> np.ndarray:
    vals: list[float] = []
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "MAR" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must contain a MAR column.")
        for row in reader:
            vals.append(float(row["MAR"]))
    if not vals:
        raise ValueError("Input CSV contains no MAR rows.")
    return np.asarray(vals, dtype=float)


def build_pseudo_labels(mar_values: np.ndarray) -> np.ndarray:
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(mar_values.reshape(-1, 1))
    open_cluster = int(kmeans.cluster_centers_.flatten().argmax())
    return (kmeans.labels_ == open_cluster).astype(int)


def run(args: argparse.Namespace) -> int:
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    mar_values = load_mar_values(input_csv)
    y_true = build_pseudo_labels(mar_values)

    rows = []
    for thr in threshold_range(args.threshold_start, args.threshold_end, args.threshold_step):
        y_pred = (mar_values > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "Threshold": f"{thr:.3f}",
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Accuracy": f"{accuracy_score(y_true, y_pred):.2%}",
            "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.3f}",
            "Recall": f"{recall_score(y_true, y_pred, zero_division=0):.3f}",
            "F1-Score": f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
        })

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    for r in rows:
        print(r)
    print(f"Saved: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
