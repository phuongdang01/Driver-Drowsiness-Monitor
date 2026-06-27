from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from runtime.config import load_runtime_config
from runtime.features import SignalFeaturePipeline
from runtime.perception import PerceptionExtractor


FEATURE_COLUMNS = [
    "ear", "mar", "pitch", "pitch_velocity", "perclos", "perclos_short",
    "yawn_frequency", "blink_frequency", "head_nod_detected",
    "eyes_closed_consecutive", "ear_below_threshold", "mar_above_threshold", "pitch_above_threshold",
]


def _read_label_segments(labels_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_path", "start_sec", "end_sec", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"labels_csv must contain columns: {sorted(required)}")
        for row in reader:
            rows.append({
                "video_path": row["video_path"],
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "label": row["label"],
            })
    return rows


def _label_at(segments: list[dict[str, Any]], video_path: str, t_sec: float) -> str | None:
    vp_name = Path(video_path).name
    for seg in segments:
        seg_name = Path(seg["video_path"]).name
        if seg["video_path"] == video_path or seg_name == vp_name:
            if seg["start_sec"] <= t_sec < seg["end_sec"]:
                return str(seg["label"])
    return None


def signals_to_row(video_path: str, frame_idx: int, t_sec: float, signals, debug: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "video_path": video_path,
        "frame_idx": frame_idx,
        "t_sec": round(t_sec, 4),
        "ear": float(signals.ear),
        "mar": float(signals.mar),
        "pitch": float(signals.pitch),
        "pitch_velocity": float(signals.pitch_velocity),
        "perclos": float(signals.perclos),
        "perclos_short": float(signals.perclos_short),
        "yawn_frequency": int(signals.yawn_frequency),
        "blink_frequency": int(signals.blink_frequency),
        "head_nod_detected": int(signals.head_nod_detected),
        "eyes_closed_consecutive": int(signals.eyes_closed_consecutive),
        "ear_below_threshold": int(signals.ear_below_threshold),
        "mar_above_threshold": int(signals.mar_above_threshold),
        "pitch_above_threshold": int(signals.pitch_above_threshold),
        "face_detected": int(debug.get("face_detected", False)),
        "ear_threshold": float(debug.get("ear_threshold", 0.0)),
        "label": label,
    }


def process_video(video_path: Path, segments: list[dict[str, Any]], config_path: str | None, sample_every: int) -> list[dict[str, Any]]:
    config = load_runtime_config(config_path, {"source": "file", "video_path": str(video_path), "display_window": False})
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.runtime.fps
    config.runtime.fps = float(fps)

    perception = PerceptionExtractor()
    features = SignalFeaturePipeline(config)
    rows: list[dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue
            t_sec = frame_idx / fps
            label = _label_at(segments, str(video_path), t_sec)
            if label is None:
                frame_idx += 1
                continue
            raw = perception.process(frame)
            # Use video time as "now" so sliding-window features are correct for offline processing.
            signals, debug = features.update(raw, t_sec)
            if not features.state.calibrated:
                frame_idx += 1
                continue
            rows.append(signals_to_row(str(video_path), frame_idx, t_sec, signals, debug, label))
            frame_idx += 1
    finally:
        cap.release()
        perception.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract labeled DMS features from labeled videos.")
    parser.add_argument("--labels", required=True, help="CSV columns: video_path,start_sec,end_sec,label")
    parser.add_argument("--output", default="data/features_labeled.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample-every", type=int, default=1, help="Use 1 for every frame, 2 for every 2 frames, ...")
    args = parser.parse_args()

    labels_csv = Path(args.labels)
    segments = _read_label_segments(labels_csv)
    video_paths = sorted({Path(seg["video_path"]) for seg in segments})

    all_rows: list[dict[str, Any]] = []
    for video_path in video_paths:
        print(f"[INFO] Processing {video_path}")
        all_rows.extend(process_video(video_path, segments, args.config, args.sample_every))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {len(all_rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
