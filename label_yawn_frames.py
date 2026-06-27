"""Interactive frame-level labeling tool for yawn videos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label yawn videos frame-by-frame and save CSV labels.",
    )
    parser.add_argument(
        "--dataset-path",
        default="Video Database/Yawn",
        help="Folder containing input videos.",
    )
    parser.add_argument(
        "--label-path",
        default="Video Database/Yawn/Labels",
        help="Output folder for generated *_labels.csv files.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=30,
        help="Delay in milliseconds per frame for keyboard polling.",
    )
    parser.add_argument(
        "--label-key",
        default="l",
        help="Key pressed while frame is yawning (default: l).",
    )
    parser.add_argument(
        "--skip-key",
        default="s",
        help="Key to skip current video (default: s).",
    )
    parser.add_argument(
        "--quit-key",
        default="q",
        help="Key to quit all labeling (default: q).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files instead of skipping them.",
    )
    return parser


def _key_code(key: str) -> int:
    if not key:
        raise ValueError("Key cannot be empty.")
    return ord(key[0].lower())


def _iter_videos(dataset_path: Path) -> list[Path]:
    videos = [
        p
        for p in sorted(dataset_path.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos


def _save_labels(path: Path, labels: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "label"])
        for idx, label in enumerate(labels):
            writer.writerow([idx, label])


def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset_path)
    label_path = Path(args.label_path)
    label_path.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return 2

    label_key = _key_code(args.label_key)
    skip_key = _key_code(args.skip_key)
    quit_key = _key_code(args.quit_key)

    videos = _iter_videos(dataset_path)
    print("--- YAWN LABELING TOOL ---")
    print(f"Dataset: {dataset_path}")
    print(f"Videos found: {len(videos)}")
    print(f"Hold '{args.label_key[0]}' => yawning, release => normal")
    print(f"Press '{args.skip_key[0]}' => skip video, '{args.quit_key[0]}' => quit")

    for video_file in videos:
        out_file = label_path / f"{video_file.name}_labels.csv"
        if out_file.exists() and not args.overwrite:
            print(f"Skip existing labels: {out_file.name}")
            continue

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Cannot open video: {video_file}")
            continue

        labels: list[int] = []
        print(f"Labeling: {video_file.name}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key == quit_key:
                cap.release()
                cv2.destroyAllWindows()
                _save_labels(out_file, labels)
                print(f"Saved partial labels: {out_file.name} ({len(labels)} frames)")
                print("Exiting by user request.")
                return 0
            if key == skip_key:
                break

            is_yawn = 1 if key == label_key else 0
            labels.append(is_yawn)

            status = "YAWNING" if is_yawn else "NORMAL"
            color = (0, 0, 255) if is_yawn else (0, 255, 0)
            frame = frame.copy()
            cv2.putText(
                frame,
                f"Status: {status}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
            )
            cv2.imshow("Yawn Labeling Tool", frame)

        cap.release()
        _save_labels(out_file, labels)
        print(f"Saved labels: {out_file.name} ({len(labels)} frames)")

    cv2.destroyAllWindows()
    print("Completed labeling pass.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
