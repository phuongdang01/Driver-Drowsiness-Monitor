from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class VideoTransport:
    source: str
    video_path: str
    loop_file: bool = True

    def __post_init__(self) -> None:
        if self.source not in {"webcam", "file"}:
            raise ValueError(f"Unsupported source: {self.source}")
        target = 0 if self.source == "webcam" else self.video_path
        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            if self.source == "file":
                raise RuntimeError(f"Cannot open video file {self.video_path}")
            raise RuntimeError("Cannot open webcam device 0")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Some webcams return 0 or absurd values; keep a safe runtime default.
        self.fps = fps if 1.0 <= fps <= 120.0 else 30.0

    def read(self) -> tuple[bool, object]:
        ret, frame = self.cap.read()
        if ret:
            return True, frame
        if self.source == "file" and self.loop_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            return ret, frame
        return False, frame

    def close(self) -> None:
        if self.cap:
            self.cap.release()
