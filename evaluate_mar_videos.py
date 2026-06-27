"""Utilities for visualizing MAR/yawn status on frames."""

from __future__ import annotations

import cv2


def draw_mar_hud(frame, mar_status, x: int = 10, y: int = 150):
    """Draw compact MAR status. Expects fields: mu, threshold, is_open."""
    color = (0, 0, 255) if getattr(mar_status, "is_open", False) else (0, 255, 255)
    txt = f"MAR {getattr(mar_status, 'mu', 0.0):.3f}/{getattr(mar_status, 'threshold', 0.0):.3f}"
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "YAWN" if getattr(mar_status, "is_open", False) else "MOUTH NORMAL", (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame
