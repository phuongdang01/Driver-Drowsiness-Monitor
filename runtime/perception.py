from __future__ import annotations

from dataclasses import dataclass, field
import math
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


@dataclass
class RawPerception:
    face_detected: bool
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    # Pixel points used only for drawing visual detection guides on the web video.
    # The decision logic still uses EAR/MAR/head-pose values above.
    left_eye_points: list[tuple[int, int]] = field(default_factory=list)
    right_eye_points: list[tuple[int, int]] = field(default_factory=list)
    mouth_points: list[tuple[int, int]] = field(default_factory=list)
    head_pose_points: dict[str, tuple[int, int]] = field(default_factory=dict)


def _eye_aspect_ratio(eye: list[tuple[int, int]]) -> float:
    a = math.hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    b = math.hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    c = math.hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def _mouth_aspect_ratio(mouth: list[tuple[int, int]]) -> float:
    a = math.hypot(mouth[2][0] - mouth[3][0], mouth[2][1] - mouth[3][1])
    b = math.hypot(mouth[0][0] - mouth[1][0], mouth[0][1] - mouth[1][1])
    if b == 0:
        return 0.0
    return a / b


def _estimate_head_pose(landmarks, img_shape) -> tuple[float, float, float]:
    h, w = img_shape[:2]
    image_points = np.array(
        [
            (landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h),
            (landmarks[CHIN].x * w, landmarks[CHIN].y * h),
            (landmarks[LEFT_EYE_CORNER].x * w, landmarks[LEFT_EYE_CORNER].y * h),
            (landmarks[RIGHT_EYE_CORNER].x * w, landmarks[RIGHT_EYE_CORNER].y * h),
            (landmarks[LEFT_MOUTH].x * w, landmarks[LEFT_MOUTH].y * h),
            (landmarks[RIGHT_MOUTH].x * w, landmarks[RIGHT_MOUTH].y * h),
        ],
        dtype="double",
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ]
    )

    focal_length = float(w)
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype="double",
    )
    dist_coeffs = np.zeros((4, 1))

    ok, rotation_vector, _translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pitch = math.degrees(math.atan2(rotation_matrix[2][1], rotation_matrix[2][2]))
    yaw = math.degrees(
        math.atan2(
            -rotation_matrix[2][0],
            math.sqrt(rotation_matrix[2][1] ** 2 + rotation_matrix[2][2] ** 2),
        )
    )
    roll = math.degrees(math.atan2(rotation_matrix[1][0], rotation_matrix[0][0]))
    return pitch, yaw, roll


class PerceptionExtractor:
    def __init__(self) -> None:
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process(self, frame) -> RawPerception:
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return RawPerception(face_detected=False)

        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        def get_xy(i: int) -> tuple[int, int]:
            return int(lm[i].x * w), int(lm[i].y * h)

        def get_xy_float(i: int) -> tuple[float, float]:
            return float(lm[i].x * w), float(lm[i].y * h)

        left_eye_pts = [get_xy(i) for i in LEFT_EYE]
        right_eye_pts = [get_xy(i) for i in RIGHT_EYE]
        mouth_pts = [get_xy(i) for i in MOUTH]

        # Use floating-point landmark coordinates for EAR/MAR. Integer pixel
        # rounding can make the closed-mouth vertical distance become exactly 0
        # on phone frames, which then forces MAR threshold to the same floor.
        left_eye_float = [get_xy_float(i) for i in LEFT_EYE]
        right_eye_float = [get_xy_float(i) for i in RIGHT_EYE]
        mouth_float = [get_xy_float(i) for i in MOUTH]

        left_ear = _eye_aspect_ratio(left_eye_float)
        right_ear = _eye_aspect_ratio(right_eye_float)
        ear = (left_ear + right_ear) / 2.0
        mar = _mouth_aspect_ratio(mouth_float)

        try:
            pitch, yaw, roll = _estimate_head_pose(lm, frame.shape)
        except Exception:
            pitch, yaw, roll = 0.0, 0.0, 0.0

        head_points = {
            "nose": get_xy(NOSE_TIP),
            "chin": get_xy(CHIN),
            "left_eye": get_xy(LEFT_EYE_CORNER),
            "right_eye": get_xy(RIGHT_EYE_CORNER),
            "left_mouth": get_xy(LEFT_MOUTH),
            "right_mouth": get_xy(RIGHT_MOUTH),
        }

        return RawPerception(
            face_detected=True,
            ear=ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            left_eye_points=left_eye_pts,
            right_eye_points=right_eye_pts,
            mouth_points=mouth_pts,
            head_pose_points=head_points,
        )

    def close(self) -> None:
        if hasattr(self.face_mesh, "close"):
            self.face_mesh.close()
