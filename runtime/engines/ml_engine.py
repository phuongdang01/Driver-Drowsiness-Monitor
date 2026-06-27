from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fsm import ALERT_CONFIGS, DrowsinessFSM, DrowsinessSignals, DrowsinessState
from runtime.config import RuntimeConfig
from runtime.contracts import DecisionResult, EngineContext
from runtime.engines.base import DecisionEngine

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


FEATURE_NAMES = [
    "ear",
    "mar",
    "pitch",
    "pitch_velocity",
    "perclos",
    "perclos_short",
    "yawn_frequency",
    "blink_frequency",
    "head_nod_detected",
    "eyes_closed_consecutive",
    "ear_below_threshold",
    "mar_above_threshold",
    "pitch_above_threshold",
]


def signals_to_vector(signals: DrowsinessSignals) -> np.ndarray:
    values = [
        signals.ear,
        signals.mar,
        signals.pitch,
        signals.pitch_velocity,
        signals.perclos,
        signals.perclos_short,
        signals.yawn_frequency,
        signals.blink_frequency,
        int(signals.head_nod_detected),
        signals.eyes_closed_consecutive,
        int(signals.ear_below_threshold),
        int(signals.mar_above_threshold),
        int(signals.pitch_above_threshold),
    ]
    return np.asarray(values, dtype=float).reshape(1, -1)


class MLDecisionEngine(DecisionEngine):
    """Random-Forest feature-fusion engine.

    If models/drowsiness_rf.joblib is not found, the engine falls back to the explainable FSM.
    """

    name = "ml"

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.model_bundle: dict[str, Any] | None = None
        self.model = None
        self.labels: list[str] = [s.value for s in DrowsinessState]
        self.fallback_fsm = DrowsinessFSM(
            fps=config.runtime.fps,
            ear_threshold=config.thresholds.ear_default,
            mar_threshold=config.thresholds.mar,
            pitch_threshold=config.thresholds.pitch,
        )
        self.model_path = Path("models/drowsiness_rf.joblib")

    def initialize(self, context: EngineContext) -> None:
        self.fallback_fsm.fps = context.fps
        if joblib is None:
            return
        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self.model_bundle = bundle
                self.model = bundle["model"]
                self.labels = list(bundle.get("labels", self.labels))
            else:
                self.model = bundle

    def update(self, signals: DrowsinessSignals) -> DecisionResult:
        # Safety fallback: keep rule-based FSM if model is absent.
        if self.model is None:
            state = self.fallback_fsm.update(signals)
            cfg = ALERT_CONFIGS[state]
            return DecisionResult(
                state=state,
                evidence=self.fallback_fsm.evidence_score,
                reasons=["ML_MODEL_NOT_FOUND_FALLBACK_FSM"],
                alert_sound=cfg.sound_type,
                color=cfg.color,
                label=f"ML-FALLBACK {cfg.text}",
                debug={"model_path": str(self.model_path)},
            )

        x = signals_to_vector(signals)
        pred = self.model.predict(x)[0]
        label = str(pred)
        try:
            state = DrowsinessState(label)
        except ValueError:
            state = DrowsinessState.ALERT

        evidence = 0.0
        debug: dict[str, Any] = {"ml_label": label}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            classes = [str(c) for c in getattr(self.model, "classes_", [])]
            debug["probabilities"] = {classes[i]: float(proba[i]) for i in range(len(classes))}
            risk_weights = {
                DrowsinessState.ALERT.value: 0.0,
                DrowsinessState.SUSPICIOUS.value: 0.45,
                DrowsinessState.DROWSY.value: 0.75,
                DrowsinessState.CRITICAL.value: 1.0,
            }
            evidence = float(sum(risk_weights.get(classes[i], 0.0) * proba[i] for i in range(len(classes))))
        else:
            evidence = {
                DrowsinessState.ALERT: 0.0,
                DrowsinessState.SUSPICIOUS: 0.45,
                DrowsinessState.DROWSY: 0.75,
                DrowsinessState.CRITICAL: 1.0,
            }[state]

        cfg = ALERT_CONFIGS[state]
        reasons = ["ML_MODEL"]
        if signals.ear_below_threshold:
            reasons.append("EAR_BELOW_THRESHOLD")
        if signals.perclos_short >= 0.60:
            reasons.append("PERCLOS_5S_HIGH")
        if signals.head_nod_detected:
            reasons.append("HEAD_NOD")
        if signals.mar_above_threshold:
            reasons.append("MAR_HIGH")

        return DecisionResult(
            state=state,
            evidence=evidence,
            reasons=reasons,
            alert_sound=cfg.sound_type,
            color=cfg.color,
            label=f"ML {cfg.text}",
            debug=debug,
        )

    def reset(self) -> None:
        self.fallback_fsm.reset()
