from __future__ import annotations

from fsm import ALERT_CONFIGS, DrowsinessFSM, DrowsinessSignals
from runtime.config import RuntimeConfig
from runtime.contracts import DecisionResult, EngineContext
from runtime.engines.base import DecisionEngine


class FSMDecisionEngine(DecisionEngine):
    name = "fsm"

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.fsm = DrowsinessFSM(
            fps=config.runtime.fps,
            ear_threshold=config.thresholds.ear_default,
            mar_threshold=config.thresholds.mar,
            pitch_threshold=config.thresholds.pitch,
        )

    def initialize(self, context: EngineContext) -> None:
        self.fsm.fps = context.fps

    def update(self, signals: DrowsinessSignals) -> DecisionResult:
        state = self.fsm.update(signals)
        alert_cfg = ALERT_CONFIGS[state]

        reasons: list[str] = []
        if signals.ear_below_threshold:
            reasons.append("EAR_BELOW_THRESHOLD")
        if signals.mar_above_threshold:
            reasons.append("MAR_ABOVE_THRESHOLD")
        if signals.head_nod_detected:
            reasons.append("HEAD_NOD_DETECTED")
        if signals.perclos_short >= 0.60:
            reasons.append("PERCLOS_5S_HIGH")

        return DecisionResult(
            state=state,
            evidence=self.fsm.evidence_score,
            reasons=reasons,
            alert_sound=alert_cfg.sound_type,
            color=alert_cfg.color,
            label=alert_cfg.text,
            debug={
                "perclos": signals.perclos,
                "perclos_short": signals.perclos_short,
            },
        )

    def reset(self) -> None:
        self.fsm.reset()
