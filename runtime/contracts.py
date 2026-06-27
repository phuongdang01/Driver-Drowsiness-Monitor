from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fsm import DrowsinessState


@dataclass
class EngineContext:
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    state: DrowsinessState
    evidence: float
    reasons: list[str] = field(default_factory=list)
    alert_sound: str = "none"
    color: tuple[int, int, int] = (0, 255, 0)
    label: str = "ALERT"
    debug: dict[str, Any] = field(default_factory=dict)
