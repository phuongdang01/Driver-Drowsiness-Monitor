from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InputConfig:
    source: str = "webcam"
    video_path: str = "Video Database\\Sub 03.avi"
    loop_file: bool = True


@dataclass
class RuntimeTuning:
    fps: float = 30.0
    warmup_seconds: float = 3.0
    calibration_frames: int = 60


@dataclass
class ThresholdConfig:
    # DYNAMIC THRESHOLDS - Computed by algorithms, not hardcoded
    # See eye_detection.py for DynamicEAR, EWMAControlChart, PERCLOS algorithms
    
    ear_default: float = 0.23  # Initial baseline (will be learned during calibration)
    ear_calibration_factor: float = 0.78  # Adaptive threshold = open-eye baseline * this factor
    ear_min: float = 0.10  # Safety floor; low enough so threshold is not always stuck at 0.15
    ear_max: float = 0.35  # Safety ceiling
    
    # MAR & Pitch - Not hardcoded, used by FSM for evidence scoring
    mar: float = 0.60  # Legacy/static fallback only
    mar_min: float = 0.15  # MAR floor from the original DynamicMAR code
    mar_max: float = 0.80  # Dynamic MAR threshold ceiling
    mar_factor: float = 0.7  # MAR factor from the original DynamicMAR code
    mar_window: int = 150  # Sliding window from the original DynamicMAR code
    mar_gap: float = 0.04  # Minimum personalized gap above closed-mouth baseline
    mar_adapt_alpha: float = 0.01  # Slow adaptation only when mouth is closed
    pitch: float = 20.0  # Head nod threshold (used for context)
    head_yaw: float = 30.0  # Head turn threshold
    head_nod_frames: int = 30  # Frames for head nod detection
    
    # Temporal thresholds (used for counting events)
    blink_freq: int = 10  # Normal blink frequency (events/minute)
    yawn_frames: int = 20  # Duration to register yawn
    max_yawns_window: int = 3  # Max yawns per monitoring window
    
    # NOTE: PERCLOS, EAR dynamic threshold, and EWMA control limits
    # are computed in eye_detection.py using SQAD, EWMA algorithms.
    # These are NOT hardcoded here to allow adaptive threshold learning.


@dataclass
class WindowConfig:
    perclos_seconds: float = 60.0
    perclos_short_seconds: float = 5.0
    yawn_window_seconds: float = 60.0
    blink_window_seconds: float = 10.0


@dataclass
class AlertPolicy:
    drowsy_cooldown_seconds: float = 5.0


@dataclass
class RuntimeConfig:
    input: InputConfig = field(default_factory=InputConfig)
    runtime: RuntimeTuning = field(default_factory=RuntimeTuning)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    alerts: AlertPolicy = field(default_factory=AlertPolicy)
    decision_engine: str = "fsm"
    enable_legacy_feature_overlay: bool = False
    display_window: bool = True
    log_every_n_frames: int = 100


CLI_OVERRIDE_MAP = {
    "source": "input.source",
    "video_path": "input.video_path",
    "loop_file": "input.loop_file",
    "decision_engine": "decision_engine",
    "enable_legacy_feature_overlay": "enable_legacy_feature_overlay",
    "display_window": "display_window",
}


def default_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _load_config_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON root must be an object.")
    return data


def _to_runtime_config(data: dict[str, Any]) -> RuntimeConfig:
    input_cfg = InputConfig(**data.get("input", {}))
    runtime_cfg = RuntimeTuning(**data.get("runtime", {}))
    threshold_cfg = ThresholdConfig(**data.get("thresholds", {}))
    window_cfg = WindowConfig(**data.get("windows", {}))
    alert_cfg = AlertPolicy(**data.get("alerts", {}))

    return RuntimeConfig(
        input=input_cfg,
        runtime=runtime_cfg,
        thresholds=threshold_cfg,
        windows=window_cfg,
        alerts=alert_cfg,
        decision_engine=data.get("decision_engine", "fsm"),
        enable_legacy_feature_overlay=bool(data.get("enable_legacy_feature_overlay", False)),
        display_window=bool(data.get("display_window", True)),
        log_every_n_frames=int(data.get("log_every_n_frames", 100)),
    )


def load_runtime_config(config_path: str | None, cli_overrides: dict[str, Any] | None = None) -> RuntimeConfig:
    base = asdict(default_runtime_config())
    json_cfg = _load_config_json(config_path)
    _deep_merge_dict(base, json_cfg)

    if cli_overrides:
        for cli_key, cli_value in cli_overrides.items():
            if cli_key not in CLI_OVERRIDE_MAP:
                continue
            if cli_value is None:
                continue
            _set_nested(base, CLI_OVERRIDE_MAP[cli_key], cli_value)

    return _to_runtime_config(base)


def config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    return asdict(config)
