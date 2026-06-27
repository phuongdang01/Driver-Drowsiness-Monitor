from __future__ import annotations

import argparse
import time

import cv2

from fsm import DrowsinessState
from legacy_feature_overlay import DrowsinessFeatureExtractor, format_overlay_lines
from runtime.alerts import AudioAlertController
from runtime.config import RuntimeConfig, load_runtime_config
from runtime.contracts import DecisionResult, EngineContext
from runtime.engines.registry import available_engines, create_engine
from runtime.features import SignalFeaturePipeline
from runtime.perception import PerceptionExtractor
from runtime.transports import VideoTransport


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Advanced FSM-based drowsiness detection runtime.")
    parser.add_argument("--config", default=None, help="Path to JSON config file.")
    parser.add_argument("--source", choices=["webcam", "file"], default=None, help="Video source selection.")
    parser.add_argument("--video-path", default=None, help="Video path when --source file is selected.")
    parser.add_argument(
        "--decision-engine",
        choices=available_engines(),
        default=None,
        help="Decision engine key from python registry map.",
    )
    parser.add_argument(
        "--enable-legacy-feature-overlay",
        dest="enable_legacy_feature_overlay",
        action="store_true",
        default=None,
        help="Enable optional bundle-style telemetry overlay.",
    )
    parser.add_argument(
        "--disable-legacy-feature-overlay",
        dest="enable_legacy_feature_overlay",
        action="store_false",
        help="Disable legacy telemetry overlay.",
    )
    parser.add_argument(
        "--display-window",
        dest="display_window",
        action="store_true",
        default=None,
        help="Force GUI display window on.",
    )
    parser.add_argument(
        "--no-display",
        dest="display_window",
        action="store_false",
        help="Run without GUI window.",
    )
    return parser


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key in (
        "source",
        "video_path",
        "decision_engine",
        "enable_legacy_feature_overlay",
        "display_window",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    return overrides


def _calibration_result(remaining: float) -> DecisionResult:
    label = f"CALIBRATING {remaining:.1f}s"
    return DecisionResult(
        state=DrowsinessState.ALERT,
        evidence=0.0,
        reasons=["WARMUP"],
        alert_sound="none",
        color=(0, 255, 255),
        label=label,
        debug={},
    )


def run_runtime(config: RuntimeConfig) -> int:
    print("[INFO] Advanced Drowsiness Detection System Started")
    print(f"[INFO] Decision engine: {config.decision_engine}")
    print(f"[INFO] Source: {config.input.source}")
    if config.input.source == "file":
        print(f"[INFO] Video path: {config.input.video_path}")
    print(f"[INFO] Legacy overlay: {config.enable_legacy_feature_overlay}")

    transport = VideoTransport(
        source=config.input.source,
        video_path=config.input.video_path,
        loop_file=config.input.loop_file,
    )
    perception = PerceptionExtractor()
    features = SignalFeaturePipeline(config)
    engine = create_engine(config.decision_engine, config)
    engine.initialize(EngineContext(fps=config.runtime.fps, metadata={"engine": config.decision_engine}))

    alerts = AudioAlertController(
        sound_file="alert.wav",
        drowsy_cooldown_seconds=config.alerts.drowsy_cooldown_seconds,
    )

    legacy_overlay = None
    if config.enable_legacy_feature_overlay:
        legacy_overlay = DrowsinessFeatureExtractor(window_duration=60.0)

    start_time = time.time()
    warmup_start = start_time
    frame_count = 0

    try:
        while True:
            ok, frame = transport.read()
            if not ok:
                break
            frame_count += 1
            now = time.time()

            raw = perception.process(frame)
            signals, debug = features.update(raw, now)

            calibrated = bool(debug.get("calibrated", False))
            warmup_elapsed = now - warmup_start
            warmup_done = calibrated and warmup_elapsed >= config.runtime.warmup_seconds

            if warmup_done:
                result = engine.update(signals)
            else:
                remaining = max(0.0, config.runtime.warmup_seconds - warmup_elapsed)
                result = _calibration_result(remaining)

            alerts.update(result.alert_sound)

            if legacy_overlay is not None:
                legacy_overlay.update(
                    ear=float(signals.ear),
                    mar=float(signals.mar),
                    pitch=float(signals.pitch),
                    yaw=float(debug.get("rel_yaw", 0.0)),
                    roll=float(raw.roll),
                    is_blink=bool(signals.blink_frequency > 0),
                    is_yawn=bool(signals.mar_above_threshold),
                    is_nod=bool(signals.head_nod_detected),
                )

            if config.display_window:
                _draw_overlay(frame, result, config, frame_count, signals, debug, legacy_overlay)
                cv2.imshow("Advanced Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if config.log_every_n_frames > 0 and frame_count % config.log_every_n_frames == 0:
                print(
                    "[INFO] frame=%d state=%s evidence=%.2f perclos=%.2f perclos5=%.2f"
                    % (frame_count, result.state.value, result.evidence, signals.perclos, signals.perclos_short)
                )

    finally:
        alerts.close()
        perception.close()
        transport.close()
        if config.display_window:
            cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - start_time)
    print(f"[INFO] System stopped. Frames={frame_count}, elapsed={elapsed:.2f}s, avg_fps={frame_count/elapsed:.2f}")
    return 0


def _draw_overlay(frame, result, config, frame_count, signals, debug, legacy_overlay) -> None:
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"{result.label}",
        (10, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        result.color,
        3,
    )
    cv2.putText(frame, f"Engine: {config.decision_engine}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    cv2.putText(frame, f"Evidence: {result.evidence:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    cv2.putText(frame, f"EAR: {signals.ear:.3f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(frame, f"MAR: {signals.mar:.3f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(frame, f"PERCLOS: {signals.perclos:.2f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(frame, f"PERCLOS_5s: {signals.perclos_short:.2f}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(
        frame,
        f"Blink/yawn: {signals.blink_frequency}/{signals.yawn_frequency}",
        (10, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
    )
    cv2.putText(frame, f"Pitch: {signals.pitch:.1f}", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    calibration_text = "CALIBRATED" if debug.get("calibrated", False) else "CALIBRATING"
    cv2.putText(frame, f"{calibration_text} | Frame: {frame_count}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if result.reasons:
        cv2.putText(
            frame,
            "Reasons: " + ", ".join(result.reasons[:3]),
            (10, h - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 255),
            1,
        )

    if legacy_overlay is not None:
        lines = format_overlay_lines(legacy_overlay.get_features())
        x0 = max(10, w - 320)
        y0 = 30
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x0, y0 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 240, 180), 1)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_runtime_config(args.config, _build_cli_overrides(args))
    return run_runtime(config)


if __name__ == "__main__":
    raise SystemExit(main())
