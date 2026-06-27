from __future__ import annotations

import argparse
import base64
import copy
import json
import threading
import time
import socket
import uuid
from dataclasses import asdict
from pathlib import Path
from werkzeug.utils import secure_filename
from typing import Any, Optional

import numpy as np

import cv2
from flask import Flask, Response, jsonify, render_template, request

from fsm import DrowsinessState
from runtime.alerts import AudioAlertController
from runtime.config import RuntimeConfig, config_to_dict, load_runtime_config
from runtime.contracts import DecisionResult, EngineContext
from runtime.engines.registry import available_engines, create_engine
from runtime.features import SignalFeaturePipeline
from runtime.perception import PerceptionExtractor
from runtime.transports import VideoTransport


app = Flask(__name__)
UPLOAD_DIR = Path(__file__).resolve().parent / 'uploaded_videos'
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.webm'}


def _encode_jpeg_b64(frame, quality: int = 80) -> str:
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(jpg.tobytes()).decode("ascii")


def _decode_data_url_image(data_url: str):
    if not data_url:
        raise ValueError("empty image")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    data = base64.b64decode(data_url)
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cannot decode image")
    return frame


def _draw_detection_only(frame, raw):
    """Clean video overlay: only eyes, mouth, head-pose guide, and centered no-face text."""
    out = frame.copy()
    h, w = out.shape[:2]

    def draw_points(points, color, closed=True):
        if not points:
            return
        pts = [(int(x), int(y)) for x, y in points]
        for pt in pts:
            cv2.circle(out, pt, max(1, int(w / 500)), color, -1, cv2.LINE_AA)
        if len(pts) >= 2:
            cv2.polylines(out, [np.array(pts, dtype=np.int32)], closed, color, max(1, int(w / 420)), cv2.LINE_AA)

    if not getattr(raw, "face_detected", False):
        text = "NO FACE DETECTED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.55, min(w, h) / 650.0)
        thickness = max(2, int(min(w, h) / 320))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(8, (w - tw) // 2)
        y = max(th + 8, (h + th) // 2)
        cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
        return out

    draw_points(getattr(raw, "left_eye_points", []), (255, 210, 40), closed=True)
    draw_points(getattr(raw, "right_eye_points", []), (255, 210, 40), closed=True)

    mouth = getattr(raw, "mouth_points", [])
    draw_points(mouth, (0, 220, 255), closed=True)
    if len(mouth) >= 4:
        cv2.line(out, mouth[0], mouth[1], (0, 220, 255), max(1, int(w / 420)), cv2.LINE_AA)
        cv2.line(out, mouth[2], mouth[3], (0, 220, 255), max(1, int(w / 420)), cv2.LINE_AA)

    hp = getattr(raw, "head_pose_points", {}) or {}
    nose = hp.get("nose")
    chin = hp.get("chin")
    le = hp.get("left_eye")
    re = hp.get("right_eye")
    lm = hp.get("left_mouth")
    rm = hp.get("right_mouth")
    thickness = max(1, int(w / 430))
    if le and re:
        cv2.line(out, le, re, (80, 255, 80), thickness, cv2.LINE_AA)
    if lm and rm:
        cv2.line(out, lm, rm, (80, 255, 80), thickness, cv2.LINE_AA)
    if nose and chin:
        cv2.line(out, nose, chin, (180, 100, 255), thickness, cv2.LINE_AA)
    if nose:
        scale = max(18, int(min(w, h) * 0.12))
        end_x = int(nose[0] + max(-1.0, min(1.0, raw.yaw / 35.0)) * scale)
        end_y = int(nose[1] - max(-1.0, min(1.0, raw.pitch / 35.0)) * scale)
        cv2.arrowedLine(out, nose, (end_x, end_y), (255, 120, 120), thickness, cv2.LINE_AA, tipLength=0.25)
    return out


def _status_payload(result: DecisionResult, signals, debug: dict[str, Any], fps: float, frame_count: int, engine: str, source: str = "browser_camera") -> dict[str, Any]:
    return {
        "running": True,
        "state": result.state.value,
        "label": result.label,
        "evidence": round(float(result.evidence), 4),
        "reasons": result.reasons,
        "alert_sound": result.alert_sound,
        "metrics": {
            "ear": round(float(signals.ear), 4),
            # Display the same smoothed MAR used by DynamicMAR for threshold comparison.
            "mar": round(float(debug.get("mar_dynamic_mu", signals.mar)), 4),
            "pitch": round(float(signals.pitch), 2),
            "pitch_velocity": round(float(signals.pitch_velocity), 2),
            "perclos": round(float(signals.perclos), 4),
            "perclos_short": round(float(signals.perclos_short), 4),
            "blink_frequency": int(signals.blink_frequency),
            "yawn_frequency": int(signals.yawn_frequency),
            "eyes_closed_consecutive": int(signals.eyes_closed_consecutive),
            "head_nod_detected": bool(signals.head_nod_detected),
            "ear_threshold": round(float(debug.get("ear_threshold", 0.0)), 4),
            "ear_baseline": round(float(debug.get("ear_baseline", 0.0)), 4),
            "mar_threshold": round(float(debug.get("mar_threshold", 0.0)), 4),
            "mar_raw": round(float(debug.get("mar_raw", 0.0)), 4),
            "face_detected": bool(debug.get("face_detected", False)),
            "calibrated": bool(debug.get("calibrated", False)),
        },
        "flags": {
            "ear_below_threshold": bool(signals.ear_below_threshold),
            "mar_above_threshold": bool(signals.mar_above_threshold),
            "perclos_high": bool(signals.perclos >= 0.20),
            "perclos_short_high": bool(signals.perclos_short >= 0.60),
            "blink_high": bool(signals.blink_frequency > 10),
            "yawn_high": bool(signals.yawn_frequency >= 3 or signals.mar_above_threshold),
            "pitch_high": bool(signals.pitch_above_threshold),
            "closed_high": bool(signals.eyes_closed_consecutive >= 15),
            "head_nod": bool(signals.head_nod_detected),
        },
        "debug": _json_safe(debug),
        "fps": round(float(fps), 2),
        "frame_count": int(frame_count),
        "source": source,
        "engine": engine,
        "available_engines": available_engines(),
    }


class DMSWebRuntime:
    """Run the full DMS pipeline in one background thread.

    Pipeline:
    VideoTransport -> PerceptionExtractor -> SignalFeaturePipeline -> DecisionEngine -> AudioAlertController
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.lock = threading.RLock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.transport: Optional[VideoTransport] = None
        self.perception: Optional[PerceptionExtractor] = None
        self.features: Optional[SignalFeaturePipeline] = None
        self.engine = None
        self.alerts: Optional[AudioAlertController] = None

        self.latest_frame = None
        self.latest_jpeg: Optional[bytes] = None
        self.latest_status: dict[str, Any] = {
            "running": False,
            "state": "STOPPED",
            "label": "STOPPED",
            "evidence": 0.0,
            "reasons": [],
            "metrics": {},
            "debug": {},
            "fps": 0.0,
            "frame_count": 0,
            "source": self.config.input.source,
            "video_path": self.config.input.video_path,
            "engine": self.config.decision_engine,
        }

    def start(self, source: Optional[str] = None, video_path: Optional[str] = None, engine_name: Optional[str] = None) -> None:
        with self.lock:
            if self.running:
                return

            if source:
                self.config.input.source = source
            if video_path:
                self.config.input.video_path = video_path
            if engine_name:
                self.config.decision_engine = engine_name

            self.stop_event.clear()
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False
            self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._close_resources()
        with self.lock:
            self.latest_status["running"] = False
            self.latest_status["state"] = "STOPPED"
            self.latest_status["label"] = "STOPPED"

    def reset(self) -> None:
        with self.lock:
            if self.engine:
                self.engine.reset()
            self.features = SignalFeaturePipeline(self.config)

    def _init_resources(self) -> None:
        self.transport = VideoTransport(
            source=self.config.input.source,
            video_path=self.config.input.video_path,
            loop_file=self.config.input.loop_file,
        )
        self.perception = PerceptionExtractor()
        self.features = SignalFeaturePipeline(self.config)
        self.engine = create_engine(self.config.decision_engine, self.config)
        self.engine.initialize(EngineContext(fps=self.config.runtime.fps, metadata={"engine": self.config.decision_engine}))
        self.alerts = AudioAlertController(
            sound_file="alert.wav",
            drowsy_cooldown_seconds=self.config.alerts.drowsy_cooldown_seconds,
        )

    def _close_resources(self) -> None:
        try:
            if self.alerts:
                self.alerts.close()
        except Exception:
            pass
        try:
            if self.perception:
                self.perception.close()
        except Exception:
            pass
        try:
            if self.transport:
                self.transport.close()
        except Exception:
            pass
        self.transport = None
        self.perception = None
        self.features = None
        self.engine = None
        self.alerts = None

    def _calibration_result(self, debug: dict[str, Any]) -> DecisionResult:
        remaining_frames = max(0, self.config.runtime.calibration_frames - int(debug.get("calibration_count", 0)))
        remaining_seconds = remaining_frames / max(self.config.runtime.fps, 1.0)
        return DecisionResult(
            state=DrowsinessState.ALERT,
            evidence=0.0,
            reasons=["CALIBRATING"],
            alert_sound="none",
            color=(0, 255, 255),
            label=f"CALIBRATING {remaining_seconds:.1f}s",
            debug={},
        )

    def _loop(self) -> None:
        frame_count = 0
        fps_ema = 0.0
        last_time = time.time()
        try:
            self._init_resources()
            assert self.transport and self.perception and self.features and self.engine and self.alerts

            target_interval = 0.0
            if self.config.input.source == "file":
                # Play uploaded/file video near its real FPS instead of racing
                # through frames as fast as the CPU can process them.
                target_fps = getattr(self.transport, "fps", 0.0) or self.config.runtime.fps
                if target_fps and target_fps > 1e-3:
                    target_interval = 1.0 / float(target_fps)

            while not self.stop_event.is_set():
                loop_started = time.time()
                ok, frame = self.transport.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                now = time.time()
                raw = self.perception.process(frame)
                signals, debug = self.features.update(raw, now)
                debug["calibration_count"] = self.features.state.calibration_count

                if not self.features.state.calibrated:
                    result = DecisionResult(
                        state=DrowsinessState.ALERT,
                        evidence=0.0,
                        reasons=["CALIBRATING"],
                        alert_sound="none",
                        color=(0, 255, 255),
                        label=f"CALIBRATING {self.features.state.calibration_count}/{self.config.runtime.calibration_frames}",
                        debug={},
                    )
                else:
                    result = self.engine.update(signals)
                    self.alerts.update(result.alert_sound)

                dt = max(now - last_time, 1e-6)
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst_fps
                last_time = now
                frame_count += 1

                overlay = self._draw_overlay(frame, raw, signals, result, debug, fps_ema)
                ok_jpg, jpg = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                jpeg_bytes = jpg.tobytes() if ok_jpg else None

                status = {
                    "running": True,
                    "state": result.state.value,
                    "label": result.label,
                    "evidence": round(float(result.evidence), 4),
                    "reasons": result.reasons,
                    "alert_sound": result.alert_sound,
                    "metrics": {
                        "ear": round(float(signals.ear), 4),
                        "mar": round(float(signals.mar), 4),
                        "pitch": round(float(signals.pitch), 2),
                        "pitch_velocity": round(float(signals.pitch_velocity), 2),
                        "perclos": round(float(signals.perclos), 4),
                        "perclos_short": round(float(signals.perclos_short), 4),
                        "blink_frequency": int(signals.blink_frequency),
                        "yawn_frequency": int(signals.yawn_frequency),
                        "eyes_closed_consecutive": int(signals.eyes_closed_consecutive),
                        "head_nod_detected": bool(signals.head_nod_detected),
                        "ear_threshold": round(float(debug.get("ear_threshold", 0.0)), 4),
                        "ear_baseline": round(float(debug.get("ear_baseline", 0.0)), 4),
                        "mar_threshold": round(float(debug.get("mar_threshold", 0.0)), 4),
                        "mar_raw": round(float(debug.get("mar_raw", 0.0)), 4),
                        "face_detected": bool(debug.get("face_detected", False)),
                        "calibrated": bool(debug.get("calibrated", False)),
                    },
                    "debug": _json_safe(debug),
                    "fps": round(float(fps_ema), 2),
                    "frame_count": frame_count,
                    "source": self.config.input.source,
                    "video_path": self.config.input.video_path,
                    "engine": self.config.decision_engine,
                    "available_engines": available_engines(),
                }

                with self.lock:
                    self.latest_jpeg = jpeg_bytes
                    self.latest_status = status

                if target_interval > 0:
                    elapsed = time.time() - loop_started
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)

        except Exception as exc:
            with self.lock:
                self.latest_status.update({"running": False, "state": "ERROR", "label": "ERROR", "error": str(exc)})
            self.running = False
        finally:
            self._close_resources()

    def _draw_overlay(self, frame, raw, signals, result: DecisionResult, debug: dict[str, Any], fps: float):
        return _draw_detection_only(frame, raw)

    def get_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest_status)


def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        return str(obj)




class BrowserCameraSession:
    """Per-user runtime for phone/laptop browser camera.

    Each session owns its own MediaPipe extractor, calibration buffers, dynamic thresholds,
    FSM state, PERCLOS windows, blink/yawn counters, and evidence score.
    This prevents users from sharing calibration/state when many people open the same link.
    """

    def __init__(self, session_id: str, config: RuntimeConfig, engine_name: str):
        self.session_id = session_id
        self.config = copy.deepcopy(config)
        self.config.decision_engine = engine_name or self.config.decision_engine
        self.perception = PerceptionExtractor()
        self.features = SignalFeaturePipeline(self.config)
        self.engine = create_engine(self.config.decision_engine, self.config)
        self.engine.initialize(EngineContext(fps=self.config.runtime.fps, metadata={"engine": self.config.decision_engine, "source": "browser_camera"}))
        self.lock = threading.RLock()
        self.frame_count = 0
        self.fps_ema = 0.0
        self.last_time: Optional[float] = None
        self.last_seen = time.time()

    def reset(self) -> None:
        with self.lock:
            self.features = SignalFeaturePipeline(self.config)
            self.engine.reset()
            self.frame_count = 0
            self.fps_ema = 0.0
            self.last_time = None

    def close(self) -> None:
        try:
            self.perception.close()
        except Exception:
            pass

    def process(self, frame):
        with self.lock:
            now = time.time()
            self.last_seen = now
            raw = self.perception.process(frame)
            signals, debug = self.features.update(raw, now)
            debug["calibration_count"] = self.features.state.calibration_count

            if not self.features.state.calibrated:
                result = DecisionResult(
                    state=DrowsinessState.ALERT,
                    evidence=0.0,
                    reasons=["CALIBRATING"],
                    alert_sound="none",
                    color=(0, 255, 255),
                    label=f"CALIBRATING {self.features.state.calibration_count}/{self.config.runtime.calibration_frames}",
                    debug={},
                )
            else:
                result = self.engine.update(signals)

            if self.last_time is not None:
                dt = max(now - self.last_time, 1e-6)
                inst_fps = 1.0 / dt
                self.fps_ema = inst_fps if self.fps_ema == 0.0 else 0.9 * self.fps_ema + 0.1 * inst_fps
            self.last_time = now
            self.frame_count += 1

            overlay = _draw_detection_only(frame, raw)
            status = _status_payload(result, signals, debug, self.fps_ema, self.frame_count, self.config.decision_engine)
            return overlay, status


browser_sessions: dict[str, BrowserCameraSession] = {}
browser_sessions_lock = threading.RLock()
BASE_CONFIG: Optional[RuntimeConfig] = None


def _get_browser_session(session_id: str, engine: str = "fsm") -> BrowserCameraSession:
    global BASE_CONFIG
    if not session_id:
        session_id = str(uuid.uuid4())
    if BASE_CONFIG is None:
        BASE_CONFIG = load_runtime_config(None, {"decision_engine": engine})
    with browser_sessions_lock:
        sess = browser_sessions.get(session_id)
        if sess is None or sess.config.decision_engine != engine:
            if sess is not None:
                sess.close()
            sess = BrowserCameraSession(session_id, BASE_CONFIG, engine)
            browser_sessions[session_id] = sess
        # Lightweight cleanup of idle browser-camera sessions.
        now = time.time()
        stale = [sid for sid, s in browser_sessions.items() if (now - s.last_seen) > 300]
        for sid in stale:
            old = browser_sessions.pop(sid, None)
            if old:
                old.close()
        return sess


runtime: Optional[DMSWebRuntime] = None


@app.get("/")
def index():
    # Public deployment opens the browser-camera/mobile realtime interface by default.
    return render_template("mobile.html")


@app.get("/mobile")
def mobile():
    return render_template("mobile.html")


@app.get("/upload")
def upload_page():
    # Kept only for offline testing with recorded videos. Not needed in real deployment.
    return render_template("index.html")


@app.post("/api/mobile/start")
def api_mobile_start():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id") or uuid.uuid4())
    engine = payload.get("engine") or "fsm"
    sess = _get_browser_session(session_id, engine)
    return jsonify({"ok": True, "session_id": sess.session_id, "engine": sess.config.decision_engine})


@app.post("/api/mobile/reset")
def api_mobile_reset():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id") or "")
    engine = payload.get("engine") or "fsm"
    sess = _get_browser_session(session_id, engine)
    sess.reset()
    return jsonify({"ok": True, "session_id": sess.session_id})


@app.post("/api/mobile/stop")
def api_mobile_stop():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id") or "")
    with browser_sessions_lock:
        sess = browser_sessions.pop(session_id, None)
    if sess:
        sess.close()
    return jsonify({"ok": True})


@app.post("/api/mobile/frame")
def api_mobile_frame():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id") or uuid.uuid4())
    engine = payload.get("engine") or "fsm"
    try:
        frame = _decode_data_url_image(str(payload.get("image") or ""))
        sess = _get_browser_session(session_id, engine)
        overlay, status = sess.process(frame)
        return jsonify({
            "ok": True,
            "session_id": sess.session_id,
            "overlay": "data:image/jpeg;base64," + _encode_jpeg_b64(overlay, quality=78),
            "status": status,
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "session_id": session_id}), 400


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            if runtime is None:
                time.sleep(0.1)
                continue
            jpg = runtime.get_jpeg()
            if jpg is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.01)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    if runtime is None:
        return jsonify({"running": False, "state": "NOT_INITIALIZED"})
    return jsonify(runtime.get_status())



@app.post("/api/upload")
def api_upload():
    """Upload a video from the browser and return the saved server-side path."""
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "Không thấy file video trong request."}), 400
    f = request.files["video"]
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "Chưa chọn file video."}), 400

    original = secure_filename(f.filename)
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"ok": False, "error": f"Định dạng {ext} chưa hỗ trợ. Hãy dùng avi/mp4/mov/mkv/webm."}), 400

    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = UPLOAD_DIR / f"{stamp}_{original}"
    f.save(str(save_path))
    return jsonify({"ok": True, "video_path": str(save_path), "filename": original})

@app.post("/api/start")
def api_start():
    if runtime is None:
        return jsonify({"ok": False, "error": "runtime not initialized"}), 500
    payload = request.get_json(silent=True) or {}
    source = payload.get("source")
    video_path = payload.get("video_path")
    engine = payload.get("engine")
    # Nếu đang chạy mà người dùng đổi source/video/engine thì dừng rồi khởi động lại.
    current = runtime.get_status()
    if current.get("running"):
        runtime.stop()
    runtime.start(source=source, video_path=video_path, engine_name=engine)
    return jsonify({"ok": True, "source": source, "video_path": video_path, "engine": engine})


@app.post("/api/stop")
def api_stop():
    if runtime:
        runtime.stop()
    return jsonify({"ok": True})


@app.post("/api/reset")
def api_reset():
    if runtime:
        runtime.reset()
    return jsonify({"ok": True})


@app.get("/api/config")
def api_config():
    if runtime is None:
        return jsonify({})
    return jsonify(config_to_dict(runtime.config))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Web UI for Driver Drowsiness Monitoring System")
    parser.add_argument("--config", default=None, help="Path to JSON runtime config")
    parser.add_argument("--host", default="0.0.0.0", help="Use 0.0.0.0 so other devices on the same Wi-Fi can access the web UI.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source", choices=["webcam", "file"], default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--decision-engine", choices=available_engines(), default=None)
    parser.add_argument("--auto-start", action="store_true")
    return parser


def main() -> int:
    global runtime
    args = build_arg_parser().parse_args()
    overrides = {}
    if args.source:
        overrides["source"] = args.source
    if args.video_path:
        overrides["video_path"] = args.video_path
    if args.decision_engine:
        overrides["decision_engine"] = args.decision_engine
    config = load_runtime_config(args.config, overrides)
    config.display_window = False
    global BASE_CONFIG
    BASE_CONFIG = copy.deepcopy(config)
    runtime = DMSWebRuntime(config)
    if args.auto_start:
        runtime.start()
    
    try:
        lan_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        lan_ip = 'YOUR_PC_IP'
    print(f'[WEB] Local:   http://127.0.0.1:{args.port}')
    print(f'[WEB] Network: http://{lan_ip}:{args.port}  (phone/other PC on same Wi-Fi)')
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
