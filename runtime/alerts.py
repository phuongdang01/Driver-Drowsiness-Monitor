from __future__ import annotations

from pathlib import Path
import threading
import time


try:
    from playsound import playsound
except Exception:
    playsound = None


class AudioAlertController:
    def __init__(self, sound_file: str, drowsy_cooldown_seconds: float = 5.0):
        self.sound_file = sound_file
        self.drowsy_cooldown_seconds = drowsy_cooldown_seconds
        self._stop_event = threading.Event()
        self._continuous_thread: threading.Thread | None = None
        self._last_double_alert = 0.0

    def update(self, sound_type: str) -> None:
        if sound_type == "continuous":
            self._start_continuous()
            return

        if self._continuous_thread and self._continuous_thread.is_alive():
            self._stop_event.set()

        if sound_type == "double":
            now = time.time()
            if now - self._last_double_alert >= self.drowsy_cooldown_seconds:
                self._last_double_alert = now
                threading.Thread(target=self._play_double, daemon=True).start()

    def close(self) -> None:
        self._stop_event.set()

    def _start_continuous(self) -> None:
        if self._continuous_thread and self._continuous_thread.is_alive():
            return
        self._stop_event.clear()
        self._continuous_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._continuous_thread.start()

    def _play_loop(self) -> None:
        while not self._stop_event.is_set():
            self._play_once()
            time.sleep(0.1)

    def _play_double(self) -> None:
        self._play_once()
        time.sleep(0.2)
        self._play_once()

    def _play_once(self) -> None:
        if playsound is None:
            return
        if not Path(self.sound_file).exists():
            return
        try:
            playsound(self.sound_file)
        except Exception:
            return
