"""Non-blocking camera acquisition and latest-frame analysis workers."""

from __future__ import annotations

import threading
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .analyzer import AlignmentConfig, analyze_alignment_frame


class AcquisitionWorker(QThread):
    frame_ready = pyqtSignal(object, float)
    status_changed = pyqtSignal(str)
    acquisition_error = pyqtSignal(str)

    def __init__(self, camera, simulate: bool = False):
        super().__init__()
        self.camera = camera
        self.simulate = bool(simulate)
        self._running = threading.Event()
        self._running.set()
        self._simulation_index = 0

    def stop(self) -> None:
        self._running.clear()

    def _simulation_frame(self) -> np.ndarray:
        height, width = 384, 1024
        y = np.arange(height, dtype=np.float32)
        x = np.arange(width, dtype=np.float32)
        normalized_y = (y - 0.5 * (height - 1)) / (height - 1)
        phase = 0.035 * self._simulation_index
        center = 510.0 + 3.0 * np.sin(0.35 * phase)
        tilt = 4.0 * np.sin(phase)
        curvature = 2.0 * np.cos(0.6 * phase)
        row_centers = center + tilt * normalized_y
        row_centers += curvature * (normalized_y**2 - np.mean(normalized_y**2))
        distance = x[None, :] - row_centers[:, None]
        vertical = 0.55 + 0.45 * np.exp(-0.5 * (normalized_y / 0.36) ** 2)
        signal = 200.0 + 1400.0 * vertical[:, None] * np.exp(-0.5 * (distance / 4.3) ** 2)
        rng = np.random.default_rng(314159 + self._simulation_index)
        signal += rng.normal(0.0, 2.2, signal.shape)
        self._simulation_index += 1
        return np.clip(signal, 0, 65535).astype(np.uint16)

    def run(self) -> None:
        capture_started = False
        try:
            if not self.simulate and self.camera.is_connected:
                self.camera.start_capture()
                capture_started = True
                self.status_changed.emit("ORCA-Quest 2 live capture")
            else:
                self.status_changed.emit("Simulation capture")

            while self._running.is_set():
                if capture_started:
                    frame = self.camera.grab_frame()
                    if frame is None:
                        continue
                    array = np.asarray(frame).copy()
                else:
                    array = self._simulation_frame()
                    time.sleep(0.05)
                self.frame_ready.emit(array, time.time())
        except Exception as exc:  # hardware boundary: report exact driver failure to the UI
            self.acquisition_error.emit(str(exc))
        finally:
            if capture_started:
                try:
                    self.camera.stop_capture()
                except Exception as exc:
                    self.acquisition_error.emit(f"Capture shutdown failed: {exc}")


class AnalysisWorker(QThread):
    result_ready = pyqtSignal(object, object)
    analysis_error = pyqtSignal(str)

    def __init__(self, config: AlignmentConfig):
        super().__init__()
        self._condition = threading.Condition()
        self._running = True
        self._latest: tuple[np.ndarray, float] | None = None
        self._config = config
        self._dark_reference: np.ndarray | None = None

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()

    def submit(self, frame: np.ndarray, timestamp: float) -> None:
        with self._condition:
            self._latest = (frame, float(timestamp))
            self._condition.notify()

    def update_settings(
        self, config: AlignmentConfig, dark_reference: np.ndarray | None
    ) -> None:
        with self._condition:
            self._config = config
            self._dark_reference = dark_reference

    def run(self) -> None:
        while True:
            with self._condition:
                while self._running and self._latest is None:
                    self._condition.wait(timeout=0.25)
                if not self._running:
                    return
                frame, timestamp = self._latest
                self._latest = None
                config = self._config
                dark_reference = self._dark_reference
            try:
                result = analyze_alignment_frame(frame, dark_reference, config, timestamp)
                self.result_ready.emit(frame, result)
            except Exception as exc:
                self.analysis_error.emit(str(exc))


__all__ = ["AcquisitionWorker", "AnalysisWorker"]
