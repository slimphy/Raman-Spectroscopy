"""Thread-safe Hamamatsu DCAM camera controller.

Drop-in replacement for the original ``camera_controller.py``.
The public methods used by main.py are preserved, while capture/buffer/property
calls are serialized so that a GUI timer and a scan worker cannot enter DCAM at
the same time.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np

try:
    from dcam import *  # noqa: F403,F401 - vendor wrapper exposes DCAM_IDPROP/Dcam/Dcamapi
    from dcamcon import *  # noqa: F403,F401

    DCAM_AVAILABLE = True
except ImportError:
    DCAM_AVAILABLE = False
    logging.exception("dcam.py 또는 dcamcon.py 파일을 찾을 수 없습니다.")

_LOG = logging.getLogger(__name__)


class CameraController:
    """Owns one DCAM device and serializes every native API call.

    ``begin_exclusive_capture`` / ``end_exclusive_capture`` are intended for
    mapping workers. While a worker owns the camera, live-view calls from the
    GUI thread return ``None`` immediately instead of competing for frames.
    """

    def __init__(self, buffer_count: Optional[int] = None) -> None:
        self.is_connected = False
        self.dcam = None

        self._lock = threading.RLock()
        self._api_initialized = False
        self._capturing = False
        self._buffer_allocated = False
        self._buffer_count = max(
            3,
            int(buffer_count or os.getenv("RAMAN_CAMERA_BUFFER_COUNT", "4")),
        )

        self._exclusive_owner: Optional[int] = None
        self._exclusive_depth = 0
        self._exclusive_previous_capture = False
        self._exclusive_trigger_mode = "INTERNAL"

        self._consecutive_grab_failures = 0
        self._last_error: Optional[str] = None

    @property
    def is_capturing(self) -> bool:
        with self._lock:
            return self._capturing

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _set_error(self, message: str, exc: Optional[BaseException] = None) -> None:
        self._last_error = message
        if exc is None:
            _LOG.error(message)
        else:
            _LOG.exception(message, exc_info=exc)

    def initialize_dcam(self) -> bool:
        if not DCAM_AVAILABLE:
            self._set_error("DCAM wrapper를 불러오지 못했습니다.")
            return False

        with self._lock:
            if self._api_initialized:
                return True
            try:
                ok = bool(Dcamapi.init())  # noqa: F405
            except Exception as exc:
                self._set_error("DCAM API 초기화 중 예외가 발생했습니다.", exc)
                return False

            self._api_initialized = ok
            if ok:
                _LOG.info("DCAM API initialized")
            else:
                self._set_error("DCAM API 초기화에 실패했습니다.")
            return ok

    def connect_first_available_camera(self) -> bool:
        if not DCAM_AVAILABLE:
            return False

        with self._lock:
            if self.is_connected and self.dcam is not None:
                return True
            if not self._api_initialized and not self.initialize_dcam():
                return False

            try:
                device = Dcam(0)  # noqa: F405
                if not device.dev_open():
                    self._set_error("첫 번째 Hamamatsu 카메라를 열지 못했습니다.")
                    return False
                self.dcam = device
                self.is_connected = True
                self._capturing = False
                self._buffer_allocated = False
                self._last_error = None
                _LOG.info("Hamamatsu qCMOS camera connected")
                return True
            except Exception as exc:
                self.dcam = None
                self.is_connected = False
                self._set_error("카메라 연결 중 예외가 발생했습니다.", exc)
                return False

    def disconnect(self) -> None:
        """Stop capture and close the device under the same lock."""
        with self._lock:
            self._exclusive_owner = None
            self._exclusive_depth = 0
            self._stop_capture_locked(force=True)
            if self.dcam is not None:
                try:
                    self.dcam.dev_close()
                except Exception as exc:
                    self._set_error("카메라 장치 닫기 중 예외가 발생했습니다.", exc)
            self.dcam = None
            self.is_connected = False
            _LOG.info("Camera disconnected")

    def uninitialize_dcam(self) -> None:
        with self._lock:
            if self.is_connected:
                self.disconnect()
            if DCAM_AVAILABLE and self._api_initialized:
                try:
                    Dcamapi.uninit()  # noqa: F405
                except Exception as exc:
                    self._set_error("DCAM API 해제 중 예외가 발생했습니다.", exc)
                finally:
                    self._api_initialized = False

    def _caller_can_control(self, force: bool = False) -> bool:
        owner = self._exclusive_owner
        return force or owner is None or owner == threading.get_ident()

    def start_capture(self, buffer_count: Optional[int] = None, force: bool = False) -> bool:
        with self._lock:
            if not self._caller_can_control(force):
                return False
            return self._start_capture_locked(buffer_count)

    def _start_capture_locked(self, buffer_count: Optional[int] = None) -> bool:
        if not self.is_connected or self.dcam is None:
            return False
        if self._capturing:
            return True

        count = max(3, int(buffer_count or self._buffer_count))
        try:
            if not self._buffer_allocated:
                if not self.dcam.buf_alloc(count):
                    self._set_error(f"DCAM 프레임 버퍼 {count}개 할당에 실패했습니다.")
                    return False
                self._buffer_allocated = True

            if not self.dcam.cap_start():
                self._set_error("DCAM capture 시작에 실패했습니다.")
                try:
                    self.dcam.buf_release()
                finally:
                    self._buffer_allocated = False
                return False

            self._capturing = True
            self._consecutive_grab_failures = 0
            _LOG.info("Live capture started (buffers=%d)", count)
            return True
        except Exception as exc:
            self._capturing = False
            self._set_error("Capture 시작 중 예외가 발생했습니다.", exc)
            return False

    def stop_capture(self, force: bool = False) -> bool:
        with self._lock:
            if not self._caller_can_control(force):
                return False
            return self._stop_capture_locked(force=force)

    def _stop_capture_locked(self, force: bool = False) -> bool:
        if self.dcam is None:
            self._capturing = False
            self._buffer_allocated = False
            return True

        ok = True
        if self._capturing:
            try:
                result = self.dcam.cap_stop()
                if result is False:
                    ok = False
                    self._set_error("DCAM capture 중지 요청이 실패했습니다.")
            except Exception as exc:
                ok = False
                self._set_error("Capture 중지 중 예외가 발생했습니다.", exc)
            finally:
                self._capturing = False

        if self._buffer_allocated:
            try:
                result = self.dcam.buf_release()
                if result is False:
                    ok = False
                    self._set_error("DCAM 버퍼 해제가 실패했습니다.")
            except Exception as exc:
                ok = False
                self._set_error("DCAM 버퍼 해제 중 예외가 발생했습니다.", exc)
            finally:
                self._buffer_allocated = False

        if ok:
            _LOG.info("Live capture stopped")
        return ok

    def grab_frame(
        self,
        timeout_ms: int = 100,
        nonblocking: bool = False,
    ) -> Optional[np.ndarray]:
        """Return one copied frame or ``None``.

        GUI live view should use ``nonblocking=True`` and a short timeout. This
        prevents a slow or disconnected camera from freezing the Qt event loop.
        """
        caller = threading.get_ident()
        if self._exclusive_owner is not None and self._exclusive_owner != caller:
            return None

        acquired = self._lock.acquire(blocking=not nonblocking)
        if not acquired:
            return None
        try:
            if self._exclusive_owner is not None and self._exclusive_owner != caller:
                return None
            if not self.is_connected or self.dcam is None or not self._capturing:
                return None

            timeout_ms = max(1, min(int(timeout_ms), 10_000))
            try:
                ready = bool(self.dcam.wait_capevent_frameready(timeout_ms))
                if not ready:
                    return None
                frame = self.dcam.buf_getlastframedata()
            except Exception as exc:
                self._consecutive_grab_failures += 1
                if self._consecutive_grab_failures in (1, 5, 20):
                    self._set_error(
                        f"프레임 획득 예외가 연속 {self._consecutive_grab_failures}회 발생했습니다.",
                        exc,
                    )
                return None

            if frame is None:
                self._consecutive_grab_failures += 1
                return None

            self._consecutive_grab_failures = 0
            # dcam.py uses dcambuf_copyframe, so this is already independent of
            # the native ring buffer. asarray avoids an unnecessary second copy.
            return np.asarray(frame)
        finally:
            self._lock.release()

    def recover_capture(self) -> bool:
        """Perform a bounded stop/release/start cycle after repeated failures."""
        with self._lock:
            if not self.is_connected or not self._caller_can_control():
                return False
            was_capturing = self._capturing
            self._stop_capture_locked(force=True)
            time.sleep(0.02)
            return (not was_capturing) or self._start_capture_locked()

    def begin_exclusive_capture(self, mode: str = "INTERNAL", restart: bool = False) -> bool:
        """Give the current worker exclusive frame access.

        ``restart=True`` is appropriate when changing to external trigger mode.
        Nested calls by the same thread are supported.
        """
        caller = threading.get_ident()
        mode = str(mode).upper()
        with self._lock:
            if self._exclusive_owner not in (None, caller):
                return False
            if self._exclusive_owner == caller:
                self._exclusive_depth += 1
                return True

            self._exclusive_owner = caller
            self._exclusive_depth = 1
            self._exclusive_previous_capture = self._capturing
            self._exclusive_trigger_mode = mode

            try:
                if restart or mode != "INTERNAL":
                    self._stop_capture_locked(force=True)
                    if not self._set_trigger_mode_locked(mode):
                        raise RuntimeError(f"trigger mode change failed: {mode}")
                    if not self._start_capture_locked():
                        raise RuntimeError("capture restart failed")
                elif not self._capturing:
                    if not self._start_capture_locked():
                        raise RuntimeError("capture start failed")
                _LOG.info("Camera acquired exclusively by thread %s (%s)", caller, mode)
                return True
            except Exception as exc:
                self._exclusive_owner = None
                self._exclusive_depth = 0
                self._set_error("카메라 독점 모드 진입에 실패했습니다.", exc)
                return False

    def end_exclusive_capture(self, resume_live: bool = True) -> bool:
        caller = threading.get_ident()
        with self._lock:
            if self._exclusive_owner != caller:
                return False
            self._exclusive_depth -= 1
            if self._exclusive_depth > 0:
                return True

            previous_capture = self._exclusive_previous_capture
            trigger_mode = self._exclusive_trigger_mode
            ok = True
            try:
                if trigger_mode != "INTERNAL":
                    self._stop_capture_locked(force=True)
                    ok = self._set_trigger_mode_locked("INTERNAL") and ok
                    if resume_live and previous_capture:
                        ok = self._start_capture_locked() and ok
            finally:
                self._exclusive_owner = None
                self._exclusive_depth = 0
                self._exclusive_previous_capture = False
                self._exclusive_trigger_mode = "INTERNAL"
            _LOG.info("Camera exclusive access released by thread %s", caller)
            return ok

    @contextmanager
    def exclusive_capture(self, mode: str = "INTERNAL", restart: bool = False) -> Iterator[bool]:
        acquired = self.begin_exclusive_capture(mode=mode, restart=restart)
        try:
            yield acquired
        finally:
            if acquired:
                self.end_exclusive_capture(resume_live=True)

    def _set_property_locked(self, prop_id, value) -> bool:
        if not self.is_connected or self.dcam is None:
            return False
        try:
            result = self.dcam.prop_setvalue(prop_id, value)
            return result is not False
        except Exception as exc:
            self._set_error(f"DCAM property 설정 실패: {prop_id}={value}", exc)
            return False

    def set_exposure_time(self, exp_time_sec: float) -> bool:
        with self._lock:
            return self._set_property_locked(DCAM_IDPROP.EXPOSURETIME, float(exp_time_sec))  # noqa: F405

    def set_binning(self, bin_size: int) -> bool:
        with self._lock:
            return self._set_property_locked(DCAM_IDPROP.BINNING, int(bin_size))  # noqa: F405

    def set_roi(self, offset_y: int, height: int) -> bool:
        """Apply vertical ROI while capture is safely stopped, then resume."""
        offset_y = int(offset_y)
        height = int(height)
        if offset_y < 0 or height <= 0:
            raise ValueError("ROI offset_y는 0 이상, height는 1 이상이어야 합니다.")

        with self._lock:
            if not self._caller_can_control():
                return False
            was_capturing = self._capturing
            if was_capturing:
                self._stop_capture_locked(force=True)
            try:
                # OFF -> configure -> ON is safer on DCAM cameras than modifying
                # subarray dimensions while acquisition is active.
                ok = self._set_property_locked(DCAM_IDPROP.SUBARRAYMODE, 1)  # noqa: F405
                ok = self._set_property_locked(DCAM_IDPROP.SUBARRAYVSIZE, height) and ok  # noqa: F405
                ok = self._set_property_locked(DCAM_IDPROP.SUBARRAYVPOS, offset_y) and ok  # noqa: F405
                ok = self._set_property_locked(DCAM_IDPROP.SUBARRAYMODE, 2) and ok  # noqa: F405
                return ok
            finally:
                if was_capturing:
                    self._start_capture_locked()

    def set_cooler(self, state: bool) -> bool:
        with self._lock:
            if not self.is_connected:
                return False
            cooler_value = 4.0 if state else 1.0
            ok = self._set_property_locked(DCAM_IDPROP.SENSORCOOLER, cooler_value)  # noqa: F405
            try:
                # Keep the original behavior. Some camera models do not expose
                # this property; that should not fail cooler control itself.
                self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLERFAN, 1.0)  # noqa: F405
            except Exception:
                _LOG.debug("SENSORCOOLERFAN is not supported", exc_info=True)
            return ok

    def get_temperature(self):
        with self._lock:
            if not self.is_connected or self.dcam is None:
                return None
            try:
                return self.dcam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE)  # noqa: F405
            except Exception as exc:
                self._set_error("센서 온도 읽기에 실패했습니다.", exc)
                return None

    def set_trigger_mode(self, mode: str = "INTERNAL") -> bool:
        with self._lock:
            if not self._caller_can_control():
                return False
            return self._set_trigger_mode_locked(mode)

    def _set_trigger_mode_locked(self, mode: str = "INTERNAL") -> bool:
        if not self.is_connected or self.dcam is None:
            return False
        mode = str(mode).upper()
        try:
            if mode == "EXTERNAL":
                if not self._set_property_locked(DCAM_IDPROP.TRIGGERSOURCE, 2.0):  # noqa: F405
                    return False
                try:
                    self.dcam.prop_setvalue(DCAM_IDPROP.TRIGGERPOLARITY, 1.0)  # noqa: F405
                except Exception:
                    _LOG.debug("TRIGGERPOLARITY is not supported", exc_info=True)
                try:
                    self.dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE, 1.0)  # noqa: F405
                except Exception:
                    _LOG.debug("TRIGGERACTIVE is not supported", exc_info=True)

                time.sleep(float(os.getenv("RAMAN_TRIGGER_SETTLE_SEC", "0.10")))
                actual = self.dcam.prop_getvalue(DCAM_IDPROP.TRIGGERSOURCE)  # noqa: F405
                if actual is not None and abs(float(actual) - 2.0) > 1e-6:
                    self._set_error(f"외부 트리거 설정 검증 실패: actual={actual}")
                    return False
                _LOG.info("External hardware trigger enabled")
                return True

            if mode != "INTERNAL":
                raise ValueError(f"지원하지 않는 trigger mode: {mode}")
            ok = self._set_property_locked(DCAM_IDPROP.TRIGGERSOURCE, 1.0)  # noqa: F405
            if ok:
                _LOG.info("Internal trigger enabled")
            return ok
        except Exception as exc:
            self._set_error(f"트리거 모드 설정 중 오류: {mode}", exc)
            return False
