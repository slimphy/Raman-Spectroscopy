"""Hardware backends and data helpers for the standalone vision test GUI.

This module deliberately has no side effects on import: IC4 is initialized by the
GUI entry point, and the CSN210 DLL is loaded only when the user presses Connect.
"""

from __future__ import annotations

import ctypes
import contextlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

try:
    import imagingcontrol4 as ic4
except (ImportError, OSError):  # The GUI still works in simulation mode.
    ic4 = None


CALIBRATION_PROFILES = ("20X", "100X")
OBJECTIVE_TO_CSN_POSITION = {"20X": 1, "100X": 2}
CSN_POSITION_TO_OBJECTIVE = {position: objective for objective, position in OBJECTIVE_TO_CSN_POSITION.items()}

# The requested X value was written as "+68m" alongside Y/Z values in um.
# It is treated as +68 um; the reverse transition uses the inverse shift.
OBJECTIVE_STAGE_COMPENSATION_UM = {
    ("100X", "20X"): (68.0, 31.0, -72.0),
    ("20X", "100X"): (-68.0, -31.0, 72.0),
}


def objective_stage_delta_um(source: str, target: str) -> tuple[float, float, float]:
    """Return the piezo compensation for an objective transition."""
    if source == target:
        return 0.0, 0.0, 0.0
    try:
        return OBJECTIVE_STAGE_COMPENSATION_UM[(source, target)]
    except KeyError as exc:
        raise ValueError(f"Unsupported objective transition: {source} -> {target}") from exc


def roi_to_mapping_ranges(
    roi: tuple[float, float, float, float],
    source_size: tuple[int, int],
    um_per_pixel: float,
) -> dict[str, float]:
    """Convert an ROI to stage coordinates: image right is +X and image up is +Y."""
    x0, y0, x1, y1 = roi
    width_px, height_px = source_size
    if width_px <= 0 or height_px <= 0:
        raise ValueError("A camera frame is required before converting the ROI.")
    if not math.isfinite(um_per_pixel) or um_per_pixel <= 0:
        raise ValueError("A valid calibration scale is required.")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI width and height must be greater than zero.")

    center_x_px = width_px / 2.0
    center_y_px = height_px / 2.0
    # Camera pixels increase downward, while this piezo stage's +Y direction
    # appears upward in the image. Keep Start <= End for the Mapping worker.
    top_y_um = (center_y_px - y0) * um_per_pixel
    bottom_y_um = (center_y_px - y1) * um_per_pixel
    return {
        "x_start": (x0 - center_x_px) * um_per_pixel,
        "x_end": (x1 - center_x_px) * um_per_pixel,
        "y_start": bottom_y_um,
        "y_end": top_y_um,
        "center_x_um": (((x0 + x1) / 2.0) - center_x_px) * um_per_pixel,
        "center_y_um": (center_y_px - ((y0 + y1) / 2.0)) * um_per_pixel,
        "width_um": (x1 - x0) * um_per_pixel,
        "height_um": (y1 - y0) * um_per_pixel,
    }


def round_mapping_bounds(
    ranges: dict[str, float],
    decimals: int = 1,
) -> dict[str, float]:
    """Round Mapping X/Y bounds using conventional half-up rounding."""
    if decimals < 0:
        raise ValueError("decimals must be zero or greater")
    quantum = Decimal(1).scaleb(-decimals)
    rounded: dict[str, float] = {}
    for key in ("x_start", "x_end", "y_start", "y_end"):
        value = float(ranges[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        result = float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))
        rounded[key] = 0.0 if result == 0.0 else result
    return rounded


def translate_mapping_xy_bounds(
    ranges: dict[str, float],
    dx_um: float,
    dy_um: float,
    decimals: int = 1,
) -> dict[str, float]:
    """Translate Mapping X/Y bounds and round the final absolute coordinates."""
    translated = {
        "x_start": float(ranges["x_start"]) + float(dx_um),
        "x_end": float(ranges["x_end"]) + float(dx_um),
        "y_start": float(ranges["y_start"]) + float(dy_um),
        "y_end": float(ranges["y_end"]) + float(dy_um),
    }
    return round_mapping_bounds(translated, decimals=decimals)


def configure_ic4_auto_brightness(property_map, enabled: bool) -> None:
    """Enable or freeze the camera's automatic exposure and gain controls."""
    if ic4 is None:
        raise RuntimeError("IC4 Python SDK is not installed.")

    mode = "Continuous" if enabled else "Off"
    failures: list[str] = []
    for property_id, name in (
        (ic4.PropId.EXPOSURE_AUTO, "ExposureAuto"),
        (ic4.PropId.GAIN_AUTO, "GainAuto"),
    ):
        try:
            property_map.set_value(property_id, mode)
        except Exception as exc:
            failures.append(f"{name}: {_exception_text(exc)}")
    if failures:
        raise RuntimeError("; ".join(failures))


@dataclass(frozen=True)
class CalibrationSample:
    known_length_um: float
    pixel_length: float
    um_per_pixel: float

    def as_dict(self) -> dict[str, float]:
        return {
            "known_length_um": self.known_length_um,
            "pixel_length": self.pixel_length,
            "um_per_pixel": self.um_per_pixel,
        }


class CalibrationStore:
    """Load, average, and atomically save 20X/100X calibration profiles."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.profiles: dict[str, list[CalibrationSample]] = {
            profile: [] for profile in CALIBRATION_PROFILES
        }
        self.updated_at: dict[str, str | None] = {
            profile: None for profile in CALIBRATION_PROFILES
        }
        self.load_error: str | None = None
        self.load()

    def load(self) -> None:
        self.load_error = None
        if not self.path.exists():
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            stored_profiles = document.get("profiles", {})
            loaded: dict[str, list[CalibrationSample]] = {}
            for profile in CALIBRATION_PROFILES:
                profile_data = stored_profiles.get(profile, {})
                samples: list[CalibrationSample] = []
                for item in profile_data.get("samples", []):
                    known = float(item["known_length_um"])
                    pixels = float(item["pixel_length"])
                    scale = float(item.get("um_per_pixel", known / pixels))
                    if known > 0 and pixels > 0 and scale > 0:
                        samples.append(CalibrationSample(known, pixels, scale))
                loaded[profile] = samples
                self.updated_at[profile] = profile_data.get("updated_at")
            self.profiles.update(loaded)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.load_error = str(exc)

    def add_sample(self, profile: str, known_length_um: float,
                   pixel_length: float) -> CalibrationSample:
        self._check_profile(profile)
        if not math.isfinite(known_length_um) or known_length_um <= 0:
            raise ValueError("Known length must be greater than zero.")
        if not math.isfinite(pixel_length) or pixel_length <= 0:
            raise ValueError("Pixel length must be greater than zero.")
        sample = CalibrationSample(
            float(known_length_um),
            float(pixel_length),
            float(known_length_um / pixel_length),
        )
        self.profiles[profile].append(sample)
        return sample

    def remove_last(self, profile: str) -> CalibrationSample | None:
        self._check_profile(profile)
        if not self.profiles[profile]:
            return None
        return self.profiles[profile].pop()

    def clear(self, profile: str) -> None:
        self._check_profile(profile)
        self.profiles[profile].clear()

    def average_scale(self, profile: str) -> float | None:
        self._check_profile(profile)
        samples = self.profiles[profile]
        if not samples:
            return None
        return float(np.mean([sample.um_per_pixel for sample in samples]))

    def save(self) -> None:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "unit": "um",
            "profiles": {},
        }
        for profile in CALIBRATION_PROFILES:
            samples = self.profiles[profile]
            if samples:
                self.updated_at[profile] = now
            payload["profiles"][profile] = {
                "um_per_pixel": self.average_scale(profile),
                "sample_count": len(samples),
                "samples": [sample.as_dict() for sample in samples],
                "updated_at": self.updated_at[profile],
            }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(self.path.name + ".tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    @staticmethod
    def _check_profile(profile: str) -> None:
        if profile not in CALIBRATION_PROFILES:
            raise KeyError(f"Unknown calibration profile: {profile}")


class IC4Runtime:
    """Own the single process-wide IC4 library initialization."""

    def __init__(self):
        self.initialized = False
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return ic4 is not None and self.initialized

    def initialize(self) -> bool:
        if ic4 is None:
            self.error = "imagingcontrol4 is not installed."
            return False
        if self.initialized:
            return True
        try:
            ic4.Library.init()
            self.initialized = True
            return True
        except Exception as exc:  # IC4Exception is unavailable when import failed.
            self.error = _exception_text(exc)
            return False

    def enumerate_devices(self) -> list[dict[str, str]]:
        if not self.available:
            return []
        try:
            return [
                {
                    "model": str(device.model_name),
                    "serial": str(device.serial),
                    "version": str(device.version),
                }
                for device in ic4.DeviceEnum.devices()
            ]
        except Exception as exc:
            self.error = _exception_text(exc)
            return []

    def close(self) -> None:
        if not self.initialized or ic4 is None:
            return
        try:
            ic4.Library.exit()
        finally:
            self.initialized = False


class CameraWorker(QThread):
    """Acquire camera frames without blocking Qt's GUI thread."""

    frame_ready = pyqtSignal(object)
    connection_changed = pyqtSignal(bool, str)
    acquisition_error = pyqtSignal(str)
    configuration_changed = pyqtSignal(bool, str)

    def __init__(self, mode: str, serial: str = "", auto_brightness: bool = True,
                 parent=None):
        super().__init__(parent)
        self.mode = mode
        self.serial = serial
        self._stop_event = threading.Event()
        self._settings_lock = threading.Lock()
        self._auto_brightness = bool(auto_brightness)
        self._auto_brightness_dirty = True

    def request_stop(self) -> None:
        self._stop_event.set()

    def set_auto_brightness(self, enabled: bool) -> None:
        with self._settings_lock:
            enabled = bool(enabled)
            if enabled != self._auto_brightness:
                self._auto_brightness = enabled
                self._auto_brightness_dirty = True

    def run(self) -> None:
        self._stop_event.clear()
        if self.mode == "simulation":
            self._run_simulation()
        else:
            self._run_ic4()

    def _run_ic4(self) -> None:
        if ic4 is None:
            self.acquisition_error.emit("IC4 Python SDK is not installed.")
            self.connection_changed.emit(False, "Disconnected")
            return

        grabber = None
        try:
            matching = [
                device for device in ic4.DeviceEnum.devices()
                if str(device.serial) == self.serial
            ]
            if not matching:
                raise RuntimeError(f"Camera serial {self.serial!r} was not found.")

            device = matching[0]
            grabber = ic4.Grabber()
            grabber.device_open(device)
            self._apply_auto_brightness(grabber, force=True)
            sink = ic4.SnapSink(accepted_pixel_formats=[ic4.PixelFormat.BGR8])
            grabber.stream_setup(
                sink,
                setup_option=ic4.StreamSetupOption.ACQUISITION_START,
            )
            self.connection_changed.emit(
                True, f"{device.model_name} / S/N {device.serial}"
            )

            last_emit = 0.0
            while not self._stop_event.is_set():
                self._apply_auto_brightness(grabber)
                try:
                    image_buffer = sink.snap_single(500)
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    # A transient timeout is not fatal; all other errors are surfaced.
                    message = _exception_text(exc)
                    if "timeout" in message.lower() or "timed out" in message.lower():
                        continue
                    raise

                try:
                    now = time.monotonic()
                    if now - last_emit < 1.0 / 30.0:
                        continue
                    bgr = image_buffer.numpy_copy()
                    if bgr.ndim != 3 or bgr.shape[2] < 3:
                        raise RuntimeError(f"Unexpected IC4 frame shape: {bgr.shape}")
                    rgb = np.ascontiguousarray(bgr[:, :, :3][:, :, ::-1])
                    self.frame_ready.emit(rgb)
                    last_emit = now
                finally:
                    release = getattr(image_buffer, "release", None)
                    if callable(release):
                        release()
        except Exception as exc:
            self.acquisition_error.emit(_exception_text(exc))
        finally:
            if grabber is not None:
                try:
                    if grabber.is_streaming:
                        grabber.stream_stop()
                except Exception:
                    pass
                try:
                    if grabber.is_device_open:
                        grabber.device_close()
                except Exception:
                    pass
            self.connection_changed.emit(False, "Disconnected")

    def _run_simulation(self) -> None:
        width, height = 1280, 720
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        self.connection_changed.emit(True, "Simulation / 1280 x 720")
        self._report_simulation_brightness(force=True)
        phase = 0.0
        try:
            while not self._stop_event.is_set():
                self._report_simulation_brightness()
                red = np.broadcast_to(35 + 100 * x, (height, width))
                green = np.broadcast_to(30 + 95 * y, (height, width))
                blue = 38 + 65 * (1.0 - x) + 45 * y
                frame = np.stack((red, green, blue), axis=2).astype(np.uint8)

                frame[::100, :, :] = 88
                frame[:, ::100, :] = 88

                # High-contrast H target for exercising line calibration.
                cx, cy = width // 2, height // 2
                frame[cy - 130:cy + 131, cx - 155:cx - 145] = (235, 235, 235)
                frame[cy - 130:cy + 131, cx + 145:cx + 155] = (235, 235, 235)
                frame[cy - 5:cy + 6, cx - 150:cx + 151] = (235, 235, 235)

                dot_x = int(cx + 250 * math.sin(phase))
                dot_y = int(cy + 150 * math.cos(phase * 0.7))
                yy, xx = np.ogrid[:height, :width]
                mask = (xx - dot_x) ** 2 + (yy - dot_y) ** 2 <= 12 ** 2
                frame[mask] = (255, 80, 70)

                self.frame_ready.emit(np.ascontiguousarray(frame))
                phase += 0.06
                self.msleep(33)
        finally:
            self.connection_changed.emit(False, "Disconnected")

    def _apply_auto_brightness(self, grabber, force: bool = False) -> None:
        enabled = self._take_auto_brightness_request(force)
        if enabled is None:
            return
        try:
            configure_ic4_auto_brightness(grabber.device_property_map, enabled)
        except RuntimeError as exc:
            self.configuration_changed.emit(False, f"Auto brightness failed: {exc}")
            return
        state = "ON" if enabled else "OFF (current exposure/gain held)"
        self.configuration_changed.emit(
            True,
            f"Auto brightness {state}: ExposureAuto + GainAuto",
        )

    def _report_simulation_brightness(self, force: bool = False) -> None:
        enabled = self._take_auto_brightness_request(force)
        if enabled is None:
            return
        state = "ON" if enabled else "OFF"
        self.configuration_changed.emit(
            True,
            f"Auto brightness {state} (simulation: camera controls are not applied)",
        )

    def _take_auto_brightness_request(self, force: bool = False) -> bool | None:
        with self._settings_lock:
            if not force and not self._auto_brightness_dirty:
                return None
            enabled = self._auto_brightness
            self._auto_brightness_dirty = False
            return enabled


@dataclass(frozen=True)
class CSN210Snapshot:
    position_code: int
    position_text: str
    homed: bool
    collision: bool


class CSN210Controller:
    """Thin ctypes wrapper around Thorlabs' official CSN210 SDK DLL."""

    PARAM_TURRET_POS = 502
    PARAM_TURRET_SERIALNUMBER = 505
    PARAM_TURRET_FIRMWAREVERSION = 507
    PARAM_TURRET_STOP = 508
    PARAM_TURRET_HOMED = 509
    PARAM_TURRET_COLLISION = 510
    PARAM_TURRET_POS_CURRENT = 726

    POSITION_TEXT = {
        0: "Between positions / unknown",
        1: "Position 1",
        2: "Position 2",
        3: "Homing",
        4: "Moving to position 1",
        5: "Moving to position 2",
        6: "Disconnected",
    }
    _cwd_lock = threading.RLock()

    def __init__(self, dll_path: str | Path | None = None):
        self.requested_dll_path = Path(dll_path) if dll_path else None
        self.dll_path: Path | None = None
        self._dll = None
        self._dll_directory_handle = None
        self.connected = False
        self.device_count = 0
        self.serial_number = ""
        self.firmware_version = ""

    @staticmethod
    def default_dll_candidates() -> list[Path]:
        candidates: list[Path] = []
        configured = os.environ.get("CSN210_DLL")
        if configured:
            candidates.append(Path(configured))
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates.append(
            Path(program_files)
            / "Thorlabs"
            / "CSN210 4.0"
            / "bin"
            / "x64"
            / "ThorObjectiveChanger.dll"
        )
        candidates.append(
            Path(program_files)
            / "Thorlabs"
            / "CSN210 4.0"
            / "Application"
            / "ThorObjectiveChanger.dll"
        )
        return candidates

    def connect(self, device_index: int = 0) -> int:
        if self.connected:
            return self.device_count
        self._load_dll()
        with self._sdk_working_directory():
            count = ctypes.c_long(0)
            self._ensure(self._dll.FindDevices(ctypes.byref(count)), "FindDevices")
            self.device_count = int(count.value)
            if self.device_count <= 0:
                raise RuntimeError("No CSN210 device was found.")
            if not 0 <= device_index < self.device_count:
                raise IndexError(f"CSN210 device index {device_index} is out of range.")
            self._ensure(self._dll.SelectDevice(device_index), "SelectDevice")
            self.connected = True
            self._ensure(self._dll.PreflightPosition(), "PreflightPosition")
            try:
                self.firmware_version = self._get_string(self.PARAM_TURRET_FIRMWAREVERSION)
            except RuntimeError:
                self.firmware_version = ""
            self._ensure(self._dll.PreflightPosition(), "PreflightPosition")
            try:
                self.serial_number = self._get_string(self.PARAM_TURRET_SERIALNUMBER)
            except RuntimeError:
                self.serial_number = ""
        return self.device_count

    def disconnect(self) -> None:
        if self._dll is not None and self.connected:
            try:
                with self._sdk_working_directory():
                    self._dll.TeardownDevice()
            finally:
                self.connected = False

    def home(self) -> None:
        self._move(0)

    def move_to(self, position: int) -> None:
        if position not in (1, 2):
            raise ValueError("CSN210 position must be 1 or 2.")
        self._move(position)

    def stop(self) -> None:
        self._require_connected()
        with self._sdk_working_directory():
            self._ensure(
                self._dll.SetParam(self.PARAM_TURRET_STOP, 1.0),
                "SetParam(PARAM_TURRET_STOP)",
            )

    def snapshot(self) -> CSN210Snapshot:
        self._require_connected()
        with self._sdk_working_directory():
            position = int(round(self._get_param(self.PARAM_TURRET_POS_CURRENT)))
            homed = bool(round(self._get_param(self.PARAM_TURRET_HOMED)))
            collision = bool(round(self._get_param(self.PARAM_TURRET_COLLISION)))
        return CSN210Snapshot(
            position_code=position,
            position_text=self.POSITION_TEXT.get(position, f"Unknown ({position})"),
            homed=homed,
            collision=collision,
        )

    def _move(self, position: int) -> None:
        self._require_connected()
        with self._sdk_working_directory():
            self._ensure(self._dll.PreflightPosition(), "PreflightPosition")
            try:
                self._ensure(
                    self._dll.SetParam(self.PARAM_TURRET_POS, float(position)),
                    f"SetParam(position={position})",
                )
                self._ensure(self._dll.SetupPosition(), "SetupPosition")
                self._ensure(self._dll.StartPosition(), "StartPosition")
            finally:
                self._dll.PostflightPosition()

    def _load_dll(self) -> None:
        if self._dll is not None:
            return
        if sys.platform != "win32":
            raise RuntimeError("The CSN210 SDK DLL is supported only on Windows.")

        candidates = (
            [self.requested_dll_path]
            if self.requested_dll_path is not None
            else self.default_dll_candidates()
        )
        selected = next((path for path in candidates if path and path.is_file()), None)
        if selected is None:
            locations = "\n".join(str(path) for path in candidates if path)
            raise FileNotFoundError(
                "ThorObjectiveChanger.dll was not found. Checked:\n" + locations
            )

        if hasattr(os, "add_dll_directory"):
            self._dll_directory_handle = os.add_dll_directory(str(selected.parent))
        self._dll = ctypes.WinDLL(str(selected))
        self.dll_path = selected
        self._define_signatures()

    def _define_signatures(self) -> None:
        c_long_p = ctypes.POINTER(ctypes.c_long)
        c_double_p = ctypes.POINTER(ctypes.c_double)
        c_wchar_p = ctypes.POINTER(ctypes.c_wchar)

        self._dll.FindDevices.argtypes = [c_long_p]
        self._dll.FindDevices.restype = ctypes.c_long
        self._dll.SelectDevice.argtypes = [ctypes.c_long]
        self._dll.SelectDevice.restype = ctypes.c_long
        self._dll.TeardownDevice.argtypes = []
        self._dll.TeardownDevice.restype = ctypes.c_long
        self._dll.SetParam.argtypes = [ctypes.c_long, ctypes.c_double]
        self._dll.SetParam.restype = ctypes.c_long
        self._dll.GetParam.argtypes = [ctypes.c_long, c_double_p]
        self._dll.GetParam.restype = ctypes.c_long
        self._dll.PreflightPosition.argtypes = []
        self._dll.PreflightPosition.restype = ctypes.c_long
        self._dll.SetupPosition.argtypes = []
        self._dll.SetupPosition.restype = ctypes.c_long
        self._dll.StartPosition.argtypes = []
        self._dll.StartPosition.restype = ctypes.c_long
        self._dll.PostflightPosition.argtypes = []
        self._dll.PostflightPosition.restype = ctypes.c_long
        self._dll.GetParamString.argtypes = [ctypes.c_long, c_wchar_p, ctypes.c_long]
        self._dll.GetParamString.restype = ctypes.c_long
        self._dll.GetLastErrorMsg.argtypes = [c_wchar_p, ctypes.c_long]
        self._dll.GetLastErrorMsg.restype = ctypes.c_long

    def _get_param(self, param_id: int) -> float:
        value = ctypes.c_double(0.0)
        self._ensure(
            self._dll.GetParam(param_id, ctypes.byref(value)),
            f"GetParam({param_id})",
        )
        return float(value.value)

    def _get_string(self, param_id: int) -> str:
        buffer = ctypes.create_unicode_buffer(256)
        self._ensure(
            self._dll.GetParamString(param_id, buffer, len(buffer)),
            f"GetParamString({param_id})",
        )
        return buffer.value.strip()

    def _ensure(self, result: int, operation: str) -> None:
        if result:
            return
        detail = ""
        if self._dll is not None:
            try:
                buffer = ctypes.create_unicode_buffer(512)
                if self._dll.GetLastErrorMsg(buffer, len(buffer)):
                    detail = buffer.value.strip()
            except Exception:
                pass
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"CSN210 {operation} failed{suffix}")

    def _require_connected(self) -> None:
        if not self.connected or self._dll is None:
            raise RuntimeError("CSN210 is not connected.")

    @contextlib.contextmanager
    def _sdk_working_directory(self):
        """Let the legacy SDK locate ThorObjectiveChangerSettings.xml safely."""
        if self.dll_path is None:
            yield
            return
        with self._cwd_lock:
            previous = Path.cwd()
            os.chdir(self.dll_path.parent)
            try:
                yield
            finally:
                os.chdir(previous)


def csn210_vendor_app_running() -> bool:
    """Return True when Thorlabs' GUI is likely holding the CSN210 device."""
    if sys.platform != "win32":
        return False
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq CSN210_Control.exe", "/NH"],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "CSN210_Control.exe" in completed.stdout


class CSN210Worker(QObject):
    """Serialize all DLL access on a daemon thread and report state to Qt."""

    connection_changed = pyqtSignal(bool, object)
    snapshot_ready = pyqtSignal(object)
    command_started = pyqtSignal(str)
    command_finished = pyqtSignal(str, bool, str)

    def __init__(self, dll_path: str | Path | None = None, parent=None):
        super().__init__(parent)
        self._commands: queue.Queue[str] = queue.Queue()
        self._controller = CSN210Controller(dll_path)
        self._shutdown_requested = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="CSN210Worker",
            daemon=True,
        )
        self._thread.start()

    def request(self, command: str) -> None:
        if command not in ("connect", "disconnect", "home", "position1", "position2", "stop"):
            raise ValueError(f"Unknown CSN210 command: {command}")
        self._commands.put(command)

    def shutdown(self) -> None:
        self._shutdown_requested.set()
        self._commands.put("shutdown")

    def _run(self) -> None:
        last_poll = 0.0
        while not self._shutdown_requested.is_set():
            command = None
            try:
                command = self._commands.get(timeout=0.08)
            except queue.Empty:
                pass

            if command == "shutdown":
                break
            if command is not None:
                self._execute(command)

            now = time.monotonic()
            if self._controller.connected and now - last_poll >= 0.2:
                try:
                    self.snapshot_ready.emit(self._controller.snapshot())
                except Exception as exc:
                    self.command_finished.emit("poll", False, _exception_text(exc))
                last_poll = now

        try:
            self._controller.disconnect()
        except Exception:
            pass

    def _execute(self, command: str) -> None:
        self.command_started.emit(command)
        try:
            if command == "connect":
                count = self._controller.connect()
                info = {
                    "count": count,
                    "serial": self._controller.serial_number,
                    "firmware": self._controller.firmware_version,
                    "dll_path": str(self._controller.dll_path or ""),
                }
                self.connection_changed.emit(True, info)
            elif command == "disconnect":
                self._controller.disconnect()
                self.connection_changed.emit(False, {})
            elif command == "home":
                self._controller.home()
            elif command == "position1":
                self._controller.move_to(1)
            elif command == "position2":
                self._controller.move_to(2)
            elif command == "stop":
                self._controller.stop()
        except Exception as exc:
            if command == "connect":
                self.connection_changed.emit(False, {})
            self.command_finished.emit(command, False, _exception_text(exc))
            return
        self.command_finished.emit(command, True, "")


def _exception_text(exc: BaseException) -> str:
    message = getattr(exc, "message", None)
    return str(message or exc or type(exc).__name__)
