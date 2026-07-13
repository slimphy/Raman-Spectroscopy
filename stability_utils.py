"""Runtime safety helpers used by the stability patch."""
from __future__ import annotations

import ast
import logging
import math
import operator
import os
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

_LOG = logging.getLogger(__name__)
_HOOKS_INSTALLED = False


def install_global_exception_logging(log_dir: str = "logs") -> Path:
    """Install rotating file logs and uncaught-exception hooks once."""
    global _HOOKS_INSTALLED
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "raman_app.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")) == log_path.resolve()
        for handler in root.handlers
    ):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"
            )
        )
        root.addHandler(handler)

    if not _HOOKS_INSTALLED:
        previous_sys_hook = sys.excepthook

        def sys_hook(exc_type, exc_value, exc_traceback):
            if exc_type is KeyboardInterrupt:
                previous_sys_hook(exc_type, exc_value, exc_traceback)
                return
            logging.getLogger("uncaught").critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback),
            )
            previous_sys_hook(exc_type, exc_value, exc_traceback)

        sys.excepthook = sys_hook

        if hasattr(threading, "excepthook"):
            previous_thread_hook = threading.excepthook

            def thread_hook(args):
                logging.getLogger("uncaught.thread").critical(
                    "Uncaught exception in thread %s",
                    getattr(args.thread, "name", "unknown"),
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )
                previous_thread_hook(args)

            threading.excepthook = thread_hook
        _HOOKS_INSTALLED = True

    return log_path


def allocate_spectrum_cube(
    shape: Sequence[int],
    dtype=np.float32,
    max_ram_mb: Optional[int] = None,
    cache_dir: str = "raman_cache",
):
    """Allocate a hyperspectral cube in RAM or transparently as a memmap.

    The original code always allocated float64 in RAM. Large XY maps can push
    Windows into paging, which appears as a whole-PC freeze. The default limit
    is 512 MiB and can be changed with ``RAMAN_MAX_CUBE_RAM_MB``.
    """
    normalized_shape = tuple(int(value) for value in shape)
    if not normalized_shape or any(value <= 0 for value in normalized_shape):
        raise ValueError(f"Invalid cube shape: {shape!r}")

    dtype = np.dtype(dtype)
    required_bytes = int(np.prod(normalized_shape, dtype=np.int64)) * dtype.itemsize
    limit_mb = int(
        max_ram_mb
        if max_ram_mb is not None
        else os.getenv("RAMAN_MAX_CUBE_RAM_MB", "512")
    )
    limit_bytes = max(64, limit_mb) * 1024 * 1024

    if required_bytes <= limit_bytes:
        _LOG.info(
            "Allocating hyperspectral cube in RAM: shape=%s dtype=%s %.1f MiB",
            normalized_shape,
            dtype,
            required_bytes / 1024**2,
        )
        return np.full(normalized_shape, np.nan, dtype=dtype)

    max_disk_gb = float(os.getenv("RAMAN_MAX_CUBE_DISK_GB", "32"))
    if required_bytes > max_disk_gb * 1024**3:
        raise MemoryError(
            f"Hyperspectral cube requires {required_bytes / 1024**3:.2f} GiB, "
            f"above RAMAN_MAX_CUBE_DISK_GB={max_disk_gb:g}. "
            "Reduce scan points, disable full-spectrum saving, or raise the limit explicitly."
        )

    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(directory).free
    reserve_bytes = max(2 * 1024**3, int(required_bytes * 0.20))
    if free_bytes < required_bytes + reserve_bytes:
        raise OSError(
            f"Not enough free disk space for hyperspectral cache: "
            f"need {(required_bytes + reserve_bytes) / 1024**3:.2f} GiB including reserve, "
            f"available {free_bytes / 1024**3:.2f} GiB."
        )

    fd, filename = tempfile.mkstemp(prefix="raw_spectra_", suffix=".mmap", dir=directory)
    os.close(fd)
    cube = np.memmap(filename, mode="w+", dtype=dtype, shape=normalized_shape)
    # Initialize in chunks so a very large map never creates a similarly large
    # temporary array. This still reserves/writes the requested disk capacity.
    flat = cube.reshape(-1)
    chunk_items = max(1, (16 * 1024 * 1024) // dtype.itemsize)
    for start in range(0, flat.size, chunk_items):
        flat[start : start + chunk_items] = np.nan
    cube.flush()
    _LOG.warning(
        "Hyperspectral cube moved to disk-backed memmap: %s (%.2f GiB)",
        filename,
        required_bytes / 1024**3,
    )
    return cube


@dataclass(frozen=True)
class TemperaturePairConfig:
    pair_id: object
    anti_min: float
    anti_max: float
    stokes_min: float
    stokes_max: float
    bg_min: float
    bg_max: float
    c_factor: float
    laser_wl_nm: float

    def calculate(self, x_axis: np.ndarray, spectrum: np.ndarray) -> Optional[float]:
        x_axis = np.asarray(x_axis)
        spectrum = np.asarray(spectrum)
        anti_mask = (x_axis >= self.anti_min) & (x_axis <= self.anti_max)
        stokes_mask = (x_axis >= self.stokes_min) & (x_axis <= self.stokes_max)
        bg_mask = (x_axis >= self.bg_min) & (x_axis <= self.bg_max)
        if not (anti_mask.any() and stokes_mask.any() and bg_mask.any()):
            return None

        bg_baseline = float(np.nanmedian(spectrum[bg_mask]))
        anti_net = float(np.nanmean(spectrum[anti_mask]) - bg_baseline)
        stokes_net = float(np.nanmean(spectrum[stokes_mask]) - bg_baseline)
        if anti_net <= 0 or stokes_net <= 0:
            return None

        v0 = 1e7 / self.laser_wl_nm
        vm = abs((self.stokes_min + self.stokes_max) / 2.0)
        if vm <= 0 or abs(v0 - vm) < 1e-12:
            return None
        ratio = anti_net / stokes_net
        correction = ((v0 + vm) / (v0 - vm)) ** 4
        ln_term = (self.c_factor * correction) / ratio
        if not np.isfinite(ln_term) or ln_term <= 1.0:
            return None

        temperature_k = (1.43877 * vm) / np.log(ln_term)
        if not np.isfinite(temperature_k):
            return None
        return float(temperature_k - 273.15)


def snapshot_temperature_pairs(pairs, laser_wl_nm: float):
    """Read Qt-backed temperature controls in the GUI thread into pure data."""
    snapshots = []
    for pair in list(pairs):
        anti_min, anti_max = pair.region_anti.getRegion()
        stokes_min, stokes_max = pair.region_stokes.getRegion()
        bg_min, bg_max = pair.region_bg.getRegion()
        snapshots.append(
            TemperaturePairConfig(
                pair_id=pair.pair_id,
                anti_min=float(anti_min),
                anti_max=float(anti_max),
                stokes_min=float(stokes_min),
                stokes_max=float(stokes_max),
                bg_min=float(bg_min),
                bg_max=float(bg_max),
                c_factor=float(pair.spin_c_factor.value()),
                laser_wl_nm=float(laser_wl_nm),
            )
        )
    return snapshots


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}


def safe_eval_formula(expression: str, values: Mapping[str, float]) -> float:
    """Evaluate arithmetic formulas without Python ``eval``."""
    expression = str(expression)
    aliases: dict[str, float] = {}
    rewritten = expression
    # Channel names may contain spaces/Korean characters. Replace them with
    # generated valid identifiers before parsing.
    for index, name in enumerate(sorted(values, key=len, reverse=True)):
        value = values[name]
        if value is None or not np.isfinite(value):
            return float("nan")
        alias = f"_channel_{index}"
        rewritten = rewritten.replace(name, alias)
        aliases[alias] = float(value)

    tree = ast.parse(rewritten, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in aliases:
            return aliases[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            # Prevent intentionally or accidentally enormous exponentiation.
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent magnitude is limited to 12")
            return _BINARY_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](evaluate(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*(evaluate(arg) for arg in node.args))
        raise ValueError(f"Unsupported formula element: {ast.dump(node)}")

    result = float(evaluate(tree))
    return result if np.isfinite(result) else float("nan")


def shutdown_hardware(window, worker_timeout_ms: int = 3000) -> None:
    """Best-effort ordered shutdown for use from a MainWindow.closeEvent."""
    # Stop GUI timers first so they cannot start a new camera call while closing.
    for name in ("live_view", "camera_view", "live_view_widget"):
        widget = getattr(window, name, None)
        timer = getattr(widget, "timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                _LOG.exception("Failed to stop %s timer", name)

    for value in list(vars(window).values()):
        if hasattr(value, "isRunning") and hasattr(value, "stop"):
            try:
                if value.isRunning():
                    value.stop()
                    value.wait(worker_timeout_ms)
            except Exception:
                _LOG.exception("Failed to stop worker %r", value)

    stage = getattr(window, "stage", None)
    if stage is not None:
        try:
            if getattr(stage, "is_connected", False):
                stage.stop_motion()
            stage.disconnect()
        except Exception:
            _LOG.exception("Stage shutdown failed")

    camera = getattr(window, "cam", None)
    if camera is not None:
        try:
            camera.disconnect()
            camera.uninitialize_dcam()
        except Exception:
            _LOG.exception("Camera shutdown failed")
