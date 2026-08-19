"""Dark correction and detector quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .models import FrameQuality


@dataclass(frozen=True)
class PreprocessedFrame:
    corrected: np.ndarray
    saturation_mask: np.ndarray
    quality: FrameQuality


def build_dark_reference(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Build a robust median dark reference from equal-shaped frames."""

    arrays = [np.asarray(frame) for frame in frames]
    if not arrays:
        raise ValueError("At least one dark frame is required.")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("All dark frames must have the same shape.")
    return np.median(np.stack(arrays).astype(np.float32), axis=0)


def preprocess_frame(
    frame: np.ndarray,
    dark_reference: np.ndarray | None,
    sensor_maximum: float,
    saturation_level: float = 0.98,
) -> PreprocessedFrame:
    raw = np.asarray(frame)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D detector frame, got shape {raw.shape}.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Detector frame contains non-finite values.")
    if sensor_maximum <= 0:
        raise ValueError("sensor_maximum must be positive.")
    if not 0.5 <= saturation_level <= 1.0:
        raise ValueError("saturation_level must be between 0.5 and 1.0.")

    raw_float = raw.astype(np.float32, copy=False)
    if dark_reference is None:
        corrected = raw_float.copy()
    else:
        dark = np.asarray(dark_reference, dtype=np.float32)
        if dark.shape != raw.shape:
            raise ValueError(
                f"Dark reference shape {dark.shape} does not match frame shape {raw.shape}."
            )
        corrected = raw_float - dark

    saturation_mask = raw_float >= float(sensor_maximum) * saturation_level
    raw_maximum = float(np.max(raw_float))
    saturation_fraction = float(np.mean(saturation_mask))
    quality = FrameQuality(
        sensor_maximum=float(sensor_maximum),
        raw_maximum=raw_maximum,
        saturation_fraction=saturation_fraction,
        headroom_fraction=float(max(0.0, 1.0 - raw_maximum / float(sensor_maximum))),
        saturated=bool(np.any(saturation_mask)),
    )
    return PreprocessedFrame(corrected, saturation_mask, quality)


__all__ = ["PreprocessedFrame", "build_dark_reference", "preprocess_frame"]
