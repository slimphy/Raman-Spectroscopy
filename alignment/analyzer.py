"""Frame-level orchestration for the Phase 1 alignment monitor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter

from .metrics_1d import analyze_peak
from .metrics_2d import analyze_trace
from .models import AlignmentResult
from .preprocessing import preprocess_frame


@dataclass(frozen=True)
class AlignmentConfig:
    peak_center_pixel: float | None = None
    peak_half_window_pixels: int = 60
    dispersion_axis: int = 1
    detector_maximum: float = 65535.0
    saturation_level: float = 0.98
    minimum_peak_snr: float = 5.0
    minimum_row_snr: float = 4.0
    minimum_valid_row_ratio: float = 0.15
    row_step: int = 4
    reference_raman_shift_cm1: float = 520.0
    raman_shift_per_pixel: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_strongest_peak(projection: np.ndarray) -> float:
    values = np.asarray(projection, dtype=np.float64)
    if values.ndim != 1 or values.size < 11:
        raise ValueError("projection must be a one-dimensional spectrum.")
    baseline_size = min(401, max(31, (values.size // 8) | 1))
    baseline = percentile_filter(values, percentile=15, size=baseline_size, mode="nearest")
    signal = gaussian_filter1d(values - baseline, sigma=1.25, mode="nearest")
    return float(np.argmax(signal))


def analyze_alignment_frame(
    frame: np.ndarray,
    dark_reference: np.ndarray | None,
    config: AlignmentConfig,
    timestamp: float | None = None,
) -> AlignmentResult:
    started = time.perf_counter()
    preprocessed = preprocess_frame(
        frame,
        dark_reference,
        sensor_maximum=config.detector_maximum,
        saturation_level=config.saturation_level,
    )
    if config.dispersion_axis == 1:
        corrected = preprocessed.corrected
        saturation_mask = preprocessed.saturation_mask
    elif config.dispersion_axis == 0:
        corrected = preprocessed.corrected.T
        saturation_mask = preprocessed.saturation_mask.T
    else:
        raise ValueError("dispersion_axis must be 0 or 1.")

    raw_projection = np.sum(corrected, axis=0, dtype=np.float64)
    selected_peak = (
        detect_strongest_peak(raw_projection)
        if config.peak_center_pixel is None
        else float(config.peak_center_pixel)
    )
    trace = analyze_trace(
        corrected,
        saturation_mask,
        peak_center_pixel=selected_peak,
        peak_half_window_pixels=config.peak_half_window_pixels,
        row_step=config.row_step,
        minimum_row_snr=config.minimum_row_snr,
        minimum_valid_row_ratio=config.minimum_valid_row_ratio,
    )
    x_axis = np.arange(raw_projection.size, dtype=np.float64)
    start = int(np.floor(selected_peak)) - config.peak_half_window_pixels
    stop = int(np.ceil(selected_peak)) + config.peak_half_window_pixels + 1
    peak = analyze_peak(
        x_axis,
        raw_projection,
        roi_start=start,
        roi_stop=stop,
        minimum_snr=config.minimum_peak_snr,
    )
    return AlignmentResult(
        timestamp=float(time.time() if timestamp is None else timestamp),
        corrected_frame=corrected,
        raw_projection=raw_projection,
        selected_peak_pixel=selected_peak,
        quality=preprocessed.quality,
        peak=peak,
        trace=trace,
        analysis_ms=float((time.perf_counter() - started) * 1000.0),
    )


__all__ = ["AlignmentConfig", "analyze_alignment_frame", "detect_strongest_peak"]
