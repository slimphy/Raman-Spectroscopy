"""Immutable result models shared by analysis and UI code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    sensor_maximum: float
    raw_maximum: float
    saturation_fraction: float
    headroom_fraction: float
    saturated: bool


@dataclass(frozen=True)
class PeakMetrics:
    valid: bool
    reason: str
    center: float
    height: float
    area: float
    snr: float
    noise: float
    fwhm: float
    hwhm_left: float
    hwhm_right: float
    width_symmetry: float
    area_asymmetry: float
    mirror_nrmse: float
    derivative_amplitude_balance: float
    derivative_position_balance: float
    derivative_area_balance: float
    lorentz_fraction: float
    fit_nrmse: float
    roi_x: np.ndarray
    roi_y: np.ndarray
    baseline_y: np.ndarray
    fitted_y: np.ndarray
    negative_second_derivative: np.ndarray


@dataclass(frozen=True)
class TraceMetrics:
    valid: bool
    reason: str
    valid_row_ratio: float
    sampled_rows: np.ndarray
    centers: np.ndarray
    row_fwhm: np.ndarray
    row_areas: np.ndarray
    trace_tilt_px_per_100_rows: float
    trace_center_drift_px: float
    trace_curvature_rms_px: float
    trace_quadratic_coefficient: float
    trace_residual_rms_px: float
    median_row_fwhm_px: float
    row_fwhm_cv: float
    row_area_cv: float
    projection_broadening: float
    vertical_centroid_px: float
    vertical_fwhm_px: float
    vertical_clipping_margin_px: float
    rectified_projection: np.ndarray


@dataclass(frozen=True)
class AlignmentResult:
    timestamp: float
    corrected_frame: np.ndarray
    raw_projection: np.ndarray
    selected_peak_pixel: float
    quality: FrameQuality
    peak: PeakMetrics
    trace: TraceMetrics
    analysis_ms: float


__all__ = [
    "AlignmentResult",
    "FrameQuality",
    "PeakMetrics",
    "TraceMetrics",
]
