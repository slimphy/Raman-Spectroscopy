"""Two-dimensional spectral trace and projection metrics."""

from __future__ import annotations

import numpy as np

from .models import TraceMetrics


def _coefficient_of_variation(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array)) if array.size else 0.0
    return float(np.std(array) / abs(mean)) if abs(mean) > 1e-12 else float("nan")


def _half_max_width(coordinate: np.ndarray, signal: np.ndarray) -> float:
    x = np.asarray(coordinate, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    if x.size < 3 or float(np.max(y)) <= 0:
        return float("nan")
    peak = int(np.argmax(y))
    half = 0.5 * float(y[peak])
    left_candidates = np.flatnonzero(y[:peak] <= half)
    right_candidates = np.flatnonzero(y[peak + 1 :] <= half)
    if not len(left_candidates) or not len(right_candidates):
        return float("nan")
    left_low = int(left_candidates[-1])
    left_high = left_low + 1
    right_high = peak + 1 + int(right_candidates[0])
    right_low = right_high - 1

    def crossing(i0: int, i1: int) -> float:
        y0, y1 = float(y[i0]), float(y[i1])
        if abs(y1 - y0) <= 1e-15:
            return float(x[i0])
        return float(x[i0] + (half - y0) * (x[i1] - x[i0]) / (y1 - y0))

    return crossing(right_low, right_high) - crossing(left_low, left_high)


def _empty_trace(reason: str, point_count: int) -> TraceMetrics:
    empty = np.empty(0, dtype=np.float64)
    return TraceMetrics(
        valid=False,
        reason=reason,
        valid_row_ratio=0.0,
        sampled_rows=empty,
        centers=empty,
        row_fwhm=empty,
        row_areas=empty,
        trace_tilt_px_per_100_rows=float("nan"),
        trace_center_drift_px=float("nan"),
        trace_curvature_rms_px=float("nan"),
        trace_quadratic_coefficient=float("nan"),
        trace_residual_rms_px=float("nan"),
        median_row_fwhm_px=float("nan"),
        row_fwhm_cv=float("nan"),
        row_area_cv=float("nan"),
        projection_broadening=float("nan"),
        vertical_centroid_px=float("nan"),
        vertical_fwhm_px=float("nan"),
        vertical_clipping_margin_px=float("nan"),
        rectified_projection=np.zeros(point_count, dtype=np.float64),
    )


def analyze_trace(
    corrected_frame: np.ndarray,
    saturation_mask: np.ndarray,
    peak_center_pixel: float,
    peak_half_window_pixels: int,
    row_step: int = 4,
    minimum_row_snr: float = 4.0,
    minimum_valid_row_ratio: float = 0.15,
) -> TraceMetrics:
    frame = np.asarray(corrected_frame, dtype=np.float64)
    saturated = np.asarray(saturation_mask, dtype=bool)
    if frame.ndim != 2 or saturated.shape != frame.shape:
        raise ValueError("corrected_frame and saturation_mask must be equal-shaped 2D arrays.")
    height, width = frame.shape
    half_window = max(8, int(peak_half_window_pixels))
    start = max(0, int(np.floor(peak_center_pixel)) - half_window)
    stop = min(width, int(np.ceil(peak_center_pixel)) + half_window + 1)
    if stop - start < 12:
        return _empty_trace("Spectral ROI is too small", width)

    x_local = np.arange(start, stop, dtype=np.float64)
    sampled_rows = np.arange(0, height, max(1, int(row_step)), dtype=int)
    centers: list[float] = []
    fwhms: list[float] = []
    areas: list[float] = []
    rows: list[float] = []
    rows_for_rectification: list[np.ndarray] = []

    edge = max(3, min(12, (stop - start) // 6))
    for row_index in sampled_rows:
        row = frame[row_index, start:stop]
        if np.any(saturated[row_index, start:stop]):
            continue
        left_level = float(np.median(row[:edge]))
        right_level = float(np.median(row[-edge:]))
        baseline = np.linspace(left_level, right_level, row.size)
        signal = row - baseline
        edge_noise_samples = np.concatenate((signal[:edge], signal[-edge:]))
        noise = max(
            1.4826 * float(np.median(np.abs(edge_noise_samples - np.median(edge_noise_samples)))),
            1e-12,
        )
        positive = np.maximum(signal, 0.0)
        area = float(np.sum(positive))
        peak_height = float(np.max(signal))
        if area <= 0 or peak_height / noise < minimum_row_snr:
            continue
        center = float(np.sum(x_local * positive) / area)
        fwhm = _half_max_width(x_local, signal)
        if not np.isfinite(fwhm) or fwhm <= 0:
            continue
        rows.append(float(row_index))
        centers.append(center)
        fwhms.append(fwhm)
        areas.append(area)
        rows_for_rectification.append(frame[row_index])

    row_array = np.asarray(rows, dtype=np.float64)
    center_array = np.asarray(centers, dtype=np.float64)
    fwhm_array = np.asarray(fwhms, dtype=np.float64)
    area_array = np.asarray(areas, dtype=np.float64)
    valid_row_ratio = float(len(row_array) / max(len(sampled_rows), 1))
    if len(row_array) < 8:
        return _empty_trace("Too few valid rows for a trace fit", width)

    quadratic = np.polyfit(row_array, center_array, 2)
    quadratic_prediction = np.polyval(quadratic, row_array)
    residual = center_array - quadratic_prediction
    mad = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    if mad > 1e-12:
        keep = np.abs(residual - np.median(residual)) <= 3.5 * mad
        if np.count_nonzero(keep) >= 8:
            row_array = row_array[keep]
            center_array = center_array[keep]
            fwhm_array = fwhm_array[keep]
            area_array = area_array[keep]
            rows_for_rectification = [row for row, retain in zip(rows_for_rectification, keep) if retain]
            quadratic = np.polyfit(row_array, center_array, 2)

    linear = np.polyfit(row_array, center_array, 1)
    quadratic_prediction = np.polyval(quadratic, row_array)
    linear_prediction = np.polyval(linear, row_array)
    curvature_component = quadratic_prediction - linear_prediction
    trace_residual = center_array - quadratic_prediction
    row_span = float(np.ptp(row_array))
    center_drift = float(abs(linear[0]) * row_span)
    median_row_fwhm = float(np.median(fwhm_array))

    full_x = np.arange(width, dtype=np.float64)
    reference_center = float(np.median(center_array))
    rectified_rows = []
    for row, center in zip(rows_for_rectification, center_array):
        rectified_rows.append(
            np.interp(full_x + (center - reference_center), full_x, row, left=0.0, right=0.0)
        )
    rectified_projection = np.sum(np.stack(rectified_rows), axis=0)

    raw_projection = np.sum(frame, axis=0)
    projection_signal = raw_projection[start:stop]
    projection_baseline = np.linspace(
        float(np.median(projection_signal[:edge])),
        float(np.median(projection_signal[-edge:])),
        projection_signal.size,
    )
    projection_width = _half_max_width(x_local, projection_signal - projection_baseline)
    projection_broadening = (
        float(projection_width / median_row_fwhm - 1.0)
        if np.isfinite(projection_width) and median_row_fwhm > 0
        else float("nan")
    )

    vertical_profile = np.maximum(np.sum(frame[:, start:stop], axis=1), 0.0)
    y_coordinate = np.arange(height, dtype=np.float64)
    vertical_total = float(np.sum(vertical_profile))
    if vertical_total > 0:
        vertical_centroid = float(np.sum(y_coordinate * vertical_profile) / vertical_total)
        vertical_fwhm = _half_max_width(y_coordinate, vertical_profile)
        active = np.flatnonzero(vertical_profile >= 0.05 * float(np.max(vertical_profile)))
        vertical_margin = (
            float(min(active[0], height - 1 - active[-1])) if len(active) else float("nan")
        )
    else:
        vertical_centroid = float("nan")
        vertical_fwhm = float("nan")
        vertical_margin = float("nan")

    valid = valid_row_ratio >= minimum_valid_row_ratio
    reason = "" if valid else f"Valid-row ratio {valid_row_ratio:.2f} is too low"
    return TraceMetrics(
        valid=valid,
        reason=reason,
        valid_row_ratio=valid_row_ratio,
        sampled_rows=row_array,
        centers=center_array,
        row_fwhm=fwhm_array,
        row_areas=area_array,
        trace_tilt_px_per_100_rows=float(linear[0] * 100.0),
        trace_center_drift_px=center_drift,
        trace_curvature_rms_px=float(np.sqrt(np.mean(curvature_component**2))),
        trace_quadratic_coefficient=float(quadratic[0]),
        trace_residual_rms_px=float(np.sqrt(np.mean(trace_residual**2))),
        median_row_fwhm_px=median_row_fwhm,
        row_fwhm_cv=_coefficient_of_variation(fwhm_array),
        row_area_cv=_coefficient_of_variation(area_array),
        projection_broadening=projection_broadening,
        vertical_centroid_px=vertical_centroid,
        vertical_fwhm_px=float(vertical_fwhm),
        vertical_clipping_margin_px=vertical_margin,
        rectified_projection=rectified_projection,
    )


__all__ = ["analyze_trace"]
