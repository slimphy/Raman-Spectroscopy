"""One-dimensional peak, symmetry, and derivative metrics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

from .models import PeakMetrics
from .peak_models import fit_asymmetric_peak


def _trapezoid(values: np.ndarray, coordinate: np.ndarray) -> float:
    return float(trapezoid(values, coordinate))


def robust_noise(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64)
    if y.size < 5:
        return float("inf")
    residual = y - gaussian_filter1d(y, sigma=max(1.0, min(6.0, y.size / 20.0)), mode="nearest")
    centered = residual - np.median(residual)
    return float(max(1.4826 * np.median(np.abs(centered)), 1e-12))


def _balance(left: float, right: float) -> float:
    maximum = max(abs(left), abs(right))
    if maximum <= 1e-15:
        return float("nan")
    return float(min(abs(left), abs(right)) / maximum)


def _empty_metrics(reason: str, x: np.ndarray, y: np.ndarray) -> PeakMetrics:
    zeros = np.zeros_like(y, dtype=np.float64)
    return PeakMetrics(
        valid=False,
        reason=reason,
        center=float("nan"),
        height=float("nan"),
        area=float("nan"),
        snr=float("nan"),
        noise=float("nan"),
        fwhm=float("nan"),
        hwhm_left=float("nan"),
        hwhm_right=float("nan"),
        width_symmetry=float("nan"),
        area_asymmetry=float("nan"),
        mirror_nrmse=float("nan"),
        derivative_amplitude_balance=float("nan"),
        derivative_position_balance=float("nan"),
        derivative_area_balance=float("nan"),
        lorentz_fraction=float("nan"),
        fit_nrmse=float("nan"),
        roi_x=x,
        roi_y=y,
        baseline_y=zeros,
        fitted_y=zeros,
        negative_second_derivative=zeros,
    )


def analyze_peak(
    coordinate: np.ndarray,
    spectrum: np.ndarray,
    roi_start: int,
    roi_stop: int,
    minimum_snr: float = 5.0,
) -> PeakMetrics:
    x_all = np.asarray(coordinate, dtype=np.float64)
    y_all = np.asarray(spectrum, dtype=np.float64)
    if x_all.ndim != 1 or y_all.ndim != 1 or x_all.size != y_all.size:
        raise ValueError("coordinate and spectrum must be equal-length 1D arrays.")
    start = max(0, int(roi_start))
    stop = min(y_all.size, int(roi_stop))
    x = x_all[start:stop]
    y = y_all[start:stop]
    if y.size < 11:
        return _empty_metrics("Peak ROI is too small", x, y)

    fit = fit_asymmetric_peak(x, y)
    if not fit.success:
        return _empty_metrics(fit.reason, x, y)

    signal = y - fit.baseline
    noise = robust_noise(y - fit.fitted)
    snr = float(fit.amplitude / noise)
    common_distance = min(fit.center - x[0], x[-1] - fit.center)
    spacing = float(np.median(np.diff(x)))
    if common_distance <= 2.0 * spacing:
        return _empty_metrics("Peak center is too close to the ROI boundary", x, y)

    distances = np.arange(0.0, common_distance + 0.25 * spacing, spacing)
    left_values = np.interp(fit.center - distances, x, signal)
    right_values = np.interp(fit.center + distances, x, signal)
    left_positive = np.maximum(left_values, 0.0)
    right_positive = np.maximum(right_values, 0.0)
    left_area = _trapezoid(left_positive, distances)
    right_area = _trapezoid(right_positive, distances)
    area_sum = left_area + right_area
    area_asymmetry = (
        float(abs(left_area - right_area) / area_sum) if area_sum > 1e-15 else float("nan")
    )
    mirror_nrmse = float(
        np.sqrt(np.mean((left_values - right_values) ** 2)) / max(fit.amplitude, 1e-12)
    )

    total_fwhm = float(fit.hwhm_left + fit.hwhm_right)
    desired_window = max(7, int(round(0.65 * total_fwhm / spacing)) | 1)
    maximum_window = y.size if y.size % 2 == 1 else y.size - 1
    window = min(desired_window, maximum_window)
    if window < 5:
        negative_second = -np.gradient(np.gradient(signal, x), x)
    else:
        polyorder = min(3, window - 2)
        negative_second = -savgol_filter(
            signal, window_length=window, polyorder=polyorder, deriv=2, delta=spacing, mode="interp"
        )

    left_distance = fit.center - x
    right_distance = x - fit.center
    minimum_lobe_distance = max(1.5 * spacing, 0.18 * total_fwhm)
    maximum_lobe_distance = max(minimum_lobe_distance + spacing, 1.5 * total_fwhm)
    left_mask = (left_distance >= minimum_lobe_distance) & (left_distance <= maximum_lobe_distance)
    right_mask = (right_distance >= minimum_lobe_distance) & (right_distance <= maximum_lobe_distance)
    if np.any(left_mask) and np.any(right_mask):
        left_indices = np.flatnonzero(left_mask)
        right_indices = np.flatnonzero(right_mask)
        left_index = int(left_indices[np.argmin(negative_second[left_indices])])
        right_index = int(right_indices[np.argmin(negative_second[right_indices])])
        left_amplitude = max(0.0, -float(negative_second[left_index]))
        right_amplitude = max(0.0, -float(negative_second[right_index]))
        derivative_amplitude_balance = _balance(left_amplitude, right_amplitude)
        derivative_position_balance = _balance(
            fit.center - float(x[left_index]), float(x[right_index]) - fit.center
        )
        left_lobe_area = _trapezoid(
            np.maximum(-negative_second[left_mask], 0.0), x[left_mask]
        )
        right_lobe_area = _trapezoid(
            np.maximum(-negative_second[right_mask], 0.0), x[right_mask]
        )
        derivative_area_balance = _balance(left_lobe_area, right_lobe_area)
    else:
        derivative_amplitude_balance = float("nan")
        derivative_position_balance = float("nan")
        derivative_area_balance = float("nan")

    positive_signal = np.maximum(signal, 0.0)
    area = _trapezoid(positive_signal, x)
    valid = bool(snr >= minimum_snr and np.isfinite(total_fwhm))
    reason = "" if valid else f"Peak SNR {snr:.2f} is below {minimum_snr:.2f}"
    return PeakMetrics(
        valid=valid,
        reason=reason,
        center=fit.center,
        height=fit.amplitude,
        area=area,
        snr=snr,
        noise=noise,
        fwhm=total_fwhm,
        hwhm_left=fit.hwhm_left,
        hwhm_right=fit.hwhm_right,
        width_symmetry=_balance(fit.hwhm_left, fit.hwhm_right),
        area_asymmetry=area_asymmetry,
        mirror_nrmse=mirror_nrmse,
        derivative_amplitude_balance=derivative_amplitude_balance,
        derivative_position_balance=derivative_position_balance,
        derivative_area_balance=derivative_area_balance,
        lorentz_fraction=fit.lorentz_fraction,
        fit_nrmse=fit.normalized_rmse,
        roi_x=x,
        roi_y=y,
        baseline_y=fit.baseline,
        fitted_y=fit.fitted,
        negative_second_derivative=negative_second,
    )


__all__ = ["analyze_peak", "robust_noise"]
