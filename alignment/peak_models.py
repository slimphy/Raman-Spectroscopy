"""Robust analytic peak models used for quantitative alignment metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class AsymmetricPeakFit:
    success: bool
    reason: str
    baseline_intercept: float
    baseline_slope: float
    amplitude: float
    center: float
    hwhm_left: float
    hwhm_right: float
    lorentz_fraction: float
    fitted: np.ndarray
    baseline: np.ndarray
    normalized_rmse: float


def split_pseudo_voigt(
    coordinate: np.ndarray,
    center: float,
    hwhm_left: float,
    hwhm_right: float,
    lorentz_fraction: float,
) -> np.ndarray:
    """Unit-height pseudo-Voigt with independent left and right HWHM."""

    x = np.asarray(coordinate, dtype=np.float64)
    width = np.where(x <= center, hwhm_left, hwhm_right)
    scaled = (x - center) / np.maximum(width, 1e-12)
    gaussian = np.exp(-np.log(2.0) * scaled**2)
    lorentzian = 1.0 / (1.0 + scaled**2)
    eta = float(np.clip(lorentz_fraction, 0.0, 1.0))
    return (1.0 - eta) * gaussian + eta * lorentzian


def _initial_half_width(x: np.ndarray, signal: np.ndarray, peak_index: int) -> tuple[float, float]:
    peak = float(signal[peak_index])
    half = 0.5 * peak
    left_candidates = np.flatnonzero(signal[:peak_index] <= half)
    right_candidates = np.flatnonzero(signal[peak_index + 1 :] <= half)
    spacing = float(np.median(np.diff(x)))
    fallback = max(2.0 * spacing, 0.08 * float(x[-1] - x[0]))
    if len(left_candidates):
        left = float(x[peak_index] - x[left_candidates[-1]])
    else:
        left = fallback
    if len(right_candidates):
        right_index = peak_index + 1 + int(right_candidates[0])
        right = float(x[right_index] - x[peak_index])
    else:
        right = fallback
    return max(left, spacing), max(right, spacing)


def fit_asymmetric_peak(coordinate: np.ndarray, values: np.ndarray) -> AsymmetricPeakFit:
    x = np.asarray(coordinate, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("coordinate and values must be equal-length 1D arrays.")
    if x.size < 11 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return AsymmetricPeakFit(
            False, "Insufficient finite samples", *(0.0,) * 7,
            np.zeros_like(y), np.zeros_like(y), float("inf")
        )
    if np.any(np.diff(x) <= 0):
        raise ValueError("coordinate must be strictly increasing.")

    edge = max(3, min(x.size // 5, 20))
    left_level = float(np.median(y[:edge]))
    right_level = float(np.median(y[-edge:]))
    x_mid = float(np.mean(x))
    span = float(x[-1] - x[0])
    baseline_slope = (right_level - left_level) / max(span, 1e-12)
    baseline_intercept = 0.5 * (left_level + right_level)
    baseline_guess = baseline_intercept + baseline_slope * (x - x_mid)
    signal = y - baseline_guess
    peak_index = int(np.argmax(signal))
    amplitude = float(signal[peak_index])
    if amplitude <= 0:
        return AsymmetricPeakFit(
            False, "No positive peak above local baseline", baseline_intercept,
            baseline_slope, amplitude, float(x[peak_index]), 0.0, 0.0, 0.0,
            baseline_guess, baseline_guess, float("inf")
        )

    hwhm_left, hwhm_right = _initial_half_width(x, signal, peak_index)
    spacing = float(np.median(np.diff(x)))
    scale = max(amplitude, float(np.ptp(y)), 1e-12)
    initial = np.asarray(
        [baseline_intercept, baseline_slope, amplitude, x[peak_index], hwhm_left, hwhm_right, 0.25],
        dtype=np.float64,
    )
    lower = np.asarray(
        [
            float(np.min(y) - 2.0 * scale),
            -5.0 * scale / max(span, spacing),
            0.0,
            x[1],
            0.35 * spacing,
            0.35 * spacing,
            0.0,
        ]
    )
    upper = np.asarray(
        [
            float(np.max(y) + 2.0 * scale),
            5.0 * scale / max(span, spacing),
            5.0 * scale,
            x[-2],
            0.8 * span,
            0.8 * span,
            1.0,
        ]
    )
    initial = np.clip(initial, lower + 1e-9, upper - 1e-9)

    def evaluate(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b0, b1, amp, center, left, right, eta = parameters
        baseline = b0 + b1 * (x - x_mid)
        peak = amp * split_pseudo_voigt(x, center, left, right, eta)
        return baseline + peak, baseline

    try:
        result = least_squares(
            lambda parameters: (evaluate(parameters)[0] - y) / scale,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.02,
            max_nfev=350,
        )
        fitted, baseline = evaluate(result.x)
        nrmse = float(np.sqrt(np.mean((fitted - y) ** 2)) / scale)
        b0, b1, amp, center, left, right, eta = result.x
        success = bool(result.success and np.all(np.isfinite(result.x)) and nrmse < 0.35)
        return AsymmetricPeakFit(
            success=success,
            reason="" if success else "Peak fit did not converge reliably",
            baseline_intercept=float(b0),
            baseline_slope=float(b1),
            amplitude=float(amp),
            center=float(center),
            hwhm_left=float(left),
            hwhm_right=float(right),
            lorentz_fraction=float(eta),
            fitted=fitted,
            baseline=baseline,
            normalized_rmse=nrmse,
        )
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        return AsymmetricPeakFit(
            False, str(exc), baseline_intercept, baseline_slope, amplitude,
            float(x[peak_index]), hwhm_left, hwhm_right, 0.25,
            baseline_guess, baseline_guess, float("inf")
        )


__all__ = ["AsymmetricPeakFit", "fit_asymmetric_peak", "split_pseudo_voigt"]
