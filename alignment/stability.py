"""Rolling alignment statistics used by the live UI."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .models import AlignmentResult


@dataclass(frozen=True)
class StabilityMetrics:
    sample_count: int
    center_jitter_rms_px: float
    fwhm_cv: float
    area_cv: float
    tilt_std_px_per_100_rows: float


def result_metric_values(result: AlignmentResult) -> dict[str, float]:
    return {
        "Peak center (px)": result.peak.center,
        "Peak FWHM (px)": result.peak.fwhm,
        "Peak area": result.peak.area,
        "Peak SNR": result.peak.snr,
        "Width symmetry": result.peak.width_symmetry,
        "Mirror NRMSE": result.peak.mirror_nrmse,
        "D2 amplitude balance": result.peak.derivative_amplitude_balance,
        "Trace tilt (px/100 rows)": result.trace.trace_tilt_px_per_100_rows,
        "Trace curvature RMS (px)": result.trace.trace_curvature_rms_px,
        "Projection broadening": result.trace.projection_broadening,
        "Row FWHM CV": result.trace.row_fwhm_cv,
        "Valid-row ratio": result.trace.valid_row_ratio,
    }


class MetricHistory:
    def __init__(self, maximum_samples: int = 300):
        self.maximum_samples = max(5, int(maximum_samples))
        self.timestamps: deque[float] = deque(maxlen=self.maximum_samples)
        self.values: dict[str, deque[float]] = {}

    def clear(self) -> None:
        self.timestamps.clear()
        self.values.clear()

    def append(self, result: AlignmentResult) -> None:
        metrics = result_metric_values(result)
        self.timestamps.append(result.timestamp)
        for name, value in metrics.items():
            series = self.values.setdefault(name, deque(maxlen=self.maximum_samples))
            series.append(float(value))

    def series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self.values:
            return np.empty(0), np.empty(0)
        values = np.asarray(self.values[name], dtype=np.float64)
        times = np.asarray(self.timestamps, dtype=np.float64)
        count = min(values.size, times.size)
        if count == 0:
            return np.empty(0), np.empty(0)
        times = times[-count:]
        return times - times[0], values[-count:]

    def stability(self, recent_samples: int = 30) -> StabilityMetrics:
        def recent(name: str) -> np.ndarray:
            data = np.asarray(self.values.get(name, []), dtype=np.float64)
            data = data[-recent_samples:]
            return data[np.isfinite(data)]

        center = recent("Peak center (px)")
        fwhm = recent("Peak FWHM (px)")
        area = recent("Peak area")
        tilt = recent("Trace tilt (px/100 rows)")

        def cv(values: np.ndarray) -> float:
            mean = float(np.mean(values)) if values.size else 0.0
            return float(np.std(values) / abs(mean)) if abs(mean) > 1e-15 else float("nan")

        return StabilityMetrics(
            sample_count=int(min(len(self.timestamps), recent_samples)),
            center_jitter_rms_px=float(np.std(center)) if center.size else float("nan"),
            fwhm_cv=cv(fwhm),
            area_cv=cv(area),
            tilt_std_px_per_100_rows=float(np.std(tilt)) if tilt.size else float("nan"),
        )


__all__ = ["MetricHistory", "StabilityMetrics", "result_metric_values"]
