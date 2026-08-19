"""Small, traceable Phase 1 alignment session exports."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyzer import AlignmentConfig
from .models import AlignmentResult
from .profile import HardwareProfile
from .stability import result_metric_values


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


class AlignmentSession:
    def __init__(self, profile: HardwareProfile):
        self.profile = profile
        self.started_at = datetime.now().astimezone()
        self.metric_rows: list[dict[str, Any]] = []
        self.snapshots: dict[str, np.ndarray] = {}
        self.snapshot_summaries: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        self.started_at = datetime.now().astimezone()
        self.metric_rows.clear()
        self.snapshots.clear()
        self.snapshot_summaries.clear()

    def record(self, result: AlignmentResult) -> None:
        row: dict[str, Any] = {"timestamp": result.timestamp}
        row.update({name: _finite_or_none(value) for name, value in result_metric_values(result).items()})
        row["saturation_fraction"] = result.quality.saturation_fraction
        row["headroom_fraction"] = result.quality.headroom_fraction
        row["analysis_ms"] = result.analysis_ms
        self.metric_rows.append(row)

    def mark(self, label: str, raw_frame: np.ndarray, result: AlignmentResult) -> None:
        safe_label = "".join(character for character in label if character.isalnum() or character in "-_")
        if not safe_label:
            raise ValueError("Snapshot label must contain a letter or number.")
        self.snapshots[f"{safe_label}_raw"] = np.asarray(raw_frame).copy()
        self.snapshots[f"{safe_label}_corrected"] = result.corrected_frame.astype(np.float32, copy=True)
        self.snapshots[f"{safe_label}_projection"] = result.raw_projection.astype(np.float64, copy=True)
        self.snapshot_summaries[safe_label] = {
            "timestamp": result.timestamp,
            "selected_peak_pixel": result.selected_peak_pixel,
            "quality": asdict(result.quality),
            "peak": {
                key: _finite_or_none(float(getattr(result.peak, key)))
                for key in (
                    "center", "height", "area", "snr", "fwhm", "hwhm_left", "hwhm_right",
                    "width_symmetry", "area_asymmetry", "mirror_nrmse",
                    "derivative_amplitude_balance", "derivative_position_balance",
                    "derivative_area_balance", "lorentz_fraction", "fit_nrmse",
                )
            },
            "trace": {
                key: _finite_or_none(float(getattr(result.trace, key)))
                for key in (
                    "valid_row_ratio", "trace_tilt_px_per_100_rows", "trace_center_drift_px",
                    "trace_curvature_rms_px", "trace_quadratic_coefficient",
                    "trace_residual_rms_px", "median_row_fwhm_px", "row_fwhm_cv",
                    "row_area_cv", "projection_broadening", "vertical_centroid_px",
                    "vertical_fwhm_px", "vertical_clipping_margin_px",
                    "spatial_valid_column_count", "spectrum_tilt_rows_per_100_columns",
                    "spectrum_vertical_drift_px", "spectrum_center_residual_rms_px",
                )
            },
        }

    def save(
        self,
        parent_directory: str | Path,
        config: AlignmentConfig,
        dark_reference: np.ndarray | None,
        notes: str = "",
    ) -> Path:
        parent = Path(parent_directory)
        session_name = f"alignment_{self.started_at.strftime('%Y%m%d_%H%M%S')}"
        output = parent / session_name
        suffix = 1
        while output.exists():
            output = parent / f"{session_name}_{suffix:02d}"
            suffix += 1
        output.mkdir(parents=True)

        metadata = {
            "format_version": 1,
            "started_at": self.started_at.isoformat(),
            "saved_at": datetime.now().astimezone().isoformat(),
            "hardware_profile": self.profile.to_dict(),
            "analysis_config": config.to_dict(),
            "notes": notes,
            "metric_row_count": len(self.metric_rows),
            "snapshots": self.snapshot_summaries,
            "dark_reference_saved": dark_reference is not None,
            "reference_caveat": (
                "The Si 520 cm^-1 band is an end-to-end Raman reference. Its measured width includes "
                "the intrinsic sample line shape and is not an absolute instrument LSF measurement."
            ),
        }
        (output / "session.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if self.metric_rows:
            fieldnames: list[str] = []
            for row in self.metric_rows:
                for name in row:
                    if name not in fieldnames:
                        fieldnames.append(name)
            with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.metric_rows)

        arrays = dict(self.snapshots)
        if dark_reference is not None:
            arrays["dark_reference"] = np.asarray(dark_reference, dtype=np.float32)
        if arrays:
            np.savez_compressed(output / "snapshots.npz", **arrays)
        return output


__all__ = ["AlignmentSession"]
