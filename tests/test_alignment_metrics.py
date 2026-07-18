from __future__ import annotations

import json

import numpy as np

from alignment import AlignmentConfig, ORCA_QUEST2_SI_520_PROFILE, analyze_alignment_frame
from alignment.preprocessing import build_dark_reference, preprocess_frame
from alignment.session import AlignmentSession


def synthetic_frame(
    *,
    height: int = 240,
    width: int = 512,
    center: float = 250.0,
    sigma_left: float = 4.0,
    sigma_right: float = 4.0,
    tilt_total: float = 0.0,
    curvature_total: float = 0.0,
    amplitude: float = 800.0,
    noise: float = 1.5,
    seed: int = 123,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.arange(height, dtype=np.float64)
    x = np.arange(width, dtype=np.float64)
    normalized_y = (y - 0.5 * (height - 1)) / max(height - 1, 1)
    row_centers = center + tilt_total * normalized_y
    row_centers += curvature_total * (normalized_y**2 - np.mean(normalized_y**2))
    distance = x[None, :] - row_centers[:, None]
    sigma = np.where(distance <= 0, sigma_left, sigma_right)
    envelope = 0.65 + 0.35 * np.exp(-0.5 * (normalized_y / 0.38) ** 2)
    frame = 200.0 + amplitude * envelope[:, None] * np.exp(-0.5 * (distance / sigma) ** 2)
    frame += rng.normal(0.0, noise, frame.shape)
    return frame.astype(np.float32)


def config(center: float = 250.0) -> AlignmentConfig:
    return AlignmentConfig(
        peak_center_pixel=center,
        peak_half_window_pixels=42,
        detector_maximum=65535.0,
        minimum_peak_snr=5.0,
        minimum_row_snr=4.0,
        row_step=2,
    )


def test_symmetric_peak_and_trace_report_high_symmetry() -> None:
    result = analyze_alignment_frame(synthetic_frame(), None, config())
    assert result.peak.valid
    assert result.trace.valid
    assert abs(result.peak.center - 250.0) < 0.15
    assert 8.5 < result.peak.fwhm < 10.5
    assert result.peak.width_symmetry > 0.97
    assert result.peak.mirror_nrmse < 0.03
    assert result.peak.derivative_amplitude_balance > 0.92
    assert abs(result.trace.trace_tilt_px_per_100_rows) < 0.05
    assert result.trace.trace_curvature_rms_px < 0.05


def test_trace_tilt_increases_drift_and_projection_broadening() -> None:
    straight = analyze_alignment_frame(synthetic_frame(tilt_total=0.0), None, config())
    tilted = analyze_alignment_frame(synthetic_frame(tilt_total=14.0), None, config())
    assert tilted.trace.trace_center_drift_px > 12.0
    assert abs(tilted.trace.trace_tilt_px_per_100_rows) > 5.0
    assert tilted.trace.projection_broadening > straight.trace.projection_broadening + 0.25
    assert abs(tilted.trace.median_row_fwhm_px - straight.trace.median_row_fwhm_px) < 0.5


def test_curvature_and_asymmetry_are_detected_independently() -> None:
    curved = analyze_alignment_frame(synthetic_frame(curvature_total=12.0), None, config())
    asymmetric = analyze_alignment_frame(
        synthetic_frame(sigma_left=3.0, sigma_right=7.0), None, config()
    )
    assert curved.trace.trace_curvature_rms_px > 0.7
    assert curved.trace.trace_center_drift_px < 0.4
    assert asymmetric.peak.width_symmetry < 0.6
    assert asymmetric.peak.area_asymmetry > 0.25
    assert asymmetric.peak.derivative_amplitude_balance < 0.85


def test_dark_reference_and_saturation_gate() -> None:
    base = synthetic_frame(height=32, width=64, center=30.0, amplitude=100.0, noise=0.0)
    dark_frames = [np.full_like(base, 200.0 + offset) for offset in (-1.0, 0.0, 1.0)]
    dark = build_dark_reference(dark_frames)
    processed = preprocess_frame(base, dark, sensor_maximum=1000.0, saturation_level=0.98)
    assert abs(float(np.median(processed.corrected[:, :8]))) < 1e-6
    saturated = base.copy()
    saturated[0, 0] = 1000.0
    quality = preprocess_frame(saturated, dark, sensor_maximum=1000.0).quality
    assert quality.saturated
    assert quality.saturation_fraction == 1.0 / saturated.size


def test_session_exports_traceable_snapshots(tmp_path) -> None:
    frame = synthetic_frame()
    result = analyze_alignment_frame(frame, None, config())
    session = AlignmentSession(ORCA_QUEST2_SI_520_PROFILE)
    session.record(result)
    session.mark("A", frame, result)
    output = session.save(tmp_path, config(), None, notes="test")
    metadata = json.loads((output / "session.json").read_text(encoding="utf-8"))
    assert metadata["hardware_profile"]["camera_model"].startswith("Hamamatsu ORCA-Quest 2")
    assert metadata["snapshots"]["A"]["peak"]["fwhm"] > 0
    assert (output / "metrics.csv").exists()
    with np.load(output / "snapshots.npz") as archive:
        assert archive["A_raw"].shape == frame.shape
