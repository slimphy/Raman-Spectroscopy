"""Hardware profiles used by the alignment application."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    camera_model: str
    sensor_width_pixels: int
    sensor_height_pixels: int
    pixel_size_um: float
    output_bit_depth: int
    cylindrical_dispersion_focal_length_mm: float
    cylindrical_spatial_focal_length_mm: float
    grating_grooves_per_mm: float
    grating_blaze_nm: float
    reference_name: str
    reference_raman_shift_cm1: float
    reference_kind: str

    @property
    def detector_maximum(self) -> int:
        return (1 << self.output_bit_depth) - 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ORCA_QUEST2_SI_520_PROFILE = HardwareProfile(
    name="ORCA-Quest 2 / Si 520 alignment",
    camera_model="Hamamatsu ORCA-Quest 2 C15550-22UP",
    sensor_width_pixels=4096,
    sensor_height_pixels=2304,
    pixel_size_um=4.6,
    output_bit_depth=16,
    cylindrical_dispersion_focal_length_mm=300.0,
    cylindrical_spatial_focal_length_mm=75.0,
    grating_grooves_per_mm=600.0,
    grating_blaze_nm=500.0,
    reference_name="Silicon Stokes peak",
    reference_raman_shift_cm1=520.0,
    reference_kind="end_to_end_raman_reference",
)


__all__ = ["HardwareProfile", "ORCA_QUEST2_SI_520_PROFILE"]
