import json
import tempfile
import unittest
from pathlib import Path

import vision_test_support as support
from vision_test_support import (
    CSN_POSITION_TO_OBJECTIVE,
    OBJECTIVE_TO_CSN_POSITION,
    CalibrationStore,
    configure_ic4_auto_brightness,
    objective_stage_delta_um,
    roi_to_mapping_ranges,
    round_mapping_bounds,
    translate_mapping_xy_bounds,
)


@unittest.skipIf(support.ic4 is None, "IC4 SDK is not installed")
class AutoBrightnessTests(unittest.TestCase):
    class FakePropertyMap:
        def __init__(self, should_fail=False):
            self.calls = []
            self.should_fail = should_fail

        def set_value(self, property_id, value):
            self.calls.append((property_id, value))
            if self.should_fail:
                raise RuntimeError("not writable")

    def test_enables_exposure_and_gain_auto(self):
        property_map = self.FakePropertyMap()
        configure_ic4_auto_brightness(property_map, True)

        self.assertEqual([value for _, value in property_map.calls], ["Continuous", "Continuous"])

    def test_disables_exposure_and_gain_auto(self):
        property_map = self.FakePropertyMap()
        configure_ic4_auto_brightness(property_map, False)

        self.assertEqual([value for _, value in property_map.calls], ["Off", "Off"])

    def test_reports_unavailable_camera_controls(self):
        with self.assertRaisesRegex(RuntimeError, "ExposureAuto.*GainAuto"):
            configure_ic4_auto_brightness(self.FakePropertyMap(should_fail=True), True)


class CalibrationStoreTests(unittest.TestCase):
    def test_profiles_are_averaged_and_saved_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "vision_calibration.json"
            store = CalibrationStore(path)
            store.add_sample("20X", known_length_um=100, pixel_length=200)
            store.add_sample("20X", known_length_um=100, pixel_length=250)
            store.add_sample("100X", known_length_um=10, pixel_length=100)

            self.assertAlmostEqual(store.average_scale("20X"), 0.45)
            self.assertAlmostEqual(store.average_scale("100X"), 0.1)
            store.save()

            reloaded = CalibrationStore(path)
            self.assertAlmostEqual(reloaded.average_scale("20X"), 0.45)
            self.assertAlmostEqual(reloaded.average_scale("100X"), 0.1)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["profiles"]["20X"]["sample_count"], 2)
            self.assertEqual(payload["profiles"]["100X"]["sample_count"], 1)

    def test_invalid_measurement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CalibrationStore(Path(temp_dir) / "cal.json")
            with self.assertRaises(ValueError):
                store.add_sample("20X", 100, 0)


class VisionMappingTests(unittest.TestCase):
    def test_csn_positions_match_installed_objectives(self):
        self.assertEqual(OBJECTIVE_TO_CSN_POSITION, {"20X": 1, "100X": 2})
        self.assertEqual(CSN_POSITION_TO_OBJECTIVE[1], "20X")
        self.assertEqual(CSN_POSITION_TO_OBJECTIVE[2], "100X")

    def test_objective_stage_compensation_is_reversible(self):
        self.assertEqual(objective_stage_delta_um("100X", "20X"), (68.0, 31.0, -72.0))
        self.assertEqual(objective_stage_delta_um("20X", "100X"), (-68.0, -31.0, 72.0))

    def test_roi_is_converted_relative_to_camera_center(self):
        ranges = roi_to_mapping_ranges(
            roi=(400.0, 200.0, 600.0, 500.0),
            source_size=(1000, 800),
            um_per_pixel=0.5,
        )

        self.assertEqual(ranges["x_start"], -50.0)
        self.assertEqual(ranges["x_end"], 50.0)
        self.assertEqual(ranges["y_start"], -50.0)
        self.assertEqual(ranges["y_end"], 100.0)
        self.assertEqual(ranges["center_x_um"], 0.0)
        self.assertEqual(ranges["center_y_um"], 25.0)
        self.assertEqual(ranges["width_um"], 100.0)
        self.assertEqual(ranges["height_um"], 150.0)

    def test_vision_top_is_positive_stage_y_and_bottom_is_negative(self):
        top_roi = roi_to_mapping_ranges(
            roi=(400.0, 100.0, 600.0, 200.0),
            source_size=(1000, 800),
            um_per_pixel=0.5,
        )
        bottom_roi = roi_to_mapping_ranges(
            roi=(400.0, 600.0, 600.0, 700.0),
            source_size=(1000, 800),
            um_per_pixel=0.5,
        )

        self.assertGreater(top_roi["center_y_um"], 0.0)
        self.assertGreater(top_roi["y_start"], 0.0)
        self.assertLess(bottom_roi["center_y_um"], 0.0)
        self.assertLess(bottom_roi["y_end"], 0.0)

    def test_mapping_bounds_are_rounded_to_one_decimal_before_sending(self):
        rounded = round_mapping_bounds(
            {
                "x_start": -13.769,
                "x_end": 13.749,
                "y_start": -27.55,
                "y_end": 13.75,
            }
        )

        self.assertEqual(
            rounded,
            {
                "x_start": -13.8,
                "x_end": 13.7,
                "y_start": -27.6,
                "y_end": 13.8,
            },
        )

    def test_mapping_xy_bounds_follow_objective_compensation_without_z(self):
        shifted = translate_mapping_xy_bounds(
            {
                "x_start": 58.2,
                "x_end": 77.8,
                "y_start": 20.4,
                "y_end": 41.6,
            },
            dx_um=-68.0,
            dy_um=-31.0,
        )

        self.assertEqual(
            shifted,
            {
                "x_start": -9.8,
                "x_end": 9.8,
                "y_start": -10.6,
                "y_end": 10.6,
            },
        )


if __name__ == "__main__":
    unittest.main()
