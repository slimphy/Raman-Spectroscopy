import unittest

import numpy as np

from mapping_correction import (
    ROW_SHIFT_BOTH,
    ROW_SHIFT_EVEN_RIGHT,
    ROW_SHIFT_ODD_LEFT,
    apply_alternating_row_shift,
    shifted_display_x_to_source_x,
)


class MappingRowShiftTests(unittest.TestCase):
    def setUp(self):
        self.data = np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
                [40.0, 41.0, 42.0, 43.0],
            ]
        )

    def test_odd_rows_shift_left_and_even_rows_shift_right_together(self):
        corrected = apply_alternating_row_shift(self.data, ROW_SHIFT_BOTH)

        np.testing.assert_equal(corrected[0], [11.0, 12.0, 13.0, np.nan])
        np.testing.assert_equal(corrected[1], [np.nan, 20.0, 21.0, 22.0])
        np.testing.assert_equal(corrected[2], [31.0, 32.0, 33.0, np.nan])
        np.testing.assert_equal(corrected[3], [np.nan, 40.0, 41.0, 42.0])

    def test_legacy_single_parity_modes_are_preserved(self):
        odd_only = apply_alternating_row_shift(self.data, ROW_SHIFT_ODD_LEFT)
        even_only = apply_alternating_row_shift(self.data, ROW_SHIFT_EVEN_RIGHT)

        np.testing.assert_equal(odd_only[1], self.data[1])
        np.testing.assert_equal(even_only[0], self.data[0])

    def test_display_click_maps_back_to_source_cell(self):
        self.assertEqual(shifted_display_x_to_source_x(1, 0, ROW_SHIFT_BOTH), 2)
        self.assertEqual(shifted_display_x_to_source_x(2, 1, ROW_SHIFT_BOTH), 1)


if __name__ == "__main__":
    unittest.main()
