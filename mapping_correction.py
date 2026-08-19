"""Display-only mapping corrections shared by the map view and tests."""

from __future__ import annotations

import numpy as np


ROW_SHIFT_BOTH = "odd_left_even_right"
ROW_SHIFT_ODD_LEFT = "odd_left"
ROW_SHIFT_EVEN_RIGHT = "even_right"
ROW_SHIFT_MODES = (ROW_SHIFT_BOTH, ROW_SHIFT_ODD_LEFT, ROW_SHIFT_EVEN_RIGHT)


def row_x_shift(row_index: int, mode: str) -> int:
    """Return the displayed X-cell shift for a zero-based array row."""
    if row_index < 0:
        raise ValueError("row_index must be zero or greater")
    if mode not in ROW_SHIFT_MODES:
        raise ValueError(f"Unknown row shift mode: {mode}")

    is_one_based_odd = row_index % 2 == 0
    if mode == ROW_SHIFT_BOTH:
        return -1 if is_one_based_odd else 1
    if mode == ROW_SHIFT_ODD_LEFT:
        return -1 if is_one_based_odd else 0
    return 0 if is_one_based_odd else 1


def apply_alternating_row_shift(data: np.ndarray, mode: str) -> np.ndarray:
    """Shift 1-based odd/even rows in X and fill newly exposed cells with NaN."""
    source = np.asarray(data)
    if source.ndim != 2:
        raise ValueError("row shift correction requires a 2D array")

    corrected = np.full(source.shape, np.nan, dtype=np.result_type(source.dtype, float))
    for row_index in range(source.shape[0]):
        shift = row_x_shift(row_index, mode)
        if shift < 0:
            corrected[row_index, :-1] = source[row_index, 1:]
        elif shift > 0:
            corrected[row_index, 1:] = source[row_index, :-1]
        else:
            corrected[row_index, :] = source[row_index, :]
    return corrected


def shifted_display_x_to_source_x(display_x: int, row_index: int, mode: str) -> int:
    """Map a corrected display X index back to the unmodified source array."""
    return int(display_x) - row_x_shift(row_index, mode)
