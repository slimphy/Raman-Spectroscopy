"""Stable output paths for mapping results."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MAPPING_RESULTS_DIR = PROJECT_ROOT / "mapping_results"


def mapping_results_dir() -> Path:
    """Return the fixed mapping output directory, creating it if necessary."""
    MAPPING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return MAPPING_RESULTS_DIR


def mapping_result_path(filename: str | Path) -> Path:
    """Place a filename in the fixed output directory regardless of process cwd."""
    name = Path(filename).name
    if not name:
        raise ValueError("A mapping result filename is required.")
    return mapping_results_dir() / name
