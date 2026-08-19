import os
import tempfile
import unittest
from pathlib import Path

from mapping_storage import MAPPING_RESULTS_DIR, mapping_result_path, mapping_results_dir


class MappingStorageTests(unittest.TestCase):
    def test_results_directory_is_fixed_under_project_root(self):
        result_dir = mapping_results_dir()

        self.assertTrue(result_dir.is_absolute())
        self.assertEqual(result_dir, Path(__file__).resolve().parents[1] / "mapping_results")
        self.assertEqual(result_dir, MAPPING_RESULTS_DIR)

    def test_result_path_ignores_process_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous = Path.cwd()
            try:
                os.chdir(temporary_directory)
                result_path = mapping_result_path("Raw_Hyperspectral_Data_test.csv")
            finally:
                os.chdir(previous)

        self.assertEqual(result_path.parent, MAPPING_RESULTS_DIR)

    def test_selected_directory_is_discarded_but_filename_is_preserved(self):
        result_path = mapping_result_path(Path("elsewhere") / "map.csv")

        self.assertEqual(result_path, MAPPING_RESULTS_DIR / "map.csv")


if __name__ == "__main__":
    unittest.main()
