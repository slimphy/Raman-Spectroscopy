#!/usr/bin/env python3
"""Apply the stability patch to a local Raman-Spectroscopy checkout.

Usage:
    python apply_stability_patch.py C:\\path\\to\\Raman-Spectroscopy
    python apply_stability_patch.py . --dry-run

The script creates timestamped backups, copies the stabilized controller/helper
modules, patches high-confidence regions of main.py, and refuses to save a
main.py that no longer parses as valid Python.
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable


MODULES = (
    "camera_controller.py",
    "stage_controller.py",
    "raman_ml.py",
    "stability_utils.py",
)


class PatchError(RuntimeError):
    pass


def backup(path: Path, stamp: str, dry_run: bool) -> Path:
    destination = path.with_name(f"{path.name}.bak_{stamp}")
    if not dry_run:
        shutil.copy2(path, destination)
    return destination


def replace_once(
    text: str,
    old: str,
    new: str,
    label: str,
    changes: list[str],
    required: bool = False,
) -> str:
    if new in text:
        changes.append(f"SKIP(already): {label}")
        return text
    count = text.count(old)
    if count == 0:
        if required:
            raise PatchError(f"Required pattern not found: {label}")
        changes.append(f"SKIP(not found): {label}")
        return text
    text = text.replace(old, new, 1)
    changes.append(f"APPLY: {label}")
    return text


def replace_regex_once(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
    changes: list[str],
    required: bool = False,
) -> str:
    compiled = re.compile(pattern, re.MULTILINE | re.DOTALL)
    if compiled.search(text) is None:
        if required:
            raise PatchError(f"Required regex not found: {label}")
        changes.append(f"SKIP(not found): {label}")
        return text
    text, count = compiled.subn(replacement, text, count=1)
    changes.append(f"APPLY: {label} ({count})")
    return text


def transform_class_block(
    text: str,
    class_name: str,
    transform: Callable[[str, list[str]], str],
    changes: list[str],
) -> str:
    start_match = re.search(rf"(?m)^class\s+{re.escape(class_name)}\b", text)
    if not start_match:
        changes.append(f"SKIP(not found): class {class_name}")
        return text
    next_match = re.search(r"(?m)^class\s+\w+\b", text[start_match.end() :])
    end = len(text) if next_match is None else start_match.end() + next_match.start()
    block = text[start_match.start() : end]
    patched = transform(block, changes)
    return text[: start_match.start()] + patched + text[end:]


def patch_mapping_worker(block: str, changes: list[str]) -> str:
    if "self._temperature_pairs" not in block:
        marker = "        self.is_running = True\n"
        injection = marker + (
            "        self._x_axis = np.array(main_window.spectrum_view.x_axis, copy=True)\n"
            "        self._virtual_channels = list(main_window.mapping_view.virtual_channels)\n"
            "        self._formulas = list(main_window.mapping_view.formulas)\n"
            "        self._use_hw_trigger = bool(main_window.control_panel.chk_hw_trigger.isChecked())\n"
            "        self._original_speed = float(main_window.control_panel.spin_speed.value())\n"
            "        try:\n"
            "            _laser_wl = float(main_window.control_panel.spin_laser.value())\n"
            "        except Exception:\n"
            "            _laser_wl = 532.0\n"
            "        self._temperature_pairs = snapshot_temperature_pairs(\n"
            "            main_window.control_panel.active_pairs, _laser_wl\n"
            "        )\n"
        )
        block = replace_once(
            block,
            marker,
            injection,
            "MappingWorker UI snapshot in constructor",
            changes,
        )

    replacements = {
        "self.main_window.spectrum_view.x_axis": "self._x_axis",
        "self.main_window.control_panel.active_pairs": "self._temperature_pairs",
        "self.main_window.mapping_view.virtual_channels": "self._virtual_channels",
        "self.main_window.mapping_view.formulas": "self._formulas",
        "self.main_window.control_panel.chk_hw_trigger.isChecked()": "self._use_hw_trigger",
        "self.main_window.control_panel.spin_speed.value()": "self._original_speed",
        "pair.update_temperature(x_axis, processed_1d, is_mapping=True)": "pair.calculate(x_axis, processed_1d)",
        "int(timeout_limit / 0.015)": "max(1, int(timeout_limit / 0.11))",
    }
    for old, new in replacements.items():
        if old in block:
            block = block.replace(old, new)
            changes.append(f"APPLY: MappingWorker expression {old}")
        else:
            changes.append(f"SKIP(not found): MappingWorker expression {old}")


    mapping_start_pattern = (
        r'(?P<i>^[ \t]*)try:\n'
        r'(?P=i)    if use_hw_trigger and cam\.is_connected and stage\.is_connected:\n'
        r'(?P=i)        cam\.stop_capture\(\)\n'
        r'(?P=i)        cam\.set_trigger_mode\("EXTERNAL"\)\n'
        r'(?P=i)        cam\.start_capture\(\)\n'
        r'(?P=i)        cam\.grab_frame\(\)\n'
        r'(?P=i)        stage\.set_trigger_out\(axis=fast_ax, value=\'0\.0\'\)\n'
    )
    mapping_start_replacement = (
        r'\g<i>camera_claimed = False\n'
        r'\g<i>try:\n'
        r'\g<i>    if cam.is_connected:\n'
        r'\g<i>        camera_claimed = cam.begin_exclusive_capture(\n'
        r'\g<i>            "EXTERNAL" if use_hw_trigger else "INTERNAL",\n'
        r'\g<i>            restart=use_hw_trigger,\n'
        r'\g<i>        )\n'
        r'\g<i>        if not camera_claimed:\n'
        r'\g<i>            raise RuntimeError("다른 작업이 카메라를 사용 중입니다.")\n'
        r'\g<i>        if use_hw_trigger:\n'
        r'\g<i>            cam.grab_frame(timeout_ms=20)\n'
        r'\g<i>    if use_hw_trigger and cam.is_connected and stage.is_connected:\n'
        r"\g<i>        stage.set_trigger_out(axis=fast_ax, value='0.0')\n"
    )
    block = replace_regex_once(
        block, mapping_start_pattern, mapping_start_replacement,
        "MappingWorker exclusive camera ownership", changes,
    )


    block = replace_regex_once(
        block,
        r'(?P<i>^[ \t]*)if use_hw_trigger and cam\.is_connected:\n'
        r'(?P=i)    cam\.stop_capture\(\)\n'
        r'(?P=i)    cam\.set_trigger_mode\("INTERNAL"\)\n',
        r'\g<i>if cam.is_connected and camera_claimed:\n'
        r'\g<i>    cam.end_exclusive_capture(resume_live=True)\n',
        "MappingWorker camera cleanup", changes,
    )


    # Add an outer error log if this worker has only try/finally.
    if "MappingWorker failed" not in block:
        block = replace_once(
            block,
            "        finally:\n",
            "        except Exception:\n"
            "            logging.exception(\"MappingWorker failed\")\n"
            "        finally:\n",
            "MappingWorker uncaught exception logging",
            changes,
        )
    return block


def patch_homoepi_worker(block: str, changes: list[str]) -> str:
    if "self._dz_step" not in block:
        marker = "        self.is_running = True\n"
        injection = marker + (
            "        self._use_hw_trigger = bool(main_window.control_panel.chk_hw_trigger.isChecked())\n"
            "        self._original_speed = float(main_window.control_panel.spin_speed.value())\n"
            "        try:\n"
            "            self._exposure_time = float(main_window.control_panel.exposure_input.text())\n"
            "        except Exception:\n"
            "            self._exposure_time = 0.01\n"
            "        self._waves = np.array(main_window.spectrum_view.x_axis, copy=True)\n"
            "        self._dz_step = float(main_window.homoepi_view.sp_z_step.value())\n"
        )
        block = replace_once(
            block,
            marker,
            injection,
            "HomoepiWorker UI snapshot in constructor",
            changes,
        )

    homo_replacements = {
        "self.main_window.control_panel.chk_hw_trigger.isChecked()": "self._use_hw_trigger",
        "self.main_window.control_panel.spin_speed.value()": "self._original_speed",
        "self.main_window.spectrum_view.x_axis": "self._waves",
        "self.main_window.homoepi_view.sp_z_step.value()": "self._dz_step",
    }
    for old, new in homo_replacements.items():
        if old in block:
            block = block.replace(old, new)
            changes.append(f"APPLY: HomoepiWorker expression {old}")
        else:
            changes.append(f"SKIP(not found): HomoepiWorker expression {old}")

    block = replace_regex_once(
        block,
        r'(?P<i>^[ \t]*)try:\n'
        r'(?P=i)    exposure_time = float\(self\.main_window\.control_panel\.exposure_input\.text\(\)\)\n'
        r'(?P=i)except(?: Exception)?:\n'
        r'(?P=i)    exposure_time = 0\.01\n',
        r'\g<i>exposure_time = self._exposure_time\n',
        "HomoepiWorker exposure snapshot", changes,
    )


    block = replace_regex_once(
        block,
        r'(?P<i>^[ \t]*)if use_hw_trigger and cam\.is_connected and stage\.is_connected:\n'
        r'(?P=i)    cam\.stop_capture\(\)\n'
        r'(?P=i)    cam\.set_trigger_mode\("EXTERNAL"\)\n'
        r'(?P=i)    cam\.start_capture\(\)\n'
        r'(?P=i)    cam\.grab_frame\(\)\n'
        r'(?P=i)    stage\.set_trigger_out\(axis=\'z\', value=\'0\.0\'\)\n',
        r'\g<i>camera_claimed = False\n'
        r'\g<i>if cam.is_connected:\n'
        r'\g<i>    camera_claimed = cam.begin_exclusive_capture(\n'
        r'\g<i>        "EXTERNAL" if use_hw_trigger else "INTERNAL",\n'
        r'\g<i>        restart=use_hw_trigger,\n'
        r'\g<i>    )\n'
        r'\g<i>    if not camera_claimed:\n'
        r'\g<i>        raise RuntimeError("다른 작업이 카메라를 사용 중입니다.")\n'
        r'\g<i>    if use_hw_trigger:\n'
        r'\g<i>        cam.grab_frame(timeout_ms=20)\n'
        r'\g<i>if use_hw_trigger and cam.is_connected and stage.is_connected:\n'
        r"\g<i>    stage.set_trigger_out(axis='z', value='0.0')\n",
        "HomoepiWorker exclusive camera ownership", changes,
    )


    block = replace_regex_once(
        block,
        r'(?P<i>^[ \t]*)if use_hw_trigger and cam\.is_connected:\n'
        r'(?P=i)    cam\.stop_capture\(\)\n'
        r'(?P=i)    cam\.set_trigger_mode\("INTERNAL"\)\n',
        r'\g<i>if cam.is_connected and camera_claimed:\n'
        r'\g<i>    cam.end_exclusive_capture(resume_live=True)\n',
        "HomoepiWorker camera cleanup", changes,
    )

    block = replace_once(
        block,
        '            raw_data = {"z": z_arr, "spectra": spectra_arr, "waves": self.waves}\n',
        '            raw_data = {\n'
        '                "z": z_arr.astype(np.float32, copy=False),\n'
        '                "spectra": spectra_arr.astype(np.float32, copy=False),\n'
        '                "waves": self.waves,\n'
        '            }\n',
        "Homoepi raw payload float32",
        changes,
    )
    return block


def insert_close_event(text: str, changes: list[str]) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise PatchError(f"main.py is invalid before closeEvent patch: {exc}") from exc

    target = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            base_names = {
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                for base in node.bases
            }
            if node.name == "MainWindow" or "QMainWindow" in base_names:
                target = node
                break
    if target is None:
        changes.append("SKIP(not found): QMainWindow closeEvent")
        return text
    if any(isinstance(item, ast.FunctionDef) and item.name == "closeEvent" for item in target.body):
        changes.append("SKIP(already): QMainWindow closeEvent")
        return text

    method = (
        "\n    def closeEvent(self, event):\n"
        "        shutdown_hardware(self)\n"
        "        event.accept()\n"
    )
    lines = text.splitlines(keepends=True)
    # AST line numbers are 1-based and end_lineno is inclusive. Inserting at
    # this zero-based index places the method after the existing class body and
    # before the next top-level statement.
    lines.insert(target.end_lineno, method)
    changes.append("APPLY: QMainWindow ordered hardware shutdown")
    return "".join(lines)


def patch_main(source: str) -> tuple[str, list[str]]:
    marker = "# RAMAN_STABILITY_PATCH_V1"
    if marker in source:
        ast.parse(source)
        return source, ["SKIP(already): main.py stability patch V1"]

    changes: list[str] = []
    text = source

    if "import logging\n" not in text:
        text = replace_once(
            text,
            "import sys\n",
            "import sys\nimport logging\n",
            "logging import",
            changes,
            required=True,
        )

    stability_import = (
        marker + "\n"
        "from stability_utils import (\n"
        "    allocate_spectrum_cube,\n"
        "    install_global_exception_logging,\n"
        "    safe_eval_formula,\n"
        "    shutdown_hardware,\n"
        "    snapshot_temperature_pairs,\n"
        ")\n"
        "install_global_exception_logging()\n"
    )
    if "from stability_utils import" not in text:
        text = replace_once(
            text,
            "from raman_ml import RamanMLProcessor\n",
            "from raman_ml import RamanMLProcessor\n" + stability_import,
            "stability helper imports",
            changes,
            required=True,
        )

    # Replace unsafe eval while preserving the class's public API.
    text = replace_regex_once(
        text,
        r"    def evaluate\(self, ch_values\):\n"
        r"        try:\n"
        r"            expr = self\.expression\n"
        r"            for ch_name in sorted\(ch_values\.keys\(\), key=len, reverse=True\):\n"
        r"                val = ch_values\[ch_name\]\n"
        r"                if val is None or np\.isnan\(val\): return np\.nan\n"
        r"                expr = expr\.replace\(ch_name, str\(val\)\)\n"
        r"            return float\(eval\(expr, \{\"__builtins__\": None\}, \{\}\)\)\n"
        r"        except Exception:\n"
        r"            return np\.nan\n",
        "    def evaluate(self, ch_values):\n"
        "        try:\n"
        "            return safe_eval_formula(self.expression, ch_values)\n"
        "        except Exception:\n"
        "            logging.exception(\"Custom formula evaluation failed: %s\", self.expression)\n"
        "            return np.nan\n",
        "safe formula evaluator",
        changes,
    )

    # Live view: prevent re-entry, short nonblocking camera waits, and reduce
    # heavy full-frame auto-level/spectrum processing to 10 Hz.
    text = replace_once(
        text,
        "        self.current_frame = None\n",
        "        self.current_frame = None\n"
        "        self._frame_update_busy = False\n"
        "        self._levels_initialized = False\n"
        "        self._last_spectrum_update = 0.0\n"
        "        self._spectrum_update_period = 0.10\n",
        "LiveView state guards",
        changes,
    )
    text = replace_once(
        text,
        "        self.timer.start(50)\n",
        "        self.timer.start(100)\n",
        "LiveView update interval 50ms -> 100ms",
        changes,
    )
    text = replace_regex_once(
        text,
        r"    def update_frame\(self\):\n.*?(?=    def save_data\(self\):)",
        "    def update_frame(self):\n"
        "        if self._frame_update_busy:\n"
        "            return\n"
        "        self._frame_update_busy = True\n"
        "        try:\n"
        "            frame = (\n"
        "                self.cam.grab_frame(timeout_ms=25, nonblocking=True)\n"
        "                if self.cam.is_connected else None\n"
        "            )\n"
        "            if frame is None and not self.cam.is_connected:\n"
        "                frame = np.random.randint(0, 40, (480, 640), dtype=np.uint8)\n"
        "            if frame is None:\n"
        "                return\n"
        "            self.current_frame = frame\n"
        "            self.img_item.setImage(\n"
        "                frame, autoLevels=not self._levels_initialized\n"
        "            )\n"
        "            self._levels_initialized = True\n"
        "            now = time.monotonic()\n"
        "            if now - self._last_spectrum_update >= self._spectrum_update_period:\n"
        "                spectrum_1d = np.sum(frame, axis=0, dtype=np.float64)\n"
        "                self.main_window.spectrum_view.process_spectrum(spectrum_1d)\n"
        "                self._last_spectrum_update = now\n"
        "        except Exception:\n"
        "            logging.exception(\"LiveView frame update failed\")\n"
        "        finally:\n"
        "            self._frame_update_busy = False\n\n",
        "LiveView nonblocking/re-entry-safe update",
        changes,
        required=True,
    )

    text = replace_once(
        text,
        "        self.rolling_buffer.append(raw_1d)\n",
        "        self.rolling_buffer.append(np.asarray(raw_1d, dtype=np.float32))\n",
        "rolling spectrum buffer float32",
        changes,
    )
    text = replace_regex_once(
        text,
        r'^(\s*)self\.map_layers\["RAW_SPECTRA"\] = np\.full\(shape_3d, np\.nan\)\s*$',
        r'\1self.map_layers["RAW_SPECTRA"] = allocate_spectrum_cube(\n'
        r'\1    shape_3d, dtype=np.float32\n'
        r'\1)',
        "bounded hyperspectral cube allocation",
        changes,
        required=True,
    )

    text = transform_class_block(text, "MappingWorker", patch_mapping_worker, changes)
    text = transform_class_block(text, "HomoepiWorker", patch_homoepi_worker, changes)
    text = insert_close_event(text, changes)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise PatchError(f"Patched main.py failed syntax validation: {exc}") from exc
    return text, changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default=".", help="repository directory")
    parser.add_argument("--dry-run", action="store_true", help="validate without writing")
    parser.add_argument(
        "--no-main-patch",
        action="store_true",
        help="only replace controller/helper modules",
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    bundle = Path(__file__).resolve().parent
    main_path = repo / "main.py"
    if not repo.is_dir() or not main_path.exists():
        print(f"ERROR: {repo} does not look like the Raman-Spectroscopy repository", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    changes: list[str] = []

    try:
        patched_main = None
        if not args.no_main_patch:
            source = main_path.read_text(encoding="utf-8")
            patched_main, main_changes = patch_main(source)
            changes.extend(main_changes)

        for filename in MODULES:
            source_path = bundle / filename
            destination = repo / filename
            if not source_path.exists():
                raise PatchError(f"Bundle file missing: {source_path}")
            if destination.exists():
                destination_backup = backup(destination, stamp, args.dry_run)
                changes.append(f"BACKUP: {destination.name} -> {destination_backup.name}")
            if not args.dry_run:
                shutil.copy2(source_path, destination)
            changes.append(f"COPY: {filename}")

        if patched_main is not None:
            main_backup = backup(main_path, stamp, args.dry_run)
            changes.append(f"BACKUP: main.py -> {main_backup.name}")
            if not args.dry_run:
                main_path.write_text(patched_main, encoding="utf-8", newline="\n")
            changes.append("WRITE: main.py")

        mode = "DRY RUN OK" if args.dry_run else "PATCH COMPLETE"
        print(f"\n{mode}: {repo}\n")
        for item in changes:
            print(f" - {item}")
        print("\nRun: python -m py_compile main.py camera_controller.py stage_controller.py raman_ml.py stability_utils.py")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("No patched main.py was written. Existing backups, if any, were kept.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
