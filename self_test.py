#!/usr/bin/env python3
"""Hardware-free smoke tests for the stability patch."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


def test_utils(bundle: Path) -> None:
    sys.path.insert(0, str(bundle))
    from stability_utils import allocate_spectrum_cube, safe_eval_formula

    result = safe_eval_formula("A / B + sqrt(C)", {"A": 8.0, "B": 2.0, "C": 9.0})
    assert abs(result - 7.0) < 1e-9

    small = allocate_spectrum_cube((2, 3, 4), max_ram_mb=64)
    assert small.dtype == np.float32 and small.shape == (2, 3, 4)

    with tempfile.TemporaryDirectory() as directory:
        mapped = allocate_spectrum_cube(
            (256, 256, 300),
            max_ram_mb=64,
            cache_dir=directory,
        )
        assert isinstance(mapped, np.memmap)
        mapped[0, 0, 0] = 1.0
        mapped.flush()


def test_stage(bundle: Path) -> None:
    sys.path.insert(0, str(bundle))
    from stage_controller import PiezoController

    stage = PiezoController(simulate=True)
    assert stage.connect()
    assert stage.move_to_logical(1.0, 2.0, 3.0, wait_ack=False)
    assert stage.get_position() == (1.0, 2.0, 3.0)
    assert stage.stop_motion()
    stage.disconnect()


def test_ml(bundle: Path) -> None:
    import torch

    sys.path.insert(0, str(bundle))
    from raman_ml import RamanMLProcessor, RamanSRNetV21

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.pth"
        model = RamanSRNetV21(num_features=8, num_blocks=2)
        torch.save(model.state_dict(), model_path)
        processor = RamanMLProcessor(str(model_path))
        spectrum = np.linspace(0.0, 100.0, 128, dtype=np.float32)
        output = processor.enhance_spectrum(spectrum)
        assert output.shape == spectrum.shape
        assert np.isfinite(output).all()


def test_camera(bundle: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        fake = Path(directory)
        (fake / "dcamcon.py").write_text("", encoding="utf-8")
        (fake / "dcam.py").write_text(
            '''\nimport time\nimport numpy as np\nclass DCAM_IDPROP:\n    EXPOSURETIME=1; BINNING=2; SUBARRAYMODE=3; SUBARRAYVSIZE=4; SUBARRAYVPOS=5\n    SENSORCOOLER=6; SENSORCOOLERFAN=7; SENSORTEMPERATURE=8\n    TRIGGERSOURCE=9; TRIGGERPOLARITY=10; TRIGGERACTIVE=11\nclass Dcamapi:\n    @staticmethod\n    def init(): return True\n    @staticmethod\n    def uninit(): return True\nclass Dcam:\n    def __init__(self, index): self.props={DCAM_IDPROP.TRIGGERSOURCE:1.0}\n    def dev_open(self): return True\n    def dev_close(self): return True\n    def buf_alloc(self, n): return True\n    def buf_release(self): return True\n    def cap_start(self): return True\n    def cap_stop(self): return True\n    def wait_capevent_frameready(self, timeout): time.sleep(min(timeout/1000,0.005)); return True\n    def buf_getlastframedata(self): return np.ones((4,8),dtype=np.uint16)\n    def prop_setvalue(self, key, value): self.props[key]=value; return True\n    def prop_getvalue(self, key): return self.props.get(key,-20.0)\n''',
            encoding="utf-8",
        )
        sys.path.insert(0, str(fake))
        spec = importlib.util.spec_from_file_location(
            "camera_controller_test", bundle / "camera_controller.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        camera = module.CameraController()
        assert camera.initialize_dcam()
        assert camera.connect_first_available_camera()
        assert camera.start_capture()
        assert camera.grab_frame(timeout_ms=10).shape == (4, 8)

        ready = threading.Event()
        release = threading.Event()

        def worker():
            assert camera.begin_exclusive_capture("INTERNAL")
            ready.set()
            release.wait(1.0)
            camera.end_exclusive_capture()

        thread = threading.Thread(target=worker)
        thread.start()
        assert ready.wait(1.0)
        assert camera.grab_frame(timeout_ms=5, nonblocking=True) is None
        release.set()
        thread.join(1.0)
        assert not thread.is_alive()
        camera.disconnect()
        camera.uninitialize_dcam()



def test_patcher(bundle: Path) -> None:
    sys.path.insert(0, str(bundle))
    from apply_stability_patch import patch_main

    source = '''import sys
import time
import numpy as np
from raman_ml import RamanMLProcessor

class CustomFormula:
    def evaluate(self, ch_values):
        try:
            expr = self.expression
            for ch_name in sorted(ch_values.keys(), key=len, reverse=True):
                val = ch_values[ch_name]
                if val is None or np.isnan(val): return np.nan
                expr = expr.replace(ch_name, str(val))
            return float(eval(expr, {"__builtins__": None}, {}))
        except Exception:
            return np.nan

class LiveView:
    def __init__(self):
        self.current_frame = None
        self.timer.start(50)

    def update_frame(self):
        frame = self.cam.grab_frame()
        if frame is not None:
            self.current_frame = frame
            self.img_item.setImage(frame, autoLevels=True)
            spectrum_1d = np.sum(frame, axis=0)
            self.main_window.spectrum_view.process_spectrum(spectrum_1d)

    def save_data(self):
        pass

class MappingWorker:
    def __init__(self, main_window):
        self.main_window = main_window
        self.is_running = True

    def run(self):
        cam = self.main_window.cam
        stage = self.main_window.stage
        x_axis = self.main_window.spectrum_view.x_axis
        active_pairs = self.main_window.control_panel.active_pairs
        virtual_channels = self.main_window.mapping_view.virtual_channels
        formulas = self.main_window.mapping_view.formulas
        use_hw_trigger = self.main_window.control_panel.chk_hw_trigger.isChecked()
        original_speed = self.main_window.control_panel.spin_speed.value()
        fast_ax = 'x'
        f_csv = None
        try:
            if use_hw_trigger and cam.is_connected and stage.is_connected:
                cam.stop_capture()
                cam.set_trigger_mode("EXTERNAL")
                cam.start_capture()
                cam.grab_frame()
                stage.set_trigger_out(axis=fast_ax, value='0.0')
            shape_3d = (2, 2, len(x_axis))
            self.map_layers["RAW_SPECTRA"] = np.full(shape_3d, np.nan)
            for pair in active_pairs:
                pair.update_temperature(x_axis, np.ones(len(x_axis)), is_mapping=True)
        finally:
            if f_csv is not None: f_csv.close()
            if stage.is_connected:
                stage.set_speed(original_speed)
            if use_hw_trigger and cam.is_connected:
                cam.stop_capture()
                cam.set_trigger_mode("INTERNAL")

class HomoepiWorker:
    def __init__(self, main_window):
        self.main_window = main_window
        self.is_running = True

    def run(self):
        use_hw_trigger = self.main_window.control_panel.chk_hw_trigger.isChecked()
        original_speed = self.main_window.control_panel.spin_speed.value()
        self.waves = self.main_window.spectrum_view.x_axis

class MainWindow(QMainWindow):
    pass
'''
    patched, changes = patch_main(source)
    compile(patched, "patched_main.py", "exec")
    assert "RAMAN_STABILITY_PATCH_V1" in patched
    assert "timeout_ms=25" in patched
    assert "allocate_spectrum_cube" in patched
    assert "def closeEvent" in patched
    assert any(item.startswith("APPLY") for item in changes)

    patched_again, changes_again = patch_main(patched)
    assert patched_again == patched
    assert changes_again == ["SKIP(already): main.py stability patch V1"]


def main() -> None:
    bundle = Path(__file__).resolve().parent
    test_utils(bundle)
    test_stage(bundle)
    test_ml(bundle)
    test_camera(bundle)
    test_patcher(bundle)
    print("All hardware-free stability patch tests passed.")


if __name__ == "__main__":
    main()
