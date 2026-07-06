import sys
import os
import serial.tools.list_ports
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTabWidget,
                             QGroupBox, QFormLayout, QLineEdit, QComboBox,
                             QRadioButton, QGridLayout, QScrollArea, QDoubleSpinBox,
                             QSpinBox, QSplitter, QCheckBox, QMessageBox, QListWidget,
                             QProgressBar, QListWidgetItem, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QColor, QBrush
from PyQt6.QtCore import QThread, pyqtSignal
import pyqtgraph.exporters  # 이미지 저장을 위해 필수
from PyQt6.QtWidgets import QFileDialog
import time
import csv
from PyQt6.QtWidgets import QFileDialog
# 커스텀 모듈
from camera_controller import CameraController
from stage_controller import PiezoController
from raman_math import remove_als_baseline, calculate_temperature, apply_gaussian_1d, calibrate_raman_axis_quadratic
from raman_ml import RamanMLProcessor
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import curve_fit

# 피팅을 위한 Sigmoid 함수 정의 (클래스 밖에 전역 함수로 추가하세요)
def sigmoid(z, A, B, z0, k):
    return A + (B - A) / (1 + np.exp(-(z - z0) / (np.abs(k) + 1e-3)))

def interface_sigmoid(z, bg, amp, z_intf, w):
    """
    샘플 내부에서 에피층(Epi) -> 기판(Sub)으로
    피크 위치가 전이되는 과정을 모델링
    """
    return bg + amp / (1 + np.exp((z - z_intf) / (np.abs(w) + 1e-3)))


# -------------------------------------------------------------------
# [신규] 가상 채널 데이터 클래스
# -------------------------------------------------------------------
class VirtualChannel:
    def __init__(self, name, mode="integrate", center=450, width=15):
        self.name = name
        self.mode = mode
        self.center = center
        self.width = width

    def get_intensity(self, spectrum, x_axis):
        mask = (x_axis >= self.center - self.width / 2) & (x_axis <= self.center + self.width / 2)
        if not np.any(mask):
            return np.nan

        region_data = spectrum[mask]
        region_x = x_axis[mask] # X축 데이터 필요

        if self.mode == "integrate":
            return float(np.sum(region_data))
        elif self.mode == "max":
            return float(np.max(region_data))
        elif self.mode == "mean":
            return float(np.mean(region_data))
        elif self.mode == "position":
            idx = np.argmax(region_data)
            return float(region_x[idx])
        elif self.mode == "com": # 신규: 무게중심(Center of Mass)
            bg_subtracted = region_data - np.min(region_data)
            sum_val = np.sum(bg_subtracted)
            if sum_val <= 0: return np.nan
            return float(np.sum(bg_subtracted * region_x) / sum_val)
        else:
            return float(np.sum(region_data))


class CustomFormula:
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression

    # 🚨 [추가된 핵심 기능] Ch1, Ch2 등의 실제 값을 받아 사용자의 수식(1ch-2ch 등)을 연산
    def evaluate(self, ch_values):
        try:
            expr = self.expression
            # 이름이 긴 채널(예: Ch10)부터 치환해야 Ch1이 꼬이는 문자열 버그를 방지
            for ch_name in sorted(ch_values.keys(), key=len, reverse=True):
                val = ch_values[ch_name]
                # 채널 값이 비어있으면 수식도 에러이므로 NaN 반환
                if val is None or np.isnan(val):
                    return np.nan

                # 수식(예: "Ch1 - Ch2") 글자를 실제 측정된 숫자(예: "150.5 - 20.1")로 바꿈
                expr = expr.replace(ch_name, str(val))

            # 파이썬 내장 연산기를 사용해 문자열 수학식을 실제 결과값으로 변환
            return float(eval(expr, {"__builtins__": None}, {}))

        except Exception:
            # 수식에 오타가 있거나 0으로 나누는 등 연산 불가 상태일 때 뻗지 않도록 예외 처리
            return np.nan

    # -------------------------------------------------------------------


# 1. 동적 온도 측정 쌍 클래스 (UI 수치 조절 기능 추가)
# -------------------------------------------------------------------
class TempPair:
    def __init__(self, pair_id, main_window, ui_parent_layout, color_hex='#00e5ff'):
        self.pair_id = pair_id
        self.main_window = main_window
        self.color_hex = color_hex

        # 1. 왼쪽 패널용 라벨
        self.lbl_temp = QLabel(f"Pair {self.pair_id}: -- ℃")
        self.lbl_temp.setStyleSheet(f"""
            QLabel {{
                background-color: #2b2b2b; 
                color: {self.color_hex}; 
                font-weight: bold; 
                font-size: 14px;
                padding: 4px;
                border-radius: 4px;
                border: 1px solid #555555;
            }}
        """)

        # 2. 반투명 영역 바 생성 (Stokes, Anti-Stokes, 그리고 BG 노이즈 영역)
        anti_color = QColor(self.color_hex);
        anti_color.setAlpha(80)
        brush_anti = QBrush(anti_color, Qt.BrushStyle.BDiagPattern)
        pen_anti = pg.mkPen(color=self.color_hex, width=2, style=Qt.PenStyle.DashLine)

        stokes_color = QColor(self.color_hex);
        stokes_color.setAlpha(50)
        brush_stokes = QBrush(stokes_color, Qt.BrushStyle.SolidPattern)
        pen_stokes = pg.mkPen(color=self.color_hex, width=2, style=Qt.PenStyle.SolidLine)

        # [신규] 배경 노이즈(BG) 영역 설정: 점선 테두리 + 성긴 점 무늬
        bg_color = QColor(self.color_hex);
        bg_color.setAlpha(30)
        brush_bg = QBrush(bg_color, Qt.BrushStyle.FDiagPattern)
        pen_bg = pg.mkPen(color=self.color_hex, width=1, style=Qt.PenStyle.DotLine)

        self.region_anti = pg.LinearRegionItem(values=[-530, -510], brush=brush_anti, pen=pen_anti)
        self.region_stokes = pg.LinearRegionItem(values=[510, 530], brush=brush_stokes, pen=pen_stokes)
        self.region_bg = pg.LinearRegionItem(values=[0, 50], brush=brush_bg, pen=pen_bg)  # BG 영역 초기값

        self.main_window.spectrum_view.plot_widget.addItem(self.region_anti)
        self.main_window.spectrum_view.plot_widget.addItem(self.region_stokes)
        self.main_window.spectrum_view.plot_widget.addItem(self.region_bg)

        # 3. 스펙트럼 상단 수치 정밀 제어 패널 생성
        self.control_widget = QWidget()
        layout = QHBoxLayout(self.control_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl_title = QLabel(f"<b style='color:{self.color_hex};'>■ P{self.pair_id}</b>")
        layout.addWidget(lbl_title)

        self.btn_delete = QPushButton("❌")
        self.btn_delete.setFixedWidth(30)
        self.btn_delete.clicked.connect(self.delete_self)
        layout.addWidget(self.btn_delete)

        layout.addWidget(QLabel("Anti:"))
        self.spin_anti_c = QDoubleSpinBox();
        self.spin_anti_c.setRange(-4000, 4000);
        self.spin_anti_c.setValue(-520.0)
        layout.addWidget(self.spin_anti_c)
        self.spin_anti_w = QDoubleSpinBox();
        self.spin_anti_w.setRange(1, 1000);
        self.spin_anti_w.setValue(30.0)
        layout.addWidget(self.spin_anti_w)

        layout.addWidget(QLabel("| Stokes:"))
        self.spin_stokes_c = QDoubleSpinBox();
        self.spin_stokes_c.setRange(-4000, 4000);
        self.spin_stokes_c.setValue(520.0)
        layout.addWidget(self.spin_stokes_c)
        self.spin_stokes_w = QDoubleSpinBox();
        self.spin_stokes_w.setRange(1, 1000);
        self.spin_stokes_w.setValue(30.0)
        layout.addWidget(self.spin_stokes_w)

        # [신규] 노이즈(BG) 조절 스핀박스
        layout.addWidget(QLabel("| BG(Noise):"))
        self.spin_bg_c = QDoubleSpinBox();
        self.spin_bg_c.setRange(-4000, 4000);
        self.spin_bg_c.setValue(0)
        layout.addWidget(self.spin_bg_c)
        self.spin_bg_w = QDoubleSpinBox();
        self.spin_bg_w.setRange(1, 1000);
        self.spin_bg_w.setValue(50.0)
        layout.addWidget(self.spin_bg_w)

        layout.addWidget(QLabel("| C:"))
        self.spin_c_factor = QDoubleSpinBox();
        self.spin_c_factor.setRange(0.001, 100.0)
        self.spin_c_factor.setDecimals(3);
        self.spin_c_factor.setSingleStep(0.1);
        self.spin_c_factor.setValue(1.000)
        layout.addWidget(self.spin_c_factor)

        ui_parent_layout.addWidget(self.control_widget)

        # 4. 시그널 동기화 (BG 영역 추가)
        self.region_anti.sigRegionChanged.connect(self.update_spinboxes_from_region)
        self.region_stokes.sigRegionChanged.connect(self.update_spinboxes_from_region)
        self.region_bg.sigRegionChanged.connect(self.update_spinboxes_from_region)

        self.spin_anti_c.valueChanged.connect(self.update_region_from_spinboxes)
        self.spin_anti_w.valueChanged.connect(self.update_region_from_spinboxes)
        self.spin_stokes_c.valueChanged.connect(self.update_region_from_spinboxes)
        self.spin_stokes_w.valueChanged.connect(self.update_region_from_spinboxes)
        self.spin_bg_c.valueChanged.connect(self.update_region_from_spinboxes)
        self.spin_bg_w.valueChanged.connect(self.update_region_from_spinboxes)
        self._updating = False

    def update_spinboxes_from_region(self):
        if self._updating: return
        self._updating = True

        anti_min, anti_max = self.region_anti.getRegion()
        self.spin_anti_c.setValue((anti_max + anti_min) / 2)
        self.spin_anti_w.setValue(anti_max - anti_min)

        stokes_min, stokes_max = self.region_stokes.getRegion()
        self.spin_stokes_c.setValue((stokes_max + stokes_min) / 2)
        self.spin_stokes_w.setValue(stokes_max - stokes_min)

        bg_min, bg_max = self.region_bg.getRegion()
        self.spin_bg_c.setValue((bg_max + bg_min) / 2)
        self.spin_bg_w.setValue(bg_max - bg_min)

        self._updating = False

    def update_region_from_spinboxes(self):
        if self._updating: return
        self._updating = True

        ac, aw = self.spin_anti_c.value(), self.spin_anti_w.value()
        self.region_anti.setRegion([ac - aw / 2, ac + aw / 2])

        sc, sw = self.spin_stokes_c.value(), self.spin_stokes_w.value()
        self.region_stokes.setRegion([sc - sw / 2, sc + sw / 2])

        bc, bw = self.spin_bg_c.value(), self.spin_bg_w.value()
        self.region_bg.setRegion([bc - bw / 2, bc + bw / 2])

        self._updating = False

    def update_temperature(self, x_axis, spectrum, is_mapping=False):
        anti_min, anti_max = self.region_anti.getRegion()
        stokes_min, stokes_max = self.region_stokes.getRegion()
        bg_min, bg_max = self.region_bg.getRegion()

        anti_mask = (x_axis >= anti_min) & (x_axis <= anti_max)
        stokes_mask = (x_axis >= stokes_min) & (x_axis <= stokes_max)
        bg_mask = (x_axis >= bg_min) & (x_axis <= bg_max)

        if np.any(anti_mask) and np.any(stokes_mask) and np.any(bg_mask):
            bg_baseline = np.median(spectrum[bg_mask])

            anti_net = np.mean(spectrum[anti_mask]) - bg_baseline
            stokes_net = np.mean(spectrum[stokes_mask]) - bg_baseline

            if anti_net > 0 and stokes_net > 0:
                try:
                    laser_wl_nm = self.main_window.control_panel.spin_laser.value()
                except:
                    laser_wl_nm = 532.0

                v0 = 1e7 / laser_wl_nm
                vm = abs(self.spin_stokes_c.value())

                ratio = anti_net / stokes_net
                C = self.spin_c_factor.value()

                K = ((v0 + vm) / (v0 - vm)) ** 4
                ln_term = (C * K) / ratio

                if ln_term > 1.0:
                    temp_K = (1.43877 * vm) / np.log(ln_term)
                    temp_C = temp_K - 273.15

                    # 🚨 [스위치 작동] 맵핑 중이 아닐 때(라이브뷰)만 UI 글자를 바꿈
                    if not is_mapping:
                        self.lbl_temp.setText(f"Pair {self.pair_id}: {temp_C:.1f} ℃")

                    return temp_C
                else:
                    if not is_mapping:
                        self.lbl_temp.setText(f"Pair {self.pair_id}: 에러 (Ratio 초과)")
                    return None

        if not is_mapping:
            self.lbl_temp.setText(f"Pair {self.pair_id}: -- ℃")
        return None

    def delete_self(self):
        """UI 위젯과 그래프 영역을 화면에서 지우고, 메인 리스트에서 자신을 제거합니다."""
        # 1. 1D 스펙트럼 그래프에서 배경 영역 지우기
        try:
            self.main_window.spectrum_view.plot_widget.removeItem(self.region_anti)
            self.main_window.spectrum_view.plot_widget.removeItem(self.region_stokes)
            self.main_window.spectrum_view.plot_widget.removeItem(self.region_bg)
        except:
            pass

        # 2. UI 위젯 지우기 (왼쪽 온도 텍스트 라벨과 스핀박스 패널)
        self.lbl_temp.deleteLater()
        self.control_widget.deleteLater()

        # 3. 메인 윈도우의 '활성화된 온도 쌍 리스트'에서 자신을 빼기
        if self in self.main_window.control_panel.active_pairs:
            self.main_window.control_panel.active_pairs.remove(self)


class MappingWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    # 🚨 [신규] 화면을 매 픽셀마다 그리지 않고, 한 줄(Row)이 끝날 때마다 그리도록 최적화
    row_finished_signal = pyqtSignal(int)

    def __init__(self, main_window, scan_params, map_layers_ref, exposure_time=0.01):
        super().__init__()
        self.main_window = main_window
        self.params = scan_params
        self.map_layers = map_layers_ref
        self.exposure_time = exposure_time
        self.is_running = True

    def run(self):
        cam = self.main_window.cam
        stage = self.main_window.stage

        x_axis = self.main_window.spectrum_view.x_axis
        active_pairs = self.main_window.control_panel.active_pairs
        virtual_channels = self.main_window.mapping_view.virtual_channels
        formulas = self.main_window.mapping_view.formulas

        map_type = self.params['map_type']

        x_points = np.arange(self.params['x_start'], self.params['x_end'] + self.params['x_step'] / 2,
                             self.params['x_step'])
        y_points = np.arange(self.params['y_start'], self.params['y_end'] + self.params['y_step'] / 2,
                             self.params['y_step'])
        z_points = np.arange(self.params['z_start'], self.params['z_end'] + self.params['z_step'] / 2,
                             self.params['z_step'])

        if "Z 1D" in map_type:
            x_points, y_points = [0.0], [0.0]
        elif "XY 2D" in map_type:
            z_points = [self.params['z_start']]

        current_pt = 0
        cur_x, cur_y, cur_z = 0.0, 0.0, 0.0

        save_full = self.params.get('save_full', False)
        f_csv = None
        writer = None
        if save_full:
            file_name = f"Raw_Hyperspectral_Data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            f_csv = open(file_name, 'w', newline='')
            writer = csv.writer(f_csv)
            writer.writerow(['X(um)', 'Y(um)', 'Z(um)'] + list(x_axis))

        try:
            for zi, z_val in enumerate(z_points):
                if not self.is_running: break
                for yi, y_val in enumerate(y_points):
                    if not self.is_running: break
                    x_iter = enumerate(x_points) if yi % 2 == 0 else reversed(list(enumerate(x_points)))

                    for xi, x_val in x_iter:
                        if not self.is_running: break

                        if stage.is_connected:
                            # 🚨 wait_ack=False로 고속 전송
                            stage.move_to_logical(x_val, y_val, z_val, wait_ack=False)
                            # 🚨 이동 직후 10ms 대기 (측정/안정화)
                            time.sleep(0.005)

                        cur_x, cur_y, cur_z = x_val, y_val, z_val

                        frame = None
                        if cam.is_connected:
                            # 🚨 [수정됨] 고정된 200회가 아닌, 노출 시간(exposure_time)에 맞춰 유동적으로 기다립니다.
                            # 카메라 응답 지연을 대비해 설정된 노출시간 + 2초 분량만큼 넉넉히 대기합니다.
                            # 루프 1회당 약 0.015~0.02초가 소요됨을 감안하여 최대 반복 횟수를 계산합니다.
                            timeout_limit = self.exposure_time + 2.0
                            max_retries = int(timeout_limit / 0.015)

                            for _ in range(max_retries):
                                temp_frame = cam.grab_frame()
                                if temp_frame is not None and hasattr(temp_frame, 'ndim'):
                                    frame = temp_frame
                                    break
                                time.sleep(0.01)
                        else:
                            time.sleep(self.exposure_time)
                            frame = np.random.randint(0, 40, (480, 640), dtype=np.uint8)

                        spectrum_1d = np.full(len(x_axis), np.nan)

                        if frame is not None and hasattr(frame, 'ndim'):
                            if frame.ndim == 2:
                                raw_1d = np.sum(frame, axis=0)
                            else:
                                raw_1d = frame

                            min_len = min(len(raw_1d), len(x_axis))
                            spectrum_1d[:min_len] = raw_1d[:min_len]

                        if save_full and writer is not None:
                            writer.writerow([cur_x, cur_y, cur_z] + list(spectrum_1d))

                        # 🚨 RAW_SPECTRA 데이터 메모리 할당 (Z 1D 스캔 형태 완벽 지원)
                        if save_full:
                            if "RAW_SPECTRA" not in self.map_layers:
                                if "Z 1D" in map_type:
                                    shape_3d = (1, len(z_points), len(x_axis))
                                else:
                                    shape_3d = (len(y_points), len(x_points), len(x_axis))
                                self.map_layers["RAW_SPECTRA"] = np.full(shape_3d, np.nan)

                            if "Z 1D" in map_type:
                                self.map_layers["RAW_SPECTRA"][0, zi, :] = spectrum_1d
                            else:
                                self.map_layers["RAW_SPECTRA"][yi, xi, :] = spectrum_1d

                        # 빈 데이터(NaN)가 아닐 때만 연산 수행
                        if not np.all(np.isnan(spectrum_1d)):

                            # 🚨 원본 복사 후 스캔 파라미터에서 넘겨받은 필터 적용
                            processed_1d = spectrum_1d.copy()

                            if self.params.get('use_gaussian', False):
                                processed_1d = apply_gaussian_1d(processed_1d, sigma=1.5)
                            if self.params.get('use_ml', False):
                                processed_1d = self.main_window.spectrum_view.ml_processor.enhance_spectrum(
                                    processed_1d)
                            if self.params.get('use_als', False):
                                processed_1d = remove_als_baseline(processed_1d, lam=1e4, p=0.01, niter=5)

                            # 🚨 스캔 타입에 따라 저장할 배열의 '인덱스 주소'를 동적으로 변경
                            idx = (0, zi) if "Z 1D" in map_type else (yi, xi)

                            if "전체 강도" in self.map_layers:
                                self.map_layers["전체 강도"][idx] = np.max(processed_1d)

                            for pair in active_pairs:
                                layer_name = f"온도 (Pair {pair.pair_id})"
                                if layer_name in self.map_layers:
                                    t = pair.update_temperature(x_axis, processed_1d, is_mapping=True)
                                    self.map_layers[layer_name][idx] = t if t is not None else np.nan

                            for ch in virtual_channels:
                                layer_name = f"채널: {ch.name}"
                                if layer_name in self.map_layers:
                                    val = ch.get_intensity(processed_1d, x_axis)
                                    self.map_layers[layer_name][idx] = val if val is not None else np.nan

                            for form in formulas:
                                layer_name = f"수식: {form.name}"
                                if layer_name in self.map_layers:
                                    ch_values = {ch.name: ch.get_intensity(processed_1d, x_axis) for ch in
                                                 virtual_channels}
                                    val = form.evaluate(ch_values)
                                    self.map_layers[layer_name][idx] = val if val is not None else np.nan

                        current_pt += 1
                        self.progress_signal.emit(current_pt)

                    self.row_finished_signal.emit(yi)


        finally:

            if f_csv is not None:
                f_csv.close()

            # 🚨 맵핑이 정상 종료된 경우 (사용자 정지가 아닐 때) 영점 복귀

            if stage.is_connected and self.is_running:
                print("[Mapping] 스캔 완료. 0점으로 복귀합니다.")

                stage.move_to_logical(0.0, 0.0, 0.0, wait_ack=True)

                time.sleep(0.5)

        self.finished_signal.emit()

    def stop(self):
        self.is_running = False


class HomoepiWorker(QThread):
    progress = pyqtSignal(int)
    update_map = pyqtSignal(float, float, float, object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, main_window, x_start, x_end, x_step, y_start, y_end, y_step, z_start, z_end, z_step,
                 crop_dist=1.5, ignore_dist=1.0):
        super().__init__()
        self.main_window = main_window
        self.x_vals = np.arange(x_start, x_end + x_step / 2, x_step)
        self.y_vals = np.arange(y_start, y_end + y_step / 2, y_step)
        self.z_vals = np.arange(z_start, z_end + z_step / 2, z_step)

        self.crop_dist = crop_dist
        self.ignore_dist = ignore_dist
        self.is_running = True

    def run(self):
        try:
            total_points = len(self.x_vals) * len(self.y_vals)
            current_point = 0

            cam = self.main_window.cam
            stage = self.main_window.stage

            self.waves = self.main_window.spectrum_view.x_axis
            if self.waves is None or len(self.waves) == 0:
                raise ValueError("X축(Wavenumber) 데이터가 없습니다. 모니터링을 시작했는지 확인하세요.")

            mask_776 = (self.waves >= 760) & (self.waves <= 790)
            mask_intf = (self.waves >= 950) & (self.waves <= 1000)
            waves_intf = self.waves[mask_intf]

            z_forward = True

            # 🚨 [수정 1] Y루프에서 인덱스(yi)를 추적합니다.
            for yi, y in enumerate(self.y_vals):

                # 🚨 [수정 2] Y 인덱스가 짝수면 X를 정방향으로, 홀수면 역방향으로 줍니다 (ㄹ자 스캔)
                current_x_vals = self.x_vals if yi % 2 == 0 else self.x_vals[::-1]

                for x in current_x_vals:
                    if not self.is_running: break

                    z_positions = []
                    spectra_list = []

                    if stage.is_connected:
                        start_z = self.z_vals[0] if z_forward else self.z_vals[-1]
                        stage.move_to_logical(x, y, start_z, wait_ack=True)

                    current_z_vals = self.z_vals if z_forward else self.z_vals[::-1]

                    for z in current_z_vals:
                        if not self.is_running: break

                        if stage.is_connected:
                            stage.move_to_logical(x, y, z, wait_ack=False)
                            # 🚨 이동 직후 10ms 대기
                            time.sleep(0.01)

                        if cam.is_connected:
                            frame = cam.grab_frame()
                        else:
                            frame = np.random.randint(0, 40, (480, 640), dtype=np.uint8)

                        if frame is not None:
                            raw_spectrum = np.sum(frame, axis=0)
                        else:
                            raw_spectrum = np.zeros_like(self.waves)

                        z_positions.append(z)
                        spectra_list.append(raw_spectrum)

                    z_forward = not z_forward

                    # --- 실시간 연산 ---
                    z_arr = np.array(z_positions)
                    spectra_arr = np.array(spectra_list)

                    sort_idx = np.argsort(z_arr)
                    z_arr = z_arr[sort_idx]
                    spectra_arr = spectra_arr[sort_idx]

                    valid_idx = z_arr >= (z_arr[0] + self.crop_dist)
                    z_fit = z_arr[valid_idx] if np.any(valid_idx) else z_arr
                    spec_fit = spectra_arr[valid_idx] if np.any(valid_idx) else spectra_arr

                    area_776 = np.sum(spec_fit[:, mask_776], axis=1)

                    # 가중치 기반 무게중심 연산
                    spec_intf = spec_fit[:, mask_intf]
                    spec_intf_bg = spec_intf - np.min(spec_intf, axis=1, keepdims=True)
                    weighted_spec = spec_intf_bg ** 2
                    peak_weighted_com = np.sum(weighted_spec * waves_intf, axis=1) / (
                                np.sum(weighted_spec, axis=1) + 1e-9)

                    norm_area = (area_776 - np.min(area_776)) / (np.ptp(area_776) + 1e-9)
                    norm_peak = (peak_weighted_com - np.min(peak_weighted_com)) / (np.ptp(peak_weighted_com) + 1e-9)

                    try:
                        bounds = ([0, 0.8, min(z_fit), 0.01], [0.2, 1.2, max(z_fit), 5.0])
                        popt_s, _ = curve_fit(sigmoid, z_fit, norm_area, p0=[0, 1, np.mean(z_fit), 0.5], bounds=bounds,
                                              maxfev=1000)
                        z_surf = popt_s[2]

                        valid_mask = np.abs(z_fit - z_surf) >= self.ignore_dist
                        if np.sum(valid_mask) >= 3:
                            popt_i, _ = curve_fit(sigmoid, z_fit[valid_mask], norm_peak[valid_mask],
                                                  p0=[0, 1, np.mean(z_fit), 0.5], bounds=bounds, maxfev=1000)
                            z_intf = popt_i[2]
                        else:
                            z_intf = z_surf

                        thickness = abs(z_intf - z_surf)
                    except:
                        thickness = 0.0

                    raw_data = {"z": z_arr, "spectra": spectra_arr, "waves": self.waves}
                    self.update_map.emit(x, y, thickness, raw_data)

                    current_point += 1
                    self.progress.emit(int(current_point / total_points * 100))

                # X루프가 끝난 뒤 스캔 강제 중지 확인
                if not self.is_running: break

            if stage.is_connected:
                stage.move_to_logical(0, 0, 0)

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.error.emit(f"스캔 중 오류 발생:\n{str(e)}\n\n{err_msg}")


        finally:

            # 🚨 맵핑이 정상 종료된 경우 영점 복귀

            if stage.is_connected and self.is_running:
                print("[Homoepi] 3D 스캔 완료. 0점으로 복귀합니다.")

                stage.move_to_logical(0.0, 0.0, 0.0, wait_ack=True)

                time.sleep(0.5)

            self.finished.emit()

    def stop(self):
        self.is_running = False


class TempLogViewWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.start_time = 0.0
        self.is_logging = False

        # 데이터 저장용 버퍼
        self.time_data = []
        self.temp_data = {}
        self.curves = {}

        # 채널 Intensity 저장용 버퍼
        self.channel_data = {}
        self.channel_curves = {}

        layout = QVBoxLayout(self)

        # 상단 제어 버튼
        ctrl_layout = QHBoxLayout()

        self.btn_toggle_log = QPushButton("▶ 로깅 시작")
        self.btn_toggle_log.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_toggle_log.clicked.connect(self.toggle_logging)

        self.btn_clear = QPushButton("🗑️ 로그 초기화")
        self.btn_clear.clicked.connect(self.clear_log)
        self.btn_save = QPushButton("💾 CSV로 내보내기")
        self.btn_save.clicked.connect(self.save_csv)
        self.btn_save.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")

        ctrl_layout.addWidget(self.btn_toggle_log)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.btn_save)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # 실시간 그래프 위젯 세팅
        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', 'd')
        self.plot_widget = pg.PlotWidget()

        # 왼쪽 축 (온도)
        self.plot_widget.setLabel('left', 'Temperature (℃)', color='#00e5ff')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # 오른쪽 보조 축 (Intensity) 생성 및 연동
        self.plot_widget.showAxis('right')
        self.plot_widget.getAxis('right').setLabel('Channel Intensity', color='#ffff00')
        self.vb_right = pg.ViewBox()
        self.plot_widget.scene().addItem(self.vb_right)
        self.plot_widget.getAxis('right').linkToView(self.vb_right)

        # X축을 메인 PlotItem의 ViewBox와 동기화
        self.vb_right.setXLink(self.plot_widget.plotItem.vb)

        # 보조 축이 X축 스케일 권한을 뺏어가는 것을 완벽 차단
        self.vb_right.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.vb_right.disableAutoRange(axis=pg.ViewBox.XAxis)
        self.vb_right.setAutoVisible(x=False, y=True)

        # 메인 축(왼쪽)은 X, Y 전체 오토스케일 확실히 켜기
        self.plot_widget.plotItem.vb.enableAutoRange(enable=True)

        # 뷰 크기가 변할 때 오른쪽 축도 동기화시키는 함수
        def updateViews():
            self.vb_right.setGeometry(self.plot_widget.plotItem.vb.sceneBoundingRect())
            self.vb_right.linkedViewChanged(self.plot_widget.plotItem.vb, self.vb_right.XAxis)

        updateViews()
        self.plot_widget.plotItem.vb.sigResized.connect(updateViews)

        # 🚨 [가장 중요한 수정] 완전히 투명한 색(Alpha=0)으로 지정하여 오토스케일 계산에 강제 참여시킴
        self.dummy_curve = self.plot_widget.plot(pen=pg.mkPen(color=(0, 0, 0, 0)))

        layout.addWidget(self.plot_widget)

    def toggle_logging(self):
        if not self.is_logging:
            self.is_logging = True
            self.start_time = time.time()
            self.btn_toggle_log.setText("⏹ 로깅 중지")
            self.btn_toggle_log.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        else:
            self.is_logging = False
            self.btn_toggle_log.setText("▶ 로깅 시작")
            self.btn_toggle_log.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

    def update_log(self, temp_dict, channel_dict=None):
        if not hasattr(self, 'is_logging') or not self.is_logging:
            return

        if channel_dict is None:
            channel_dict = {}

        current_time = time.time() - self.start_time
        self.time_data.append(current_time)

        max_points = 1000
        if len(self.time_data) > max_points:
            self.time_data.pop(0)

        # 🚨 [핵심 해결 2] 온도 데이터가 없을 때 가상 온도(50℃)를 넣어 메인 Y축이 0~100 근처를 유지하도록 속임
        if not temp_dict and not self.temp_data:
            dummy_y = [50.0] * len(self.time_data)
            self.dummy_curve.setData(self.time_data, dummy_y)
        else:
            self.dummy_curve.setData([], [])  # 실제 온도 데이터가 들어오면 가짜 선은 비움

        # 1. 온도 데이터 업데이트
        for pair_id, temp in temp_dict.items():
            if pair_id not in self.temp_data:
                self.temp_data[pair_id] = []
                curve_color = '#FFFFFF'
                if hasattr(self.main_window, 'control_panel'):
                    for pair in self.main_window.control_panel.active_pairs:
                        if str(pair.pair_id) == str(pair_id):
                            curve_color = getattr(pair, 'color_hex', '#00e5ff')
                            break

                self.curves[pair_id] = self.plot_widget.plot(
                    pen=pg.mkPen(color=curve_color, width=2),
                    name=f"Pair {pair_id} Temp"
                )

            self.temp_data[pair_id].append(temp)
            if len(self.temp_data[pair_id]) > max_points:
                self.temp_data[pair_id].pop(0)

            plot_len = min(len(self.time_data), len(self.temp_data[pair_id]))
            self.curves[pair_id].setData(
                self.time_data[-plot_len:],
                self.temp_data[pair_id][-plot_len:]
            )

        # 2. 채널 Intensity 데이터 업데이트 (오른쪽 축에 표시)
        for ch_name, intensity in channel_dict.items():
            if ch_name not in self.channel_data:
                self.channel_data[ch_name] = []
                # 채널 그래프는 노란색 계열의 점선으로 차별화
                curve = pg.PlotDataItem(
                    pen=pg.mkPen(color='#FFFF00', width=2, style=Qt.PenStyle.DashLine),
                    name=f"Ch {ch_name}"
                )
                self.channel_curves[ch_name] = curve
                self.vb_right.addItem(curve)

            self.channel_data[ch_name].append(intensity)
            if len(self.channel_data[ch_name]) > max_points:
                self.channel_data[ch_name].pop(0)

            plot_len = min(len(self.time_data), len(self.channel_data[ch_name]))
            self.channel_curves[ch_name].setData(
                self.time_data[-plot_len:],
                self.channel_data[ch_name][-plot_len:]
            )

        # 3. A버튼(오토스케일) 눌렀을 때 보조 축도 함께 동기화
        auto_range_state = self.plot_widget.plotItem.vb.state.get('autoRange', [True, True])
        if auto_range_state[0]:
            self.vb_right.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    def clear_log(self):
        self.start_time = time.time()
        self.time_data.clear()
        self.dummy_curve.setData([], [])

        self.temp_data.clear()
        for curve in self.curves.values():
            self.plot_widget.removeItem(curve)
        self.curves.clear()

        self.channel_data.clear()
        for curve in self.channel_curves.values():
            self.vb_right.removeItem(curve)
        self.channel_curves.clear()

    def save_csv(self):
        if not self.time_data:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Multi-Log Data", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                headers = ["Time (s)"]
                headers += [f"Pair {pid} Temp(℃)" for pid in self.temp_data.keys()]
                headers += [f"Channel {c_name} Intensity" for c_name in self.channel_data.keys()]
                writer.writerow(headers)

                for i in range(len(self.time_data)):
                    row = [f"{self.time_data[i]:.2f}"]
                    for pid in self.temp_data.keys():
                        val = self.temp_data[pid][i] if i < len(self.temp_data[pid]) else ""
                        row.append(f"{val:.2f}" if isinstance(val, float) else val)
                    for c_name in self.channel_data.keys():
                        val = self.channel_data[c_name][i] if i < len(self.channel_data[c_name]) else ""
                        row.append(f"{val:.2f}" if isinstance(val, float) else val)

                    writer.writerow(row)
# -------------------------------------------------------------------
# 2. 메인 시각화 위젯 (2D, 1D)
# -------------------------------------------------------------------

class LiveViewWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.cam = main_window.cam
        self.current_frame = None

        layout = QVBoxLayout(self)

        # 저장 버튼 추가
        top_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 2D Live 데이터 저장 (이미지/CSV)")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_data)
        top_layout.addWidget(self.btn_save)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        # 기존 QLabel 대신 pyqtgraph 적용 (마우스 스크롤 줌, 드래그 팬 자동 지원)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(title="2D Camera Feed")
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        self.plot.invertY(True)  # 카메라 센서 방향에 맞게 Y축 뒤집기
        layout.addWidget(self.glw)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_live(self):
        if self.cam.is_connected: self.cam.start_capture()
        self.timer.start(50)

    def stop_live(self):
        self.timer.stop()
        if self.cam.is_connected: self.cam.stop_capture()

    def update_frame(self):
        frame = self.cam.grab_frame() if self.cam.is_connected else None

        # 가상 노이즈 생성 (테스트용)
        if frame is None and not self.cam.is_connected:
            x_pixels = np.arange(640)
            frame = np.random.randint(0, 40, (480, 640), dtype=np.uint8)

        if frame is not None:
            self.current_frame = frame
            # autoLevels=True 로 화면 밝기 자동 최적화
            self.img_item.setImage(frame, autoLevels=True)

            spectrum_1d = np.sum(frame, axis=0)
            self.main_window.spectrum_view.process_spectrum(spectrum_1d)

    def save_data(self):
        if self.current_frame is None: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save 2D Data", "", "CSV Files (*.csv);;PNG Image (*.png)")
        if file_path:
            if file_path.endswith('.csv'):
                np.savetxt(file_path, self.current_frame, delimiter=",", fmt='%d')
            else:
                exporter = pg.exporters.ImageExporter(self.plot.scene())
                exporter.export(file_path)


class SpectrumViewWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.ml_processor = RamanMLProcessor("raman_model_v22_final.pth")
        self.rolling_buffer = []

        self.x_axis = np.arange(640)
        self.current_y_data = None
        self.is_calibrated = False
        self.calibration_func = None

        main_layout = QVBoxLayout(self)

        # 상단 제어 바
        top_bar = QHBoxLayout()
        self.chk_filter = QCheckBox("Gaussian 필터")
        self.chk_ml = QCheckBox("ML Enhancement")
        self.chk_als = QCheckBox("ALS 베이스라인")
        self.btn_save = QPushButton("💾 1D 스펙트럼 저장")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_save.clicked.connect(self.save_data)
        top_bar.addWidget(self.btn_save)

        self.spin_accum = QSpinBox()
        self.spin_accum.setRange(1, 100)
        self.spin_accum.setValue(10)

        top_bar.addWidget(self.chk_filter)
        top_bar.addWidget(self.chk_ml)
        top_bar.addWidget(self.chk_als)
        top_bar.addWidget(QLabel("Accumulation:"))
        top_bar.addWidget(self.spin_accum)
        top_bar.addStretch()

        self.lbl_click_info = QLabel("Double Click on Graph to Inspect")
        self.lbl_click_info.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 13px;")
        top_bar.addWidget(self.lbl_click_info)
        main_layout.addLayout(top_bar)

        self.pair_control_layout = QVBoxLayout()
        main_layout.addLayout(self.pair_control_layout)

        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', 'd')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Intensity (a.u.)')
        self.plot_widget.setLabel('bottom', 'Pixels')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)

        self.spectrum_line = self.plot_widget.plot(pen=pg.mkPen(color='#FFFFFF', width=2))
        main_layout.addWidget(self.plot_widget)

    def on_plot_clicked(self, ev):
        if ev.double():
            pos = ev.scenePos()
            if self.plot_widget.plotItem.vb.sceneBoundingRect().contains(pos):
                mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                x_clicked = mousePoint.x()

                if len(self.x_axis) > 0 and self.current_y_data is not None:
                    idx = np.argmin(np.abs(self.x_axis - x_clicked))
                    pixel_num = idx
                    raman_shift = self.x_axis[idx]
                    intensity = self.current_y_data[idx]

                    unit = "cm⁻¹" if self.is_calibrated else "px"
                    self.lbl_click_info.setText(
                        f"🎯 Pixel: {pixel_num} | Shift: {raman_shift:.1f} {unit} | Intensity: {intensity:.1f}")

    def apply_calibration_quadratic(self, pixels, wavelengths, laser_wl):
        try:
            coeffs, transform_func = calibrate_raman_axis_quadratic(pixels, wavelengths, laser_wl)
            self.calibration_func = transform_func  # 보정 함수 저장

            # 현재 들어온 카메라 데이터 길이에 맞춰서 픽셀 배열 생성 (데이터가 없으면 기본 2304)
            data_len = len(self.current_y_data) if self.current_y_data is not None else 2304
            raw_pixels = np.arange(data_len)

            self.x_axis = self.calibration_func(raw_pixels)  # 변환 적용
            self.is_calibrated = True
            self.plot_widget.setLabel('bottom', 'Raman Shift (cm⁻¹)')
            return True
        except Exception as e:
            print(f"Calibration Error: {e}")
            raise RuntimeError(f"보정 계산 실패: {e}")

    def process_spectrum(self, raw_1d: np.ndarray):
        target_frames = self.spin_accum.value()
        self.rolling_buffer.append(raw_1d)

        # 버퍼 길이 조절
        while len(self.rolling_buffer) > target_frames:
            self.rolling_buffer.pop(0)

        processed = np.mean(self.rolling_buffer, axis=0)

        if self.chk_filter.isChecked(): processed = apply_gaussian_1d(processed, sigma=1.5)
        if self.chk_ml.isChecked(): processed = self.ml_processor.enhance_spectrum(processed)
        if self.chk_als.isChecked(): processed = remove_als_baseline(processed, lam=1e4, p=0.01, niter=5)

        if len(self.x_axis) != len(processed):
            if self.is_calibrated and self.calibration_func is not None:
                self.x_axis = self.calibration_func(np.arange(len(processed)))
            else:
                self.x_axis = np.arange(len(processed))

        self.current_y_data = processed
        self.spectrum_line.setData(self.x_axis, processed)

        temp_results = {}  # 실시간 온도를 모아둘 딕셔너리

        for pair in list(self.main_window.control_panel.active_pairs):
            try:
                temp = pair.update_temperature(self.x_axis, processed)
                if temp is not None:
                    temp_results[pair.pair_id] = temp
            except Exception as e:
                pass

        # 🚨 [신규 추가] 가상 채널 Intensity 값을 계산하여 모아둠
        channel_results = {}
        if hasattr(self.main_window, 'mapping_view'):
            for ch in list(self.main_window.mapping_view.virtual_channels):
                try:
                    intensity = ch.get_intensity(processed, self.x_axis)
                    if intensity is not None and not np.isnan(intensity):
                        channel_results[ch.name] = intensity
                except Exception as e:
                    pass

        # 🚨 [수정] 온도 데이터와 채널 데이터를 함께 로깅 위젯으로 전송
        if hasattr(self.main_window, 'temp_log_view') and (temp_results or channel_results):
            self.main_window.temp_log_view.update_log(temp_results, channel_results)

    def save_data(self):
        if self.current_y_data is None: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save 1D Spectrum", "", "CSV Files (*.csv);;PNG Image (*.png)")
        if file_path:
            if file_path.endswith('.csv'):
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Raman Shift / Pixel", "Intensity"])
                    for x, y in zip(self.x_axis, self.current_y_data):
                        writer.writerow([x, y])
            else:
                exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
                exporter.export(file_path)


# -------------------------------------------------------------------
# 3. 3번째 탭: Mapping & Scan 설정 위젯
# -------------------------------------------------------------------
class MappingViewWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.virtual_channels = []
        self.formulas = []
        self.map_layers = {}

        layout = QHBoxLayout(self)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(400)

        # --- (좌측 스캔 모드 및 수식 빌더 UI: 기존과 동일하게 유지) ---
        scan_group = QGroupBox("Mapping Type & Range (Relative to Zero, μm)")
        scan_layout = QGridLayout()
        self.combo_map_type = QComboBox()
        self.combo_map_type.addItems(["XY 2D 맵핑", "Z 1D 깊이 스캔", "XYZ 3D 맵핑"])
        self.combo_map_type.currentIndexChanged.connect(self.on_map_type_changed)

        scan_layout.addWidget(QLabel("Type:"), 0, 0)
        scan_layout.addWidget(self.combo_map_type, 0, 1, 1, 3)

        scan_layout.addWidget(QLabel("Axis"), 1, 0)
        scan_layout.addWidget(QLabel("Start"), 1, 1)
        scan_layout.addWidget(QLabel("End"), 1, 2)
        scan_layout.addWidget(QLabel("Step"), 1, 3)

        self.x_start = QDoubleSpinBox();
        self.x_start.setRange(-10000, 10000);
        self.x_start.setValue(0.0)
        self.x_end = QDoubleSpinBox();
        self.x_end.setRange(-10000, 10000);
        self.x_end.setValue(10.0)
        self.x_step = QDoubleSpinBox();
        self.x_step.setRange(0.01, 1000);
        self.x_step.setValue(1.0)

        self.y_start = QDoubleSpinBox();
        self.y_start.setRange(-10000, 10000);
        self.y_start.setValue(0.0)
        self.y_end = QDoubleSpinBox();
        self.y_end.setRange(-10000, 10000);
        self.y_end.setValue(10.0)
        self.y_step = QDoubleSpinBox();
        self.y_step.setRange(0.01, 1000);
        self.y_step.setValue(1.0)

        self.z_start = QDoubleSpinBox();
        self.z_start.setRange(-10000, 10000);
        self.z_start.setValue(0.0)
        self.z_end = QDoubleSpinBox();
        self.z_end.setRange(-10000, 10000);
        self.z_end.setValue(0.0)
        self.z_step = QDoubleSpinBox();
        self.z_step.setRange(0.01, 1000);
        self.z_step.setValue(1.0)

        scan_layout.addWidget(QLabel("X:"), 2, 0);
        scan_layout.addWidget(self.x_start, 2, 1);
        scan_layout.addWidget(self.x_end, 2, 2);
        scan_layout.addWidget(self.x_step, 2, 3)
        scan_layout.addWidget(QLabel("Y:"), 3, 0);
        scan_layout.addWidget(self.y_start, 3, 1);
        scan_layout.addWidget(self.y_end, 3, 2);
        scan_layout.addWidget(self.y_step, 3, 3)
        scan_layout.addWidget(QLabel("Z:"), 4, 0);
        scan_layout.addWidget(self.z_start, 4, 1);
        scan_layout.addWidget(self.z_end, 4, 2);
        scan_layout.addWidget(self.z_step, 4, 3)

        scan_group.setLayout(scan_layout)
        left_layout.addWidget(scan_group)

        builder_group = QGroupBox("Virtual Channel & Formula Builder")
        builder_layout = QVBoxLayout()
        self.list_channels = QListWidget()
        self.list_channels.setFixedHeight(80)
        builder_layout.addWidget(self.list_channels)

        self.btn_delete_item = QPushButton("🗑️ 선택 항목 삭제")
        self.btn_delete_item.clicked.connect(self.delete_selected_item)
        builder_layout.addWidget(self.btn_delete_item)

        add_ch_layout = QHBoxLayout()
        self.ch_name = QLineEdit("Ch1");
        self.ch_name.setFixedWidth(40)
        self.ch_mode = QComboBox();
        self.ch_mode.addItems(["적분", "최대값", "피크 위치", "무게중심"])
        self.ch_center = QLineEdit("520");
        self.ch_width = QLineEdit("15")
        btn_add_ch = QPushButton("+ 채널");
        btn_add_ch.clicked.connect(self.add_virtual_channel)

        add_ch_layout.addWidget(self.ch_name);
        add_ch_layout.addWidget(self.ch_mode)
        add_ch_layout.addWidget(QLabel("C:"));
        add_ch_layout.addWidget(self.ch_center)
        add_ch_layout.addWidget(QLabel("W:"));
        add_ch_layout.addWidget(self.ch_width)
        add_ch_layout.addWidget(btn_add_ch)
        builder_layout.addLayout(add_ch_layout)

        add_form_layout = QHBoxLayout()
        self.form_name = QLineEdit("Ratio");
        self.form_name.setFixedWidth(50)
        self.form_expr = QLineEdit("Ch1 / Ch2")
        btn_add_form = QPushButton("+ 수식");
        btn_add_form.clicked.connect(self.add_formula)

        add_form_layout.addWidget(self.form_name);
        add_form_layout.addWidget(self.form_expr);
        add_form_layout.addWidget(btn_add_form)
        builder_layout.addLayout(add_form_layout)

        builder_group.setLayout(builder_layout)
        left_layout.addWidget(builder_group)
        left_layout.addStretch()

        # 🚨 [핵심 복구] 상단은 맵(Map), 하단은 스펙트럼(Plot)을 보여주기 위한 QSplitter 구성
        right_panel = QSplitter(Qt.Orientation.Vertical)

        # 1. 상단 맵 패널
        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)

        view_select_layout = QHBoxLayout()
        view_select_layout.addWidget(QLabel("<b>👁️ 시각화 대상 선택:</b>"))
        self.combo_display_target = QComboBox()
        self.combo_display_target.setStyleSheet("color: #00e5ff; font-weight: bold; background-color: #2b2b2b;")
        self.combo_display_target.currentIndexChanged.connect(self.refresh_mapping_display)
        view_select_layout.addWidget(self.combo_display_target, 1)
        map_layout.addLayout(view_select_layout)

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.glw_map = pg.GraphicsLayoutWidget()
        self.plot_map = self.glw_map.addPlot()

        self.img_item = pg.ImageItem()
        self.plot_map.addItem(self.img_item)

        self.line_item = self.plot_map.plot(pen=pg.mkPen('y', width=2), symbol='o', symbolSize=5, symbolBrush='b')
        self.line_item.hide()

        # 스케일 자동 조절바(히스토그램) 부착
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.glw_map.addItem(self.hist)

        self.plot_map.hideAxis('left')
        self.plot_map.hideAxis('bottom')

        # 💡 [신규] 맵에서 픽셀 클릭 시 스펙트럼 표시 이벤트 연결
        self.img_item.scene().sigMouseClicked.connect(self.on_map_clicked)

        map_layout.addWidget(self.glw_map)

        # 2. 하단 스펙트럼 패널 (부활!)
        spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_widget)

        self.glw_spec = pg.GraphicsLayoutWidget()
        self.plot_spec = self.glw_spec.addPlot(title="Point Spectrum")
        self.plot_spec.setLabel('bottom', "Wavenumber", units="cm⁻¹")
        self.plot_spec.setLabel('left', "Intensity")
        self.spec_curve = self.plot_spec.plot(pen='c', name="Spectrum")

        spectrum_layout.addWidget(self.glw_spec)

        # Splitter에 상/하 패널 추가
        right_panel.addWidget(map_widget)
        right_panel.addWidget(spectrum_widget)
        right_panel.setSizes([600, 300])  # 상하 비율 2:1 설정

        # 제어용 버튼들 (우측 패널 최하단)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        self.lbl_scan_status = QLabel("스캔 상태: 대기 중")
        self.lbl_scan_status.setStyleSheet("font-weight: bold; color: #ff9800;")
        control_layout.addWidget(self.lbl_scan_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        self.chk_save_full = QCheckBox("전체 스펙트럼 실시간 저장 (.csv)")
        self.chk_save_full.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.chk_save_full.setChecked(True)
        control_layout.addWidget(self.chk_save_full)

        btn_layout = QHBoxLayout()
        self.btn_save_map = QPushButton("💾 현재 표시된 데이터 저장 (이미지/CSV)")
        self.btn_save_map.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_save_map.clicked.connect(self.save_data)

        self.btn_start_mapping = QPushButton("▶ 스캔 시작")
        self.btn_start_mapping.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; height: 40px;")
        self.btn_start_mapping.clicked.connect(self.toggle_mapping)

        btn_layout.addWidget(self.btn_save_map)
        btn_layout.addWidget(self.btn_start_mapping)
        control_layout.addLayout(btn_layout)

        # 전체 레이아웃 조립
        right_main_layout = QVBoxLayout()
        right_main_layout.addWidget(right_panel, stretch=1)
        right_main_layout.addWidget(control_widget)

        right_main_container = QWidget()
        right_main_container.setLayout(right_main_layout)

        layout.addWidget(left_panel)
        layout.addWidget(right_main_container, stretch=1)

        self.on_map_type_changed()

    # 💡 [핵심 복구] 맵 픽셀 클릭 시 해당 위치의 원본 스펙트럼을 하단에 그려주는 기능
    def on_map_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = self.img_item.mapFromScene(event.scenePos())
        x_idx = int(pos.x())
        y_idx = int(pos.y())

        # 🚨 [수정됨] RAW 데이터가 메모리에 없을 경우 안내 메시지 출력
        if "RAW_SPECTRA" in self.map_layers:
            raw_data = self.map_layers["RAW_SPECTRA"]
            if 0 <= y_idx < raw_data.shape[0] and 0 <= x_idx < raw_data.shape[1]:
                spec = raw_data[y_idx, x_idx, :]
                if not np.all(np.isnan(spec)):
                    x_axis = self.main_window.spectrum_view.x_axis
                    self.spec_curve.setData(x=x_axis, y=spec)
                    self.plot_spec.setTitle(f"Point Spectrum (X: {x_idx}, Y: {y_idx})")
        else:
            self.plot_spec.setTitle("※ RAW 저장 체크 해제됨 (스펙트럼 정보 없음)")
            self.spec_curve.setData([], [])

    # (이하 on_map_type_changed, add_virtual_channel, add_formula, delete_selected_item 동일 유지)
    def on_map_type_changed(self):
        mode = self.combo_map_type.currentText()
        is_xy = "XY" in mode or "XYZ" in mode
        is_z = "Z 1D" in mode or "XYZ" in mode

        self.x_start.setEnabled(is_xy);
        self.x_end.setEnabled(is_xy);
        self.x_step.setEnabled(is_xy)
        self.y_start.setEnabled(is_xy);
        self.y_end.setEnabled(is_xy);
        self.y_step.setEnabled(is_xy)
        self.z_start.setEnabled(is_z);
        self.z_end.setEnabled(is_z);
        self.z_step.setEnabled(is_z)

    def add_virtual_channel(self):
        name = self.ch_name.text().strip()
        txt = self.ch_mode.currentText()

        # 🚨 오타 및 로직 완벽 수정
        if "적분" in txt: mode = "integrate"
        elif "최대값" in txt: mode = "max"
        elif "위치" in txt: mode = "position"
        elif "중심" in txt: mode = "com"
        else: mode = "integrate"

        center = float(self.ch_center.text());
        width = float(self.ch_width.text())
        self.virtual_channels.append(VirtualChannel(name, mode, center, width))
        self.list_channels.addItem(f"[채널: {name}] {txt} | C: {center}, W: {width}")
        if name.startswith("Ch") and name[2:].isdigit(): self.ch_name.setText(f"Ch{int(name[2:]) + 1}")

    def add_formula(self):
        name = self.form_name.text().strip();
        expr = self.form_expr.text().strip()
        self.formulas.append(CustomFormula(name, expr))
        self.list_channels.addItem(f"[수식: {name}] = {expr}")

    def delete_selected_item(self):
        current_item = self.list_channels.currentItem()
        if not current_item: return
        text = current_item.text()
        if text.startswith("[채널:"):
            ch_name = text.split("]")[0].replace("[채널: ", "").strip()
            self.virtual_channels = [ch for ch in self.virtual_channels if ch.name != ch_name]
        elif text.startswith("[수식:"):
            f_name = text.split("]")[0].replace("[수식: ", "").strip()
            self.formulas = [f for f in self.formulas if f.name != f_name]
        self.list_channels.takeItem(self.list_channels.row(current_item))

    def toggle_mapping(self):
        if not hasattr(self, 'mapping_worker') or not self.mapping_worker.isRunning():

            # 🚨 카메라 충돌(None 발생) 방지를 위해 라이브뷰 타이머 일시 정지
            if hasattr(self.main_window, 'live_view'):
                self.main_window.live_view.timer.stop()

            map_type = self.combo_map_type.currentText()
            scan_params = {
                'map_type': map_type,
                'save_full': self.chk_save_full.isChecked(),
                'x_start': self.x_start.value(), 'x_end': self.x_end.value(), 'x_step': self.x_step.value(),
                'y_start': self.y_start.value(), 'y_end': self.y_end.value(), 'y_step': self.y_step.value(),
                'z_start': self.z_start.value(), 'z_end': self.z_end.value(), 'z_step': self.z_step.value(),
                # 🚨 필터 상태를 스레드에 전달
                'use_gaussian': self.main_window.spectrum_view.chk_filter.isChecked(),
                'use_ml': self.main_window.spectrum_view.chk_ml.isChecked(),
                'use_als': self.main_window.spectrum_view.chk_als.isChecked()
            }

            pts_x = max(1,
                        int(abs(scan_params['x_end'] - scan_params['x_start']) / max(0.01, scan_params['x_step'])) + 1)
            pts_y = max(1,
                        int(abs(scan_params['y_end'] - scan_params['y_start']) / max(0.01, scan_params['y_step'])) + 1)
            pts_z = max(1,
                        int(abs(scan_params['z_end'] - scan_params['z_start']) / max(0.01, scan_params['z_step'])) + 1)

            shape = (1, pts_z) if "Z 1D" in map_type else (pts_y, pts_x)
            self.sim_total_points = pts_z if "Z 1D" in map_type else pts_x * pts_y * (pts_z if "XYZ" in map_type else 1)

            self.map_layers = {
                "전체 강도": np.full(shape, np.nan)
            }

            active_pairs = self.main_window.control_panel.active_pairs
            for pair in active_pairs:
                self.map_layers[f"온도 (Pair {pair.pair_id})"] = np.full(shape, np.nan)

            for ch in self.virtual_channels:
                self.map_layers[f"채널: {ch.name}"] = np.full(shape, np.nan)
            for form in self.formulas:
                self.map_layers[f"수식: {form.name}"] = np.full(shape, np.nan)

            self.combo_display_target.blockSignals(True)
            self.combo_display_target.clear()
            display_keys = [k for k in self.map_layers.keys() if k != "RAW_SPECTRA"]
            self.combo_display_target.addItems(display_keys)
            self.combo_display_target.blockSignals(False)

            self.progress_bar.setRange(0, self.sim_total_points)
            exposure_val = float(self.main_window.control_panel.exposure_input.text())

            self.mapping_worker = MappingWorker(self.main_window, scan_params, self.map_layers,
                                                exposure_time=exposure_val)
            self.mapping_worker.progress_signal.connect(self.update_mapping_progress)
            self.mapping_worker.row_finished_signal.connect(self.refresh_mapping_display)
            self.mapping_worker.finished_signal.connect(self.stop_simulation)
            self.mapping_worker.start()

            self.btn_start_mapping.setText("⏹ 스캔 중단")
            self.btn_start_mapping.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
            if self.main_window.control_panel.chk_auto_update.isChecked():
                self.main_window.control_panel.update_timer.stop()
        else:
            self.stop_simulation()

    def refresh_mapping_display(self, dummy_arg=None):
        target_key = self.combo_display_target.currentText()
        if not target_key or target_key not in self.map_layers: return

        selected_data = self.map_layers[target_key]
        valid_data = selected_data[~np.isnan(selected_data)]

        if selected_data.shape[0] == 1:
            self.img_item.hide()
            self.line_item.show()
            self.plot_map.showAxis('bottom')
            self.plot_map.showAxis('left')
            z_start = self.z_start.value();
            z_step = self.z_step.value()
            z_axis = np.arange(selected_data.shape[1]) * z_step + z_start
            mask = ~np.isnan(selected_data[0, :])
            if np.any(mask): self.line_item.setData(x=z_axis[mask], y=selected_data[0, mask])
        else:
            self.line_item.hide()
            self.img_item.show()
            self.plot_map.hideAxis('bottom')
            self.plot_map.hideAxis('left')

            if len(valid_data) > 0:
                v_min, v_max = valid_data.min(), valid_data.max()
                if v_min == v_max: v_max += 0.1
                self.img_item.setImage(selected_data, autoLevels=False, levels=(v_min, v_max))
            else:
                self.img_item.setImage(selected_data, autoLevels=False, levels=(0, 1))

    # 화면 갱신 없이 프로그레스 바만 업데이트하는 전용 함수
    def update_mapping_progress(self, current_pt):
        self.progress_bar.setValue(current_pt)
        self.lbl_scan_status.setText(f"스캔 중... {current_pt} / {self.sim_total_points}")

    def stop_simulation(self):
        if hasattr(self, 'mapping_worker'):
            self.mapping_worker.stop()
            self.mapping_worker.quit()
            self.mapping_worker.wait()

        self.btn_start_mapping.setText("▶ 스캔 시작")
        self.btn_start_mapping.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; height: 40px;")
        self.lbl_scan_status.setText("스캔 완료 / 대기 중")
        self.refresh_mapping_display()  # 마지막 종료 시 한 번 더 갱신

        # 🚨 스캔 종료 후 라이브 모니터링 중이었다면 타이머를 다시 켬
        if getattr(self.main_window, 'is_measuring', False):
            self.main_window.live_view.timer.start(50)

        if self.main_window.control_panel.chk_auto_update.isChecked():
            self.main_window.control_panel.update_timer.start(100)

    def save_data(self):
        target_key = self.combo_display_target.currentText()
        if not target_key or target_key not in self.map_layers: return

        current_data = self.map_layers[target_key]

        file_path, _ = QFileDialog.getSaveFileName(self, f"Save [{target_key}] Result", "",
                                                   "CSV Files (*.csv);;PNG Image (*.png)")
        if file_path:
            if file_path.endswith('.csv'):
                import pandas as pd
                # 1D Z Depth 스캔인 경우
                if current_data.shape[0] == 1:
                    z_vals = np.arange(current_data.shape[1]) * self.z_step.value() + self.z_start.value()
                    df = pd.DataFrame({"Z (um)": z_vals, "Value": current_data[0, :]})
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                # 2D XY 스캔인 경우
                else:
                    x_vals = np.arange(self.x_start.value(), self.x_end.value() + self.x_step.value() / 2,
                                       self.x_step.value())
                    y_vals = np.arange(self.y_start.value(), self.y_end.value() + self.y_step.value() / 2,
                                       self.y_step.value())

                    min_y = min(len(y_vals), current_data.shape[0])
                    min_x = min(len(x_vals), current_data.shape[1])

                    df = pd.DataFrame(current_data[:min_y, :min_x],
                                      index=np.round(y_vals[:min_y], 3),
                                      columns=np.round(x_vals[:min_x], 3))
                    df.index.name = "Y \ X"
                    df.to_csv(file_path, encoding='utf-8-sig')
            else:
                exporter = pg.exporters.ImageExporter(self.plot_map.scene())
                exporter.export(file_path)


class HomoepiViewWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.depth_profiles = {}

        # [신규] 현재 클릭한 픽셀 위치 추적용 변수
        self.selected_x = None
        self.selected_y = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- 3D 스캔 범위 및 스텝 설정 그룹박스 ---
        scan_group = QGroupBox("3D 스캔 범위 및 스텝 설정")
        scan_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        scan_layout = QGridLayout(scan_group)

        # X축 설정
        self.sp_x_start = QDoubleSpinBox();
        self.sp_x_start.setRange(-10000, 10000);
        self.sp_x_start.setValue(-10)
        self.sp_x_end = QDoubleSpinBox();
        self.sp_x_end.setRange(-10000, 10000);
        self.sp_x_end.setValue(10)
        self.sp_x_step = QDoubleSpinBox();
        self.sp_x_step.setRange(0.01, 1000);
        self.sp_x_step.setValue(2)
        scan_layout.addWidget(QLabel("X Start(µm):"), 0, 0);
        scan_layout.addWidget(self.sp_x_start, 0, 1)
        scan_layout.addWidget(QLabel("X End(µm):"), 0, 2);
        scan_layout.addWidget(self.sp_x_end, 0, 3)
        scan_layout.addWidget(QLabel("X Step(µm):"), 0, 4);
        scan_layout.addWidget(self.sp_x_step, 0, 5)

        # Y축 설정
        self.sp_y_start = QDoubleSpinBox();
        self.sp_y_start.setRange(-10000, 10000);
        self.sp_y_start.setValue(-10)
        self.sp_y_end = QDoubleSpinBox();
        self.sp_y_end.setRange(-10000, 10000);
        self.sp_y_end.setValue(10)
        self.sp_y_step = QDoubleSpinBox();
        self.sp_y_step.setRange(0.01, 1000);
        self.sp_y_step.setValue(2)
        scan_layout.addWidget(QLabel("Y Start(µm):"), 1, 0);
        scan_layout.addWidget(self.sp_y_start, 1, 1)
        scan_layout.addWidget(QLabel("Y End(µm):"), 1, 2);
        scan_layout.addWidget(self.sp_y_end, 1, 3)
        scan_layout.addWidget(QLabel("Y Step(µm):"), 1, 4);
        scan_layout.addWidget(self.sp_y_step, 1, 5)

        # Z축 설정
        self.sp_z_start = QDoubleSpinBox();
        self.sp_z_start.setRange(-10000, 10000);
        self.sp_z_start.setValue(-5)
        self.sp_z_end = QDoubleSpinBox();
        self.sp_z_end.setRange(-10000, 10000);
        self.sp_z_end.setValue(15)
        self.sp_z_step = QDoubleSpinBox();
        self.sp_z_step.setRange(0.01, 1000);
        self.sp_z_step.setValue(0.5)
        scan_layout.addWidget(QLabel("Z Start(µm):"), 2, 0);
        scan_layout.addWidget(self.sp_z_start, 2, 1)
        scan_layout.addWidget(QLabel("Z End(µm):"), 2, 2);
        scan_layout.addWidget(self.sp_z_end, 2, 3)
        scan_layout.addWidget(QLabel("Z Step(µm):"), 2, 4);
        scan_layout.addWidget(self.sp_z_step, 2, 5)

        layout.addWidget(scan_group)

        # --- 상단 컨트롤 패널 ---
        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ Homoepi 3D 스캔 시작")
        self.btn_start.clicked.connect(self.start_scan)

        self.btn_stop = QPushButton("⏹ 스캔 정지")
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")

        self.spin_crop = QDoubleSpinBox()
        self.spin_crop.setRange(0, 10);
        self.spin_crop.setValue(1.5);
        self.spin_crop.setPrefix("허공 무시: ")

        self.spin_ignore = QDoubleSpinBox()
        self.spin_ignore.setRange(0, 10);
        self.spin_ignore.setValue(1.0);
        self.spin_ignore.setPrefix("표면 직후 무시: ")

        # [신규] 노이즈 하한 설정
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setRange(0, 5.0);
        self.spin_noise.setSingleStep(0.1);
        self.spin_noise.setValue(0.5);
        self.spin_noise.setPrefix("노이즈 하한: ")

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.spin_crop)
        ctrl_layout.addWidget(self.spin_ignore)
        ctrl_layout.addWidget(self.spin_noise)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # 🚨 [신규] 저장 전용 버튼 패널
        save_layout = QHBoxLayout()

        self.btn_save_raw = QPushButton("💾 1. Raw 3D 스펙트럼 전체 저장")
        self.btn_save_raw.clicked.connect(self.save_raw_data)

        self.chk_stokes = QCheckBox("Stokes(0~1500cm⁻¹) 영역만")
        self.chk_stokes.setChecked(True)

        self.btn_save_map = QPushButton("🗺️ 2. 두께 Map 결과 저장 (CSV)")
        self.btn_save_map.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_save_map.clicked.connect(self.save_map_csv)

        self.btn_save_prof = QPushButton("📉 3. 클릭된 픽셀 Profile 저장 (CSV)")
        self.btn_save_prof.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_save_prof.clicked.connect(self.save_profile_csv)

        save_layout.addWidget(self.btn_save_raw)
        save_layout.addWidget(self.chk_stokes)
        save_layout.addWidget(QLabel("  |  "))
        save_layout.addWidget(self.btn_save_map)
        save_layout.addWidget(self.btn_save_prof)
        save_layout.addStretch()
        layout.addLayout(save_layout)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # --- 하단 맵 & 프로파일 뷰어 ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)
        self.map_plot = pg.PlotWidget(title="Epi Thickness Map (µm)")
        self.hist = pg.HistogramLUTWidget()
        self.hist.gradient.loadPreset('thermal')

        map_content = QHBoxLayout()
        map_content.addWidget(self.map_plot)
        map_content.addWidget(self.hist)
        map_layout.addLayout(map_content)

        self.img_item = pg.ImageItem()
        self.map_plot.addItem(self.img_item)
        self.hist.setImageItem(self.img_item)
        self.map_plot.scene().sigMouseClicked.connect(self.on_map_click)
        splitter.addWidget(map_widget)

        self.prof_plot = pg.PlotWidget(title="Z-Depth Profile at Selected Point")
        splitter.addWidget(self.prof_plot)

        layout.addWidget(splitter, stretch=1)

    def start_scan(self):
        self.depth_profiles.clear()
        self.img_item.clear()

        # 상태 초기화
        self.selected_x = None
        self.selected_y = None
        self.prof_plot.clear()

        crop_d = self.spin_crop.value()
        ignore_d = self.spin_ignore.value()
        noise_floor = self.spin_noise.value()  # 노이즈 하한값 워커에 전달 가능

        self.worker = HomoepiWorker(
            self.main_window,
            self.sp_x_start.value(), self.sp_x_end.value(), self.sp_x_step.value(),
            self.sp_y_start.value(), self.sp_y_end.value(), self.sp_y_step.value(),
            self.sp_z_start.value(), self.sp_z_end.value(), self.sp_z_step.value(),
            crop_dist=crop_d, ignore_dist=ignore_d
        )

        # 만약 Worker에서 noise_floor를 받도록 업데이트했다면 파라미터로 넘겨주세요.
        # 예: self.worker.noise_floor = noise_floor

        self.worker.progress.connect(self.progress.setValue)
        self.worker.update_map.connect(self.handle_new_data)

        self.btn_start.setEnabled(False)
        self.btn_start.setText("⏳ 3D 스캔 진행 중...")
        self.btn_stop.setEnabled(True)

        self.worker.start()

        if self.main_window.control_panel.chk_auto_update.isChecked():
            self.main_window.control_panel.update_timer.stop()

    def stop_scan(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.btn_start.setText("🛑 모터 정지 및 복귀 중...")

            if self.main_window.control_panel.chk_auto_update.isChecked():
                self.main_window.control_panel.update_timer.start(100)

    def scan_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("▶ Homoepi 3D 스캔 시작")
        self.btn_stop.setEnabled(False)

        if self.main_window.control_panel.chk_auto_update.isChecked():
            self.main_window.control_panel.update_timer.start(100)

    def handle_new_data(self, x, y, thickness, raw_data):
        self.depth_profiles[(x, y)] = {"thickness": thickness, "data": raw_data}

        keys = list(self.depth_profiles.keys())
        self.xs = sorted(list(set(k[0] for k in keys)))
        self.ys = sorted(list(set(k[1] for k in keys)))

        grid = np.zeros((len(self.ys), len(self.xs)))

        for (kx, ky), val in self.depth_profiles.items():
            ix = self.xs.index(kx)
            iy = self.ys.index(ky)
            grid[iy, ix] = val["thickness"]

        self.img_item.setImage(grid)

    def on_map_click(self, ev):
        if not hasattr(self, 'xs') or not hasattr(self, 'ys') or not self.depth_profiles:
            return

        pos = self.img_item.mapFromScene(ev.scenePos())
        ix, iy = int(pos.x()), int(pos.y())

        if 0 <= ix < len(self.xs) and 0 <= iy < len(self.ys):
            self.selected_x = self.xs[ix]
            self.selected_y = self.ys[iy]
            x_val, y_val = self.selected_x, self.selected_y

            if (x_val, y_val) in self.depth_profiles:
                pt_data = self.depth_profiles[(x_val, y_val)]
                raw = pt_data["data"]

                z_arr = raw["z"]
                spectra_arr = raw["spectra"]
                waves = raw["waves"]

                # 1. 신호 연산
                mask_776 = (waves >= 760) & (waves <= 790)
                area_776 = np.sum(spectra_arr[:, mask_776], axis=1)

                mask_intf = (waves >= 950) & (waves <= 1000)
                waves_intf = waves[mask_intf]

                spec_intf = spectra_arr[:, mask_intf]
                spec_intf_bg = spec_intf - np.min(spec_intf, axis=1, keepdims=True)
                weighted_spec = spec_intf_bg ** 2
                peak_weighted_com = np.sum(weighted_spec * waves_intf, axis=1) / (np.sum(weighted_spec, axis=1) + 1e-9)

                # 정규화
                norm_area = (area_776 - np.min(area_776)) / (np.ptp(area_776) + 1e-9)
                raw_peak = peak_weighted_com
                norm_peak = (raw_peak - np.min(raw_peak)) / (np.ptp(raw_peak) + 1e-9)

                self.prof_plot.clear()
                self.prof_plot.addLegend(offset=(10, 10))

                self.prof_plot.plot(z_arr, norm_area, pen=None, symbol='s', symbolSize=6, symbolBrush='b',
                                    name="Surface (776cm⁻¹)")

                try:
                    # 표면 피팅
                    bounds_s = ([0, 0.8, min(z_arr), 0.01], [0.2, 1.2, max(z_arr), 5.0])
                    popt_s, _ = curve_fit(sigmoid, z_arr, norm_area, p0=[0, 1, np.mean(z_arr), 0.5], bounds=bounds_s,
                                          maxfev=1000)
                    z_surf = popt_s[2]
                    self.prof_plot.plot(z_arr, sigmoid(z_arr, *popt_s), pen=pg.mkPen(color='b', width=2),
                                        name="Surface Fit")

                    # 🚨 [신규] 계면 피팅 (Air 마스킹 + interface_sigmoid)
                    valid_mask = norm_area > 0.1  # 허공 노이즈 버리기
                    z_intf = z_surf
                    popt_i = None
                    thickness = 0.0

                    if np.sum(valid_mask) >= 5:
                        z_valid = z_arr[valid_mask]
                        peak_valid = raw_peak[valid_mask]

                        bg_guess = np.median(peak_valid[-3:])
                        epi_guess = np.median(peak_valid[:3])
                        amp_guess = epi_guess - bg_guess

                        if abs(amp_guess) >= self.spin_noise.value():
                            p0_i = [bg_guess, amp_guess, z_surf + 1.0, 0.5]
                            amp_bounds = (3.0, 30.0) if amp_guess > 0 else (-30.0, -3.0)
                            bounds_i = (
                                [bg_guess - 5.0, amp_bounds[0], z_surf, 0.05],
                                [bg_guess + 5.0, amp_bounds[1], max(z_valid), 2.0]
                            )
                            popt_i, _ = curve_fit(interface_sigmoid, z_valid, peak_valid, p0=p0_i, bounds=bounds_i,
                                                  maxfev=2000)
                            z_intf = popt_i[2]
                            thickness = abs(z_intf - z_surf)

                            # 계면 포인트 및 피팅선 그리기 (시각화를 위해 정규화)
                            fit_curve_raw = interface_sigmoid(z_valid, *popt_i)
                            fit_curve_norm = (fit_curve_raw - np.min(raw_peak)) / (np.ptp(raw_peak) + 1e-9)
                            self.prof_plot.plot(z_arr, norm_peak, pen=None, symbol='o', symbolSize=6, symbolBrush='g',
                                                name="Interface Pos")
                            self.prof_plot.plot(z_valid, fit_curve_norm, pen=pg.mkPen(color='r', width=2),
                                                name="Interface Fit")

                    self.prof_plot.addItem(
                        pg.InfiniteLine(z_surf, angle=90, pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine)))
                    if thickness > 0:
                        self.prof_plot.addItem(
                            pg.InfiniteLine(z_intf, angle=90, pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine)))

                    # 저장을 위해 연산된 파라미터 저장
                    pt_data['popt_i'] = popt_i
                    pt_data['zs'] = z_surf
                    pt_data['zi'] = z_intf
                    pt_data['norm_area'] = norm_area
                    pt_data['peak'] = raw_peak
                    # 실시간 맵의 색상도 더 정확한 피팅 결과로 업데이트
                    pt_data['thickness'] = thickness

                    info_text = f"📍 X: {x_val} µm, Y: {y_val} µm\n🎯 Thickness: {thickness:.3f} µm"

                except Exception as e:
                    info_text = f"📍 X: {x_val} µm, Y: {y_val} µm\n🎯 Thickness: Error"
                    print(f"Fit Error: {e}")

                text_item = pg.TextItem(info_text, color='k', anchor=(0, 0))
                self.prof_plot.addItem(text_item)
                text_item.setPos(z_arr[0], 1.0)

                # --- 저장 기능 1. 기존 Raw 스펙트럼 데이터 ---

    def save_raw_data(self):
        if not self.depth_profiles:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Raw Data", "", "CSV (*.csv)")
        if not path: return

        sample_data = next(iter(self.depth_profiles.values()))["data"]
        waves = sample_data["waves"]

        if self.chk_stokes.isChecked():
            valid_wave_mask = (waves >= 0) & (waves <= 1500)
            waves_to_save = waves[valid_wave_mask]
        else:
            valid_wave_mask = np.ones(len(waves), dtype=bool)
            waves_to_save = waves

        rows = []
        headers = ["X(um)", "Y(um)", "Z(um)"] + [f"{w:.2f}" for w in waves_to_save]

        for (x, y), val in self.depth_profiles.items():
            z_arr = val["data"]["z"]
            spectra_arr = val["data"]["spectra"]

            for i, z in enumerate(z_arr):
                spec_to_save = spectra_arr[i, valid_wave_mask]
                row = [x, y, z] + spec_to_save.tolist()
                rows.append(row)

        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(path, index=False)
        print(f"✅ Raw Data saved to {path}.")

    # --- 저장 기능 2. 두께 Map 결과 평탄화 저장 ---
    def save_map_csv(self):
        if not self.depth_profiles:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Map CSV", "live_thickness_map.csv", "CSV Files (*.csv)")
        if not path:
            return

        export_data = []
        for (x, y), val in self.depth_profiles.items():
            export_data.append({
                'X (um)': x,
                'Y (um)': y,
                'Thickness (um)': val.get('thickness', 0),
                'Surface Z (um)': val.get('zs', 0),
                'Interface Z (um)': val.get('zi', 0)
            })

        df_export = pd.DataFrame(export_data)
        df_export.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"✅ Map Saved: {path}")

    # --- 저장 기능 3. 현재 픽셀 Profile 저장 ---
    def save_profile_csv(self):
        if self.selected_x is None or self.selected_y is None:
            print("먼저 맵에서 픽셀을 클릭하여 프로파일을 띄워주세요.")
            return

        data = self.depth_profiles.get((self.selected_x, self.selected_y))
        if not data:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Profile CSV",
                                              f"live_profile_X{self.selected_x}_Y{self.selected_y}.csv",
                                              "CSV Files (*.csv)")
        if not path:
            return

        raw = data["data"]
        df_export = pd.DataFrame({
            'Z_Depth (um)': raw['z'],
            'Norm_Area (0-1)': data.get('norm_area', np.nan),
            'Interface_Peak_Pos (cm-1)': data.get('peak', np.nan)
        })

        if data.get('popt_i') is not None and data.get('thickness', 0) > 0:
            df_export['Interface_Fit_Curve'] = interface_sigmoid(raw['z'], *data['popt_i'])
        else:
            df_export['Interface_Fit_Curve'] = np.nan

        df_export.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"✅ Profile Saved: {path}")


class LibrarySearchWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.library_db = {}  # 라이브러리 데이터를 담을 딕셔너리 { "Si": (x_array, y_array) }
        self.target_data = None  # 분석할 타겟 스펙트럼 데이터 (x_array, y_array)

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # --- 왼쪽: 제어부 및 결과 랭킹 ---
        left_panel = QGroupBox("Library Search Control")
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(350)

        # 1. 파일 로드 버튼
        self.btn_load_target = QPushButton("📁 분석할 스펙트럼 불러오기 (.csv)")
        self.btn_load_target.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; height: 35px;")
        self.btn_load_target.clicked.connect(self.load_target_spectrum)

        self.btn_load_library = QPushButton("📚 라이브러리 폴더 지정하기")
        self.btn_load_library.clicked.connect(self.load_library_folder)

        self.lbl_lib_status = QLabel("현재 로드된 라이브러리: 0개")
        self.lbl_lib_status.setStyleSheet("color: #ff9800;")

        # 2. 매칭 실행 버튼
        self.btn_run_search = QPushButton("🔍 자동 매칭 분석 시작")
        self.btn_run_search.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 40px;")
        self.btn_run_search.clicked.connect(self.run_search)

        # 3. 결과 리스트
        self.list_results = QListWidget()
        self.list_results.itemClicked.connect(self.on_result_clicked)

        left_layout.addWidget(self.btn_load_target)
        left_layout.addWidget(self.btn_load_library)
        left_layout.addWidget(self.lbl_lib_status)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.btn_run_search)
        left_layout.addWidget(QLabel(
            "<b>매칭 결과 랭킹 (유사도 %):</b><br><span style='color:gray; font-size:11px;'>항목을 클릭하면 그래프에 겹쳐서 표시됩니다.</span>"))
        left_layout.addWidget(self.list_results)

        # --- 오른쪽: 비교 시각화 그래프 ---
        self.plot_compare = pg.PlotWidget(title="Spectrum Analysis & Comparison")
        self.plot_compare.setLabel('bottom', "Raman Shift", units="cm⁻¹")
        self.plot_compare.setLabel('left', "Normalized Intensity")
        self.plot_compare.addLegend()

        # 타겟 라인(흰색, 두껍게)과 라이브러리 매칭 라인(초록색)
        self.target_line = self.plot_compare.plot(pen=pg.mkPen('w', width=2), name="Target Spectrum")
        self.match_line = self.plot_compare.plot(pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine),
                                                 name="Library Match")

        layout.addWidget(left_panel)
        layout.addWidget(self.plot_compare, stretch=1)

    # --- 기능 함수 구현 ---
    def _read_csv(self, file_path):
        """CSV 파일을 읽어 X, Y 배열을 반환합니다. (텍스트 헤더/메타데이터 자동 무시)"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue

                    # 쉼표(,) 또는 탭(\t) 기준으로 분리
                    parts = line.split(',')
                    if len(parts) < 2:
                        parts = line.split('\t')

                    if len(parts) >= 2:
                        try:
                            # 데이터를 숫자로 변환 시도
                            x_val = float(parts[0])
                            y_val = float(parts[1])
                            data.append([x_val, y_val])
                        except ValueError:
                            # 변환 실패 (텍스트 헤더나 빈 칸인 경우) -> 에러 띄우지 않고 자연스럽게 무시
                            pass

            if len(data) > 0:
                data = np.array(data)
                x, y = data[:, 0], data[:, 1]

                # 형태(Shape) 비교를 위해 최소값 0, 최대값 1로 자동 정규화
                y_min = np.min(y)
                y_norm = y - y_min
                y_max = np.max(y_norm)
                if y_max > 0:
                    y_norm = y_norm / y_max
                return x, y_norm
            else:
                print(f"[Library] 에러: 파일에 유효한 숫자 데이터가 없습니다. ({file_path})")
                return None, None

        except Exception as e:
            print(f"[Library] 파일 읽기 에러 ({file_path}): {e}")
            return None, None

    def load_target_spectrum(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Target Spectrum", "",
                                                   "CSV Files (*.csv);;Text Files (*.txt)")
        if file_path:
            x, y = self._read_csv(file_path)
            if x is not None:
                self.target_data = (x, y)
                self.target_line.setData(x, y)
                self.match_line.setData([], [])  # 이전 매칭 지우기
                self.list_results.clear()
                self.plot_compare.setTitle(f"Target: {os.path.basename(file_path)}")

    def load_library_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Library Folder")
        if folder_path:
            self.library_db.clear()
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv') or file_name.endswith('.txt'):
                    path = os.path.join(folder_path, file_name)
                    x, y = self._read_csv(path)
                    if x is not None:
                        mat_name = os.path.splitext(file_name)[0]
                        self.library_db[mat_name] = (x, y)

            self.lbl_lib_status.setText(f"현재 로드된 라이브러리: {len(self.library_db)}개")

    def run_search(self):
        if self.target_data is None:
            QMessageBox.warning(self, "알림", "먼저 분석할 타겟 스펙트럼을 불러오세요.")
            return
        if not self.library_db:
            QMessageBox.warning(self, "알림", "라이브러리 데이터가 없습니다. 폴더를 지정해주세요.")
            return

        tx, ty = self.target_data
        results = []

        # 💡 핵심 로직: 코사인 유사도(Cosine Similarity) 연산
        for mat_name, (lx, ly) in self.library_db.items():
            # 1. 라이브러리 X축을 타겟의 X축 간격에 맞게 보간(Interpolation)
            sort_idx = np.argsort(lx)
            lx_sorted, ly_sorted = lx[sort_idx], ly[sort_idx]
            ly_interp = np.interp(tx, lx_sorted, ly_sorted)

            # 2. 내적 계산
            dot_product = np.dot(ty, ly_interp)
            norm_target = np.linalg.norm(ty)
            norm_lib = np.linalg.norm(ly_interp)

            similarity = 0.0
            if norm_target > 0 and norm_lib > 0:
                similarity = dot_product / (norm_target * norm_lib)

            results.append((mat_name, similarity * 100))

        # 유사도가 높은 순으로 정렬
        results.sort(key=lambda item: item[1], reverse=True)

        self.list_results.clear()
        for i, (name, score) in enumerate(results):
            # 글자색 및 아이콘 배치용
            icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🔹"
            item = QListWidgetItem(f"{icon} {i + 1}위: {name} (일치율: {score:.1f}%)")
            item.setData(Qt.ItemDataRole.UserRole, name)  # 숨겨진 데이터로 이름 저장
            self.list_results.addItem(item)

            # 1위 항목은 텍스트를 초록색으로 강조
            if i == 0 and score > 80.0:
                item.setForeground(QBrush(QColor("#4CAF50")))

        # 매칭이 완료되면 1위 결과를 그래프에 바로 표시
        if self.list_results.count() > 0:
            self.on_result_clicked(self.list_results.item(0))

    def on_result_clicked(self, item):
        """리스트 항목 클릭 시 해당 라이브러리를 초록 점선으로 그래프에 오버레이"""
        mat_name = item.data(Qt.ItemDataRole.UserRole)
        if mat_name in self.library_db:
            lx, ly = self.library_db[mat_name]
            self.match_line.setData(lx, ly)
            self.plot_compare.setTitle(f"Target vs {mat_name}")

# -------------------------------------------------------------------
# 4. 제어 패널 (사이드바)
# -------------------------------------------------------------------
class ControlPanelWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setMinimumWidth(450)

        self.active_pairs = []
        self.pair_counter = 1
        self.colors = ['#00BFFF', '#FF8C00', '#32CD32', '#FF1493']

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # [1. Hardware Connection]
        conn_group = QGroupBox("Hardware Connection")
        conn_layout = QGridLayout()

        self.btn_cam_connect = QPushButton("카메라 연결")
        self.btn_stage_connect = QPushButton("스테이지 연결")

        self.combo_com_port = QComboBox()
        available_ports = [port.device for port in serial.tools.list_ports.comports()]
        if available_ports:
            self.combo_com_port.addItems(available_ports)
        else:
            self.combo_com_port.addItem("COM3")

        self.btn_find_index = QPushButton("Find Index (영점)")
        self.btn_find_index.setStyleSheet("background-color: #FFE082; font-weight: bold;")
        self.btn_find_index.clicked.connect(self.main_window.run_find_index)
        self.btn_find_index.setEnabled(False)

        conn_layout.addWidget(self.btn_cam_connect, 0, 0, 1, 3)
        conn_layout.addWidget(QLabel("Stage Port:"), 1, 0)
        conn_layout.addWidget(self.combo_com_port, 1, 1, 1, 2)
        conn_layout.addWidget(self.btn_stage_connect, 2, 0, 1, 2)
        conn_layout.addWidget(self.btn_find_index, 2, 2, 1, 1)

        self.btn_cam_connect.clicked.connect(self.main_window.toggle_camera)
        self.btn_stage_connect.clicked.connect(self.main_window.toggle_stage)
        conn_group.setLayout(conn_layout)
        scroll_layout.addWidget(conn_group)

        # [2. Camera Settings]
        cam_group = QGroupBox("Camera Settings")
        cam_layout = QFormLayout()

        self.chk_cooler = QCheckBox("Sensor Cooler (자동 켜기)")
        self.chk_cooler.setChecked(True)
        self.chk_cooler.stateChanged.connect(lambda state: self.main_window.cam.set_cooler(bool(state)))

        self.lbl_sensor_temp = QLabel("현재 온도: -- ℃")
        self.lbl_sensor_temp.setStyleSheet("color: #00e5ff; font-weight: bold; margin-left: 10px;")

        cooler_layout = QHBoxLayout()
        cooler_layout.addWidget(self.chk_cooler)
        cooler_layout.addWidget(self.lbl_sensor_temp)
        cooler_layout.addStretch()

        self.sensor_mode_combo = QComboBox()
        self.sensor_mode_combo.addItems(["AREA (1)", "PROGRESSIVE (12)", "PHOTON RESOLVING (18)"])
        self.exposure_input = QLineEdit("0.01")
        self.binning_combo = QComboBox()
        self.binning_combo.addItems(["1x1", "2x2", "4x4"])

        roi_layout = QHBoxLayout()
        self.roi_start = QLineEdit("1144")
        self.roi_height = QLineEdit("8")
        roi_layout.addWidget(QLabel("Y:"));
        roi_layout.addWidget(self.roi_start)
        roi_layout.addWidget(QLabel("H:"));
        roi_layout.addWidget(self.roi_height)

        cam_layout.addRow("센서 쿨러:", cooler_layout)
        cam_layout.addRow("센서 모드:", self.sensor_mode_combo)
        cam_layout.addRow("노출 시간(s):", self.exposure_input)
        cam_layout.addRow("비닝 (Binning):", self.binning_combo)
        cam_layout.addRow("ROI 범위:", roi_layout)

        self.btn_apply_cam = QPushButton("카메라 설정 적용")
        self.btn_apply_cam.clicked.connect(self.apply_camera_settings)
        cam_layout.addRow(self.btn_apply_cam)
        cam_group.setLayout(cam_layout)
        scroll_layout.addWidget(cam_group)

        self.cooler_timer = QTimer()
        self.cooler_timer.timeout.connect(self.update_sensor_temperature)
        self.cooler_timer.start(2000)

        # [3. Calibration Settings]
        calib_group = QGroupBox("Calibration Settings")
        calib_layout = QGridLayout()
        self.btn_auto_si = QPushButton("Si 피크 자동 보정 (520 cm⁻¹)")
        self.btn_auto_si.setStyleSheet("background-color: #009688; color: white; font-weight: bold;")
        self.btn_auto_si.clicked.connect(self.apply_auto_si_calibration)
        calib_layout.addWidget(self.btn_auto_si, 0, 0, 1, 2)
        calib_layout.addWidget(QLabel("Excitation Laser (nm):"), 1, 0)
        self.spin_laser = QDoubleSpinBox();
        self.spin_laser.setRange(200, 1500);
        self.spin_laser.setValue(532.0)
        calib_layout.addWidget(self.spin_laser, 1, 1)

        self.px1 = QDoubleSpinBox();
        self.px1.setRange(0, 4096);
        self.px1.setDecimals(1);
        self.px1.setValue(2244)
        self.wl1 = QDoubleSpinBox();
        self.wl1.setRange(200, 1500);
        self.wl1.setDecimals(2);
        self.wl1.setValue(532)
        calib_layout.addWidget(self.px1, 3, 0);
        calib_layout.addWidget(self.wl1, 3, 1)

        self.px2 = QDoubleSpinBox();
        self.px2.setRange(0, 4096);
        self.px2.setDecimals(1);
        self.px2.setValue(1649)
        self.wl2 = QDoubleSpinBox();
        self.wl2.setRange(200, 1500);
        self.wl2.setDecimals(2);
        self.wl2.setValue(547.14)
        calib_layout.addWidget(self.px2, 4, 0);
        calib_layout.addWidget(self.wl2, 4, 1)

        self.px3 = QDoubleSpinBox();
        self.px3.setRange(0, 4096);
        self.px3.setDecimals(1);
        self.px3.setValue(2805)
        self.wl3 = QDoubleSpinBox();
        self.wl3.setRange(200, 1500);
        self.wl3.setDecimals(2);
        self.wl3.setValue(517.68)
        calib_layout.addWidget(self.px3, 5, 0);
        calib_layout.addWidget(self.wl3, 5, 1)

        self.btn_apply_calib = QPushButton("3-Point 다항 보정 적용")
        self.btn_apply_calib.setStyleSheet("background-color: #3F51B5; color: white;")
        self.btn_apply_calib.clicked.connect(self.apply_3pt_calibration)
        calib_layout.addWidget(self.btn_apply_calib, 6, 0, 1, 2)
        calib_group.setLayout(calib_layout)
        scroll_layout.addWidget(calib_group)

        # [4. Temperature Analysis Manager]
        peak_group = QGroupBox("Temperature Analysis Manager")
        self.peak_layout = QVBoxLayout()
        self.btn_add_pair = QPushButton("➕ 온도 측정 쌍 추가")
        self.btn_add_pair.clicked.connect(self.add_temp_pair)
        self.peak_layout.addWidget(self.btn_add_pair)
        self.cards_layout = QVBoxLayout()
        self.peak_layout.addLayout(self.cards_layout)
        peak_group.setLayout(self.peak_layout)
        scroll_layout.addWidget(peak_group)

        # ==========================================
        # 🚨 [5. NEW Piezo Stage Control 통합 패널] 🚨
        # ==========================================
        stage_master_group = QGroupBox("Piezo Stage Control")
        stage_master_layout = QVBoxLayout()

        # --- A. Current Position ---
        pos_group = QGroupBox("Current Position & Zero")
        pos_layout = QGridLayout()

        self.chk_auto_update = QCheckBox("Auto Update (10Hz)")
        self.chk_auto_update.setChecked(True)
        self.chk_auto_update.stateChanged.connect(self.toggle_auto_update)

        self.btn_refresh = QPushButton("🔄 수동 업데이트")
        self.btn_refresh.setStyleSheet("background-color: #BBDEFB; font-weight: bold;")
        self.btn_refresh.clicked.connect(self.manual_update_indicator)

        self.chk_absolute = QCheckBox("Show Absolute (Hardware) Position")

        self.lbl_x = QLabel("X: 0.000 µm")
        self.lbl_y = QLabel("Y: 0.000 µm")
        self.lbl_z = QLabel("Z: 0.000 µm")
        font = self.lbl_x.font();
        font.setPointSize(12);
        font.setBold(True)
        self.lbl_x.setFont(font);
        self.lbl_y.setFont(font);
        self.lbl_z.setFont(font)

        self.btn_set_zero = QPushButton("SET ZERO (현재 위치를 상대 영점으로)")
        self.btn_set_zero.setStyleSheet("background-color: #C8E6C9;")
        self.btn_set_zero.clicked.connect(self.set_zero)

        pos_layout.addWidget(self.chk_auto_update, 0, 0)
        pos_layout.addWidget(self.btn_refresh, 0, 1)
        pos_layout.addWidget(self.chk_absolute, 1, 0, 1, 2)
        pos_layout.addWidget(self.lbl_x, 2, 0, 1, 2)
        pos_layout.addWidget(self.lbl_y, 3, 0, 1, 2)
        pos_layout.addWidget(self.lbl_z, 4, 0, 1, 2)
        pos_layout.addWidget(self.btn_set_zero, 5, 0, 1, 2)
        pos_group.setLayout(pos_layout)
        stage_master_layout.addWidget(pos_group)

        # --- B. Go To ---
        go_group = QGroupBox("Saved / Target Position (Go To)")
        go_layout = QGridLayout()

        self.radio_go_rel = QRadioButton("Relative")
        self.radio_go_abs = QRadioButton("Absolute")
        self.radio_go_abs.setChecked(True)

        self.input_go_x = QDoubleSpinBox();
        self.input_go_x.setRange(-25000, 25000);
        self.input_go_x.setDecimals(3)
        self.input_go_y = QDoubleSpinBox();
        self.input_go_y.setRange(-25000, 25000);
        self.input_go_y.setDecimals(3)
        self.input_go_z = QDoubleSpinBox();
        self.input_go_z.setRange(-25000, 25000);
        self.input_go_z.setDecimals(3)

        self.btn_go = QPushButton("GO")
        self.btn_go.setStyleSheet("background-color: #AED581; font-weight: bold; height: 30px;")
        self.btn_go.clicked.connect(self.go_target)

        go_layout.addWidget(self.radio_go_abs, 0, 0)
        go_layout.addWidget(self.radio_go_rel, 0, 1)
        go_layout.addWidget(QLabel("X:"), 1, 0);
        go_layout.addWidget(self.input_go_x, 1, 1)
        go_layout.addWidget(QLabel("Y:"), 2, 0);
        go_layout.addWidget(self.input_go_y, 2, 1)
        go_layout.addWidget(QLabel("Z:"), 3, 0);
        go_layout.addWidget(self.input_go_z, 3, 1)
        go_layout.addWidget(self.btn_go, 4, 0, 1, 2)
        go_group.setLayout(go_layout)
        stage_master_layout.addWidget(go_group)

        # --- C. Motion Control ---
        ctrl_group = QGroupBox("Motion Control (Speed & Mode)")
        ctrl_layout = QGridLayout()

        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 8000)
        self.spin_speed.setValue(1000)
        self.spin_speed.setSuffix(" µm/s")
        self.btn_apply_speed = QPushButton("Apply Speed")
        self.btn_apply_speed.clicked.connect(self.apply_speed)

        self.radio_jog_step = QRadioButton("Stepping")
        self.radio_jog_cont = QRadioButton("Continuous")
        self.radio_jog_step.setChecked(True)

        self.spin_step = QDoubleSpinBox()
        self.spin_step.setRange(0.001, 1000)
        self.spin_step.setValue(1.0)
        self.spin_step.setSuffix(" µm")

        ctrl_layout.addWidget(QLabel("Speed:"), 0, 0)
        ctrl_layout.addWidget(self.spin_speed, 0, 1)
        ctrl_layout.addWidget(self.btn_apply_speed, 0, 2)
        ctrl_layout.addWidget(self.radio_jog_step, 1, 0)
        ctrl_layout.addWidget(self.radio_jog_cont, 1, 1)
        ctrl_layout.addWidget(QLabel("Move Step:"), 2, 0)
        ctrl_layout.addWidget(self.spin_step, 2, 1)
        ctrl_group.setLayout(ctrl_layout)
        stage_master_layout.addWidget(ctrl_group)

        # --- D. Jog ---
        jog_group = QGroupBox("Relative Motion (Jog)")
        jog_layout = QGridLayout()

        self.btn_y_plus = QPushButton("∧\n+y")
        self.btn_y_minus = QPushButton("∨\n-y")
        self.btn_x_minus = QPushButton("<\n-x")
        self.btn_x_plus = QPushButton(">\n+x")
        self.btn_z_plus = QPushButton("∧\n(+z)")
        self.btn_z_minus = QPushButton("∨\n(-z)")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("background-color: #EF9A9A; font-weight: bold;")
        self.btn_stop.clicked.connect(lambda: self.main_window.stage.stop_motion())

        self.setup_jog_btn(self.btn_y_plus, 0, 1, 0)
        self.setup_jog_btn(self.btn_y_minus, 0, -1, 0)
        self.setup_jog_btn(self.btn_x_plus, 1, 0, 0)
        self.setup_jog_btn(self.btn_x_minus, -1, 0, 0)
        self.setup_jog_btn(self.btn_z_plus, 0, 0, 1)
        self.setup_jog_btn(self.btn_z_minus, 0, 0, -1)

        jog_layout.addWidget(self.btn_y_plus, 0, 1)
        jog_layout.addWidget(self.btn_x_minus, 1, 0)
        jog_layout.addWidget(self.btn_x_plus, 1, 2)
        jog_layout.addWidget(self.btn_y_minus, 2, 1)
        jog_layout.addWidget(self.btn_z_plus, 0, 3)
        jog_layout.addWidget(self.btn_z_minus, 2, 3)
        jog_layout.addWidget(self.btn_stop, 3, 0, 1, 4)
        jog_group.setLayout(jog_layout)
        stage_master_layout.addWidget(jog_group)

        stage_master_group.setLayout(stage_master_layout)
        scroll_layout.addWidget(stage_master_group)

        # [6. Live View Start]
        self.btn_start_meas = QPushButton("▶ 라이브 뷰/스펙트럼 모니터링 시작")
        self.btn_start_meas.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 35px;")
        self.btn_start_meas.clicked.connect(self.main_window.toggle_measurement)
        scroll_layout.addWidget(self.btn_start_meas)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # --- 백그라운드 위치 타이머 설정 ---
        self._is_updating = False
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.auto_update_indicator)
        self.update_timer.start(100)

    # ---------------- Piezo UI Action Methods ----------------
    def toggle_auto_update(self, state):
        if state == Qt.CheckState.Checked.value:
            self.update_timer.start(100)
        else:
            self.update_timer.stop()

    def auto_update_indicator(self):
        if getattr(self, '_is_updating', False): return
        self._is_updating = True
        try:
            self._update_ui_labels()
        except Exception as e:
            pass
        finally:
            self._is_updating = False

    def manual_update_indicator(self):
        if not self.main_window.stage.is_connected:
            QMessageBox.warning(self, "오류", "스테이지가 연결되어 있지 않습니다.")
            return
        self._update_ui_labels()
        print("[UI] 위치 수동 업데이트 완료")

    def _update_ui_labels(self):
        if self.main_window.stage.is_connected:
            log_x, log_y, log_z = self.main_window.stage.read_position()

            if self.chk_absolute.isChecked():
                self.lbl_x.setText(f"X: {self.main_window.stage.hard_x:.3f} µm (Abs)")
                self.lbl_y.setText(f"Y: {self.main_window.stage.hard_y:.3f} µm (Abs)")
                self.lbl_z.setText(f"Z: {self.main_window.stage.hard_z:.3f} µm (Abs)")
            else:
                self.lbl_x.setText(f"X: {log_x:.3f} µm")
                self.lbl_y.setText(f"Y: {log_y:.3f} µm")
                self.lbl_z.setText(f"Z: {log_z:.3f} µm")

    def set_zero(self):
        if self.main_window.stage.is_connected:
            # 🚨 1. 통신 충돌 방지를 위해 자동 업데이트 타이머 임시 정지
            timer_was_active = False
            if hasattr(self, 'update_timer') and self.update_timer.isActive():
                self.update_timer.stop()
                timer_was_active = True

            # 2. 영점 설정 및 UI 즉시 갱신
            self.main_window.stage.set_zero()
            self._update_ui_labels()

            # 🚨 3. 영점 설정 완료 후 타이머 원상복구
            if timer_was_active:
                self.update_timer.start(100)

    def apply_speed(self):
        if self.main_window.stage.is_connected:
            self.main_window.stage.set_speed(self.spin_speed.value())

    def go_target(self):
        if not self.main_window.stage.is_connected: return
        x = self.input_go_x.value()
        y = self.input_go_y.value()
        z = self.input_go_z.value()

        if self.radio_go_abs.isChecked():
            self.main_window.stage.move_absolute(x, y, z)
        else:
            self.main_window.stage.move_relative(x, y, z)

        QTimer.singleShot(100, self._update_ui_labels)

    def setup_jog_btn(self, btn, dx, dy, dz):
        btn.pressed.connect(lambda: self.jog_start(dx, dy, dz))
        btn.released.connect(self.jog_stop)

    def jog_start(self, dx, dy, dz):
        if not self.main_window.stage.is_connected: return

        if self.radio_jog_step.isChecked():
            step = self.spin_step.value()
            self.main_window.stage.move_relative(dx * step, dy * step, dz * step)
            QTimer.singleShot(100, self._update_ui_labels)
        else:
            large_step = 10000.0
            self.main_window.stage.move_relative(dx * large_step, dy * large_step, dz * large_step)

    def jog_stop(self):
        if not self.main_window.stage.is_connected: return
        if self.radio_jog_cont.isChecked():
            self.main_window.stage.stop_motion()
            QTimer.singleShot(200, self._update_ui_labels)

    # ---------------- Camera & General Methods ----------------
    def apply_camera_settings(self):
        cam = self.main_window.cam
        if not cam.is_connected: return
        try:
            cam.set_exposure_time(float(self.exposure_input.text()))
            cam.set_binning(int(self.binning_combo.currentText()[0]))
            cam.set_roi(int(self.roi_start.text()), int(self.roi_height.text()))
        except:
            pass

    def apply_auto_si_calibration(self):
        from scipy.signal import find_peaks
        try:
            spectrum = getattr(self.main_window.spectrum_view, 'current_y_data', None)
            if spectrum is None or len(spectrum) == 0: return

            mid_idx = len(spectrum) // 2
            margin = 50
            left_region = spectrum[:mid_idx - margin]
            right_region = spectrum[mid_idx + margin:]

            prom_left = np.max(left_region) * 0.05
            peaks_left, _ = find_peaks(left_region, prominence=prom_left)

            prom_right = np.max(right_region) * 0.05
            peaks_right, _ = find_peaks(right_region, prominence=prom_right)

            if len(peaks_left) == 0 or len(peaks_right) == 0: return

            px_left_si = peaks_left[-1]
            px_right_si = peaks_right[0] + mid_idx + margin

            if spectrum[px_left_si] > spectrum[px_right_si]:
                px_stokes = px_left_si
                px_anti_stokes = px_right_si
            else:
                px_stokes = px_right_si
                px_anti_stokes = px_left_si

            shift_diff = 520.45 - (-520.45)
            px_diff = px_stokes - px_anti_stokes

            if px_diff == 0: return
            slope = shift_diff / px_diff
            intercept = 520.45 - (slope * px_stokes)

            new_x_axis = np.arange(len(spectrum)) * slope + intercept
            self.main_window.spectrum_view.x_axis = new_x_axis
            self.main_window.spectrum_view.is_calibrated = True
            self.main_window.spectrum_view.calibration_func = lambda px: px * slope + intercept
        except Exception as e:
            print(f"Auto Calib Error: {e}")

    def apply_3pt_calibration(self):
        laser_wl = self.spin_laser.value()
        pixels = [self.px1.value(), self.px2.value(), self.px3.value()]
        wavelengths = [self.wl1.value(), self.wl2.value(), self.wl3.value()]
        try:
            self.main_window.spectrum_view.apply_calibration_quadratic(pixels, wavelengths, laser_wl)
            self.main_window.statusBar().showMessage("3-Point 캘리브레이션 적용 완료!")
        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", f"캘리브레이션 실패:\n{str(e)}")

    def add_temp_pair(self):
        colors = ['#00e5ff', '#ff9800', '#00e676', '#e040fb', '#ffeb3b']
        new_id = len(self.active_pairs) + 1
        assigned_color = colors[(new_id - 1) % len(colors)]
        target_layout = self.main_window.spectrum_view.pair_control_layout
        pair = TempPair(new_id, self.main_window, target_layout, color_hex=assigned_color)
        self.active_pairs.append(pair)
        self.cards_layout.addWidget(pair.lbl_temp)

    def remove_temp_pair(self, pair):
        self.active_pairs.remove(pair)
        pair.remove_from_plot()

    def update_sensor_temperature(self):
        if self.main_window.cam.is_connected:
            temp = self.main_window.cam.get_temperature()
            if temp is not None:
                color = "#00e5ff" if temp < 0 else "#ff9800"
                self.lbl_sensor_temp.setStyleSheet(f"color: {color}; font-weight: bold; margin-left: 10px;")
                self.lbl_sensor_temp.setText(f"현재 온도: {temp:.1f} ℃")
            else:
                self.lbl_sensor_temp.setText("현재 온도: 읽기 오류")
        else:
            self.lbl_sensor_temp.setStyleSheet("color: #888888; margin-left: 10px;")
            self.lbl_sensor_temp.setText("현재 온도: -- ℃ (연결 안 됨)")

# -------------------------------------------------------------------
# 4. 메인 윈도우 (호스트)
# -------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raman Thermography Pro (Real Hardware Mode)")
        self.resize(1400, 850)

        self.cam = CameraController()
        self.stage = PiezoController(port='COM4', simulate=False)
        self.is_measuring = False

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tabs = QTabWidget()
        self.spectrum_view = SpectrumViewWidget(self)
        self.live_view = LiveViewWidget(self)
        self.mapping_view = MappingViewWidget(self)
        self.library_view = LibrarySearchWidget(self)

        self.control_panel = ControlPanelWidget(self)
        splitter.addWidget(self.control_panel)

        self.temp_log_view = TempLogViewWidget(self)  # 4번째 로깅 탭 생성
        self.homoepi_view = HomoepiViewWidget(self)

        self.tabs.addTab(self.live_view, "📷 2D Live View")
        self.tabs.addTab(self.spectrum_view, "📈 1D Spectrum & Temp")
        self.tabs.addTab(self.mapping_view, "🗺️ Mapping & Scan")
        self.tabs.addTab(self.homoepi_view, "💎 Homoepi (Z-Depth)")
        self.tabs.addTab(self.temp_log_view, "⏱️ Real-time Temp Log")
        self.tabs.addTab(self.library_view, "Material Analysis")# 탭 등록

        splitter.addWidget(self.tabs)
        splitter.setSizes([450, 950])
        main_layout.addWidget(splitter)

    def toggle_camera(self):
        if not self.cam.is_connected:
            self.cam.initialize_dcam()
            if self.cam.connect_first_available_camera():
                self.control_panel.btn_cam_connect.setText("카메라 연결 해제")
                # 카메라가 연결되자마자 UI의 체크박스 상태를 읽어 쿨러 자동 가동
                cooler_state = self.control_panel.chk_cooler.isChecked()
                self.cam.set_cooler(cooler_state)
            else:
                self.control_panel.btn_cam_connect.setText("카메라(가상) 해제")
        else:
            if self.is_measuring: self.toggle_measurement()
            self.cam.disconnect()
            self.control_panel.btn_cam_connect.setText("카메라 연결")

    def toggle_stage(self):
        if not self.stage.is_connected:
            selected_port = self.control_panel.combo_com_port.currentText()
            if self.stage.connect(port_name=selected_port):
                self.control_panel.btn_stage_connect.setText("스테이지 연결 해제")
                self.control_panel.btn_find_index.setEnabled(True)
                self.control_panel.apply_speed()

                if not self.stage.check_is_indexed():
                    reply = QMessageBox.question(
                        self, 'Hardware Index Not Found',
                        '장비의 기계적 절대 영점(Index)이 설정되어 있지 않습니다.\n정확한 맵핑과 이동을 위해 지금 영점(INdex) 찾기를 수행하시겠습니까?',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.run_find_index()
            else:
                QMessageBox.warning(self, "Error", f"{selected_port} 연결 실패")
        else:
            self.stage.disconnect()
            self.control_panel.btn_stage_connect.setText("스테이지 연결")
            self.control_panel.btn_find_index.setEnabled(False)

    def toggle_measurement(self):
        if not self.is_measuring:
            self.live_view.start_live()
            self.control_panel.btn_start_meas.setText("⏹ 모니터링 정지")
            self.control_panel.btn_start_meas.setStyleSheet(
                "background-color: #f44336; color: white; font-weight: bold;")
            self.is_measuring = True
        else:
            self.live_view.stop_live()
            self.control_panel.btn_start_meas.setText("▶ 라이브 뷰/스펙트럼 모니터링 시작")
            self.control_panel.btn_start_meas.setStyleSheet(
                "background-color: #4CAF50; color: white; font-weight: bold;")
            self.is_measuring = False

    def closeEvent(self, event):
        self.live_view.stop_live()

        # [수정된 부분] 이전 시뮬레이션 타이머 확인 로직을 지우고,
        # 새로운 MappingWorker 스레드가 돌고 있다면 안전하게 중단 및 대기하도록 변경
        if hasattr(self.mapping_view, 'mapping_worker') and self.mapping_view.mapping_worker.isRunning():
            self.mapping_view.stop_simulation()

        self.cam.uninitialize_dcam()
        self.stage.disconnect()
        event.accept()

    def run_find_index(self):
        if self.stage.is_connected:
            # 🚨 1. 하드웨어가 영점을 찾는 동안 통신 과부하 방지
            timer_was_active = False
            if self.control_panel.chk_auto_update.isChecked():
                self.control_panel.update_timer.stop()
                timer_was_active = True

            # 2. 하드웨어 영점(INdex) 탐색 명령 전송
            self.stage.find_index()

            QMessageBox.information(
                self,
                "진행 중",
                "하드웨어 영점(Index) 탐색을 시작했습니다.\n모든 축이 기준점으로 이동할 때까지 잠시 기다려주세요."
            )

            # 🚨 3. 탐색 명령이 안전하게 들어간 후 타이머 복구
            if timer_was_active:
                self.control_panel.update_timer.start(100)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())