import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar, QSplitter, QDoubleSpinBox, QSpinBox,
                             QCheckBox, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import pyqtgraph as pg


# --- 피팅 함수 정의 ---
def sigmoid(z, A, B, z0, k):
    return A + (B - A) / (1 + np.exp(-(z - z0) / (np.abs(k) + 1e-3)))


def interface_sigmoid(z, bg, amp, z_intf, w):
    return bg + amp / (1 + np.exp((z - z_intf) / (np.abs(w) + 1e-3)))


class AnalysisThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def __init__(self, df, crop_dist, ignore_dist, noise_floor):
        super().__init__()
        self.df = df
        self.crop_dist = crop_dist
        self.ignore_dist = ignore_dist
        self.noise_floor = noise_floor

    def run(self):
        def find_col(k):
            return next((c for c in self.df.columns if k in c.lower()), None)

        x_col, y_col, z_col = find_col('x'), find_col('y'), find_col('z')
        z_idx = self.df.columns.get_loc(z_col)
        waves = np.array([float(w) for w in self.df.columns[z_idx + 1:]])

        x_arr = self.df[x_col].values
        y_arr = self.df[y_col].values
        z_arr = self.df[z_col].values
        spectra_arr = self.df.iloc[:, z_idx + 1:].values

        mask_776 = (waves >= 760) & (waves <= 790)
        mask_intf = (waves >= 950) & (waves <= 1000)
        waves_intf = waves[mask_intf]

        grouped_indices = self.df.groupby([x_col, y_col]).indices
        total = len(grouped_indices)
        thickness_map = {}

        for i, ((x, y), indices) in enumerate(grouped_indices.items()):
            z = z_arr[indices]
            spectra = spectra_arr[indices]

            sort_idx = np.argsort(z)
            z, spectra = z[sort_idx], spectra[sort_idx]
            valid_idx = z >= (z[0] + self.crop_dist)
            z, spectra = z[valid_idx], spectra[valid_idx]

            if len(z) < 3: continue

            # 1. 신호 연산
            area_776 = np.sum(spectra[:, mask_776], axis=1)
            norm_area = (area_776 - np.min(area_776)) / (np.ptp(area_776) + 1e-9)

            spec_intf = spectra[:, mask_intf]
            spec_intf_bg = spec_intf - np.min(spec_intf, axis=1, keepdims=True)
            weighted_spec = spec_intf_bg ** 2
            peak_weighted_com = np.sum(weighted_spec * waves_intf, axis=1) / (np.sum(weighted_spec, axis=1) + 1e-9)

            try:
                # 2. 표면(Surface) 피팅
                bounds_s = ([0, 0.8, min(z), 0.01], [0.2, 1.2, max(z), 5.0])
                popt_s, _ = curve_fit(sigmoid, z, norm_area, p0=[0, 1, np.mean(z), 0.5], bounds=bounds_s, maxfev=1000)
                z_surf = popt_s[2]

                # 표면 위치(z_surf)와 가장 가까운 실제 인덱스를 찾아 해당 위치의 피크값 추출
                idx_surf = np.argmin(np.abs(z - z_surf))
                surf_peak_val = peak_weighted_com[idx_surf]

                # 3. 계면(Interface) 피팅 (Air 마스킹)
                valid_mask = norm_area > 0.1

                if np.sum(valid_mask) < 5:
                    thickness = 0.0
                    z_intf = z_surf
                    popt_i = None
                else:
                    z_valid = z[valid_mask]
                    peak_valid = peak_weighted_com[valid_mask]

                    bg_guess = np.median(peak_valid[-3:])
                    epi_guess = np.median(peak_valid[:3])
                    amp_guess = epi_guess - bg_guess

                    if abs(amp_guess) < self.noise_floor:
                        thickness = 0.0
                        z_intf = z_surf
                        popt_i = None
                    else:
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

                # ====================================================
                # 🚀 4. 계면(z_intf)에서 10um 뒤의 Peak 위치 찾기
                # ====================================================
                if z_intf != 0.0:
                    z_target = z_intf + 10.0  # 3.0에서 10.0으로 변경
                    target_idx = np.argmin(np.abs(z - z_target))
                    # 💡 950~1000cm-1 대역의 무게중심 피크값 배열 사용
                    peak_pos_10um = peak_weighted_com[target_idx]
                    actual_z = z[target_idx]
                else:
                    peak_pos_10um = np.nan
                    actual_z = np.nan
                # ====================================================

            except Exception as e:
                thickness, z_surf, z_intf, surf_peak_val = 0, 0, 0, np.nan
                popt_i = None
                peak_pos_10um = np.nan
                actual_z = np.nan

            # 데이터 추가 저장 (변수명 10um 반영)
            thickness_map[(x, y)] = {
                't': thickness, 'zs': z_surf, 'zi': z_intf, 'z': z,
                'area': area_776, 'peak': peak_weighted_com, 'popt_i': popt_i, 'norm_area': norm_area,
                'surf_peak': surf_peak_val,
                'peak_10um': peak_pos_10um,
                'z_10um': actual_z
            }

            if i % 10 == 0: self.progress.emit(int((i + 1) / total * 100))

        self.progress.emit(100)
        self.finished.emit(thickness_map)


class MapViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SiC Epi Data Analyzer (Dual Map View)")
        self.resize(1300, 800)

        self.current_file_path = None
        self.cached_df = None
        self.data_map = {}
        self.display_xs = []

        self.selected_x = None
        self.selected_y = None

        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        # --- UI 패널 1: 분석 컨트롤 ---
        ctrl_layout = QHBoxLayout()
        self.btn_load = QPushButton("📁 데이터 로드 및 분석")
        self.btn_load.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.btn_load.clicked.connect(self.start_analysis)

        self.spin_crop = QDoubleSpinBox()
        self.spin_crop.setRange(0, 10)
        self.spin_crop.setValue(1.5)
        self.spin_ignore = QDoubleSpinBox()
        self.spin_ignore.setRange(0, 10)
        self.spin_ignore.setValue(1.0)
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setRange(0, 5.0)
        self.spin_noise.setSingleStep(0.1)
        self.spin_noise.setValue(0.5)
        self.spin_edge_crop = QSpinBox()
        self.spin_edge_crop.setRange(0, 20)
        self.spin_edge_crop.setValue(1)
        self.chk_despike = QCheckBox("스파이크 제거(Median)")
        self.chk_despike.setChecked(True)

        self.spin_edge_crop.valueChanged.connect(self.refresh_map)
        self.chk_despike.stateChanged.connect(self.refresh_map)

        ctrl_layout.addWidget(self.btn_load)
        ctrl_layout.addWidget(QLabel(" | 허공 무시:"))
        ctrl_layout.addWidget(self.spin_crop)
        ctrl_layout.addWidget(QLabel("표면 직후 무시:"))
        ctrl_layout.addWidget(self.spin_ignore)
        ctrl_layout.addWidget(QLabel(" | 노이즈 하한:"))
        ctrl_layout.addWidget(self.spin_noise)
        ctrl_layout.addWidget(QLabel(" | 픽셀 자르기:"))
        ctrl_layout.addWidget(self.spin_edge_crop)
        ctrl_layout.addWidget(self.chk_despike)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # --- UI 패널 2: 시각화 모드 및 저장 ---
        save_layout = QHBoxLayout()

        self.combo_map_mode = QComboBox()
        # 🚀 텍스트 10um 반영
        self.combo_map_mode.addItems([
            "🗺️ 모드: 에피층 두께 (Thickness)",
            "🗺️ 모드: 표면 피크 위치 (Surface Peak)",
            "🗺️ 모드: 계면+10µm 피크 (Intf+10µm Peak)"
        ])
        self.combo_map_mode.setStyleSheet("font-weight: bold; padding: 4px;")
        self.combo_map_mode.currentIndexChanged.connect(self.refresh_map)

        self.btn_save_map = QPushButton("💾 전체 Map 데이터 저장 (CSV)")
        self.btn_save_map.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_save_map.clicked.connect(self.save_map_csv)

        self.btn_save_prof = QPushButton("📉 선택된 픽셀 Profile 저장")
        self.btn_save_prof.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_save_prof.clicked.connect(self.save_profile_csv)

        save_layout.addWidget(self.combo_map_mode)
        save_layout.addWidget(QLabel("  |  "))
        save_layout.addWidget(self.btn_save_map)
        save_layout.addWidget(self.btn_save_prof)
        save_layout.addStretch()
        layout.addLayout(save_layout)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # --- 뷰어 영역 ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)
        map_layout.setContentsMargins(0, 0, 0, 0)
        self.map_view = pg.PlotWidget(title="2D Map Visualization")
        self.hist = pg.HistogramLUTWidget()
        self.hist.gradient.loadPreset('thermal')
        map_content = QHBoxLayout()
        map_content.addWidget(self.map_view)
        map_content.addWidget(self.hist)
        map_layout.addLayout(map_content)
        splitter.addWidget(map_widget)

        self.prof_view = pg.PlotWidget(title="Z-Depth Profile")
        splitter.addWidget(self.prof_view)
        layout.addWidget(splitter, stretch=1)

        self.img_item = pg.ImageItem()
        self.map_view.addItem(self.img_item)
        self.hist.setImageItem(self.img_item)
        self.map_view.scene().sigMouseClicked.connect(self.on_click)

        self.txt_info = pg.TextItem("", color='k', anchor=(0, 0))

    def start_analysis(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV")
        if not path: return

        if path == self.current_file_path and self.cached_df is not None:
            df = self.cached_df
        else:
            cache_path = path + ".pkl"
            try:
                if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(path):
                    df = pd.read_pickle(cache_path)
                else:
                    df = pd.read_csv(path, encoding='cp949')
                    df.to_pickle(cache_path)
            except:
                df = pd.read_csv(path, encoding='cp949')

            self.current_file_path = path
            self.cached_df = df

        self.thread = AnalysisThread(df, self.spin_crop.value(), self.spin_ignore.value(), self.spin_noise.value())
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.store_and_draw_map)
        self.thread.start()

    def store_and_draw_map(self, data_map):
        self.data_map = data_map
        self.refresh_map()

    def refresh_map(self):
        if not self.data_map: return
        keys = list(self.data_map.keys())
        self.raw_xs = sorted(list(set(k[0] for k in keys)))
        self.ys = sorted(list(set(k[1] for k in keys)))

        grid = np.zeros((len(self.ys), len(self.raw_xs)))

        current_mode = self.combo_map_mode.currentText()

        if "Thickness" in current_mode:
            self.map_view.setTitle("Epi Thickness Map (µm)")
        elif "Surface Peak" in current_mode:
            self.map_view.setTitle("Surface Peak Position Map (cm⁻¹)")
        else:
            self.map_view.setTitle("Interface + 10µm Peak Map (cm⁻¹)")

        for (x, y), val in self.data_map.items():
            if "Thickness" in current_mode:
                grid[self.ys.index(y), self.raw_xs.index(x)] = val['t']
            elif "Surface Peak" in current_mode:
                grid[self.ys.index(y), self.raw_xs.index(x)] = val.get('surf_peak', 0)
            else:
                # 🚀 10um 데이터 매핑
                grid[self.ys.index(y), self.raw_xs.index(x)] = val.get('peak_10um', 0)

        if self.chk_despike.isChecked():
            grid = median_filter(grid, size=3)

        crop_x = self.spin_edge_crop.value()
        if crop_x > 0 and crop_x * 2 < len(self.raw_xs):
            grid = grid[:, crop_x:-crop_x]
            self.display_xs = self.raw_xs[crop_x:-crop_x]
        else:
            self.display_xs = self.raw_xs

        self.img_item.setImage(grid.T)

    def on_click(self, ev):
        pos = self.img_item.mapFromScene(ev.scenePos())
        ix, iy = int(pos.x()), int(pos.y())

        if 0 <= ix < len(self.display_xs) and 0 <= iy < len(self.ys):
            self.selected_x = self.display_xs[ix]
            self.selected_y = self.ys[iy]

            x_val, y_val = self.selected_x, self.selected_y
            data = self.data_map[(x_val, y_val)]

            self.prof_view.clear()
            self.prof_view.addLegend(offset=(10, 10))
            self.prof_view.addItem(self.txt_info)

            z_arr = data['z']
            raw_peak = data['peak']

            norm_area = data['norm_area']
            norm_peak = (raw_peak - np.min(raw_peak)) / (np.ptp(raw_peak) + 1e-9)

            self.prof_view.plot(z_arr, norm_area, pen=None, symbol='s', symbolSize=6, symbolBrush='b',
                                name="Surface (776cm⁻¹)")

            if data['t'] > 0:
                self.prof_view.plot(z_arr, norm_peak, pen=None, symbol='o', symbolSize=6, symbolBrush='g',
                                    name="Interface Pos")

            try:
                bounds_s = ([0, 0.8, min(z_arr), 0.01], [0.2, 1.2, max(z_arr), 5.0])
                popt_s, _ = curve_fit(sigmoid, z_arr, norm_area, p0=[0, 1, np.mean(z_arr), 0.5], bounds=bounds_s,
                                      maxfev=1000)
                self.prof_view.plot(z_arr, sigmoid(z_arr, *popt_s), pen=pg.mkPen(color='b', width=2),
                                    name="Surface Fit")

                if data.get('popt_i') is not None and data['t'] > 0:
                    valid_mask = norm_area > 0.1
                    z_valid = z_arr[valid_mask]

                    fit_curve_raw = interface_sigmoid(z_valid, *data['popt_i'])
                    fit_curve_norm = (fit_curve_raw - np.min(raw_peak)) / (np.ptp(raw_peak) + 1e-9)

                    self.prof_view.plot(z_valid, fit_curve_norm, pen=pg.mkPen(color='r', width=2), name="Interface Fit")

                self.prof_view.addItem(
                    pg.InfiniteLine(data['zs'], angle=90, pen=pg.mkPen('b', width=2, style=Qt.PenStyle.DashLine)))
                if data['t'] > 0:
                    self.prof_view.addItem(
                        pg.InfiniteLine(data['zi'], angle=90, pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine)))
            except Exception as e:
                print(f"Plotting Error: {e}")

            # 🚀 10um 피크와 추출된 Z좌표 반영
            peak_10um_val = data.get('peak_10um', np.nan)
            z_10um_val = data.get('z_10um', np.nan)

            info_text = (f"📍 X: {x_val}µm, Y: {y_val}µm\n"
                         f"🎯 Thickness: {data['t']:.3f} µm\n"
                         f"✨ Surf Peak: {data.get('surf_peak', np.nan):.2f} cm⁻¹\n"
                         f"🔍 Intf+10µm Peak: {peak_10um_val:.2f} cm⁻¹\n"
                         f"   (📍 Picked at Z = {z_10um_val:.2f} µm)")

            self.txt_info.setText(info_text)
            self.txt_info.setPos(z_arr[0], 1.0)

    def save_map_csv(self):
        if not self.data_map:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Map CSV", "epi_analysis_map.csv", "CSV Files (*.csv)")
        if not path:
            return

        export_data = []
        for (x, y), val in self.data_map.items():
            export_data.append({
                'X (um)': x,
                'Y (um)': y,
                'Thickness (um)': val['t'],
                'Surface Z (um)': val['zs'],
                'Interface Z (um)': val['zi'],
                'Surface Peak Pos (cm-1)': val.get('surf_peak', np.nan),
                'Intf+10um Peak Pos (cm-1)': val.get('peak_10um', np.nan),  # 🚀 10um 반영
                'Intf+10um Z Depth (um)': val.get('z_10um', np.nan)  # 🚀 10um 반영
            })

        df_export = pd.DataFrame(export_data)
        df_export.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"✅ Map Saved: {path}")

    def save_profile_csv(self):
        if self.selected_x is None or self.selected_y is None:
            print("먼저 맵에서 픽셀을 클릭하여 프로파일을 띄워주세요.")
            return

        data = self.data_map.get((self.selected_x, self.selected_y))
        if not data:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Profile CSV",
                                              f"profile_X{self.selected_x}_Y{self.selected_y}.csv", "CSV Files (*.csv)")
        if not path:
            return

        df_export = pd.DataFrame({
            'Z_Depth (um)': data['z'],
            'Surface_776_Area': data['area'],
            'Norm_Area (0-1)': data['norm_area'],
            'Interface_Peak_Pos (cm-1)': data['peak']
        })

        if data.get('popt_i') is not None and data['t'] > 0:
            df_export['Interface_Fit_Curve'] = interface_sigmoid(data['z'], *data['popt_i'])
        else:
            df_export['Interface_Fit_Curve'] = np.nan

        df_export.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"✅ Profile Saved: {path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MapViewer()
    w.show()
    sys.exit(app.exec())