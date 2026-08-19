"""PyQt6 Phase 1 live alignment window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from camera_controller import CameraController

from .analyzer import AlignmentConfig, detect_strongest_peak
from .profile import HardwareProfile
from .session import AlignmentSession
from .stability import MetricHistory
from .workers import AcquisitionWorker, AnalysisWorker


pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)


def _spinbox(minimum: float, maximum: float, value: float, decimals: int = 2) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    return widget


def _format(value: float, precision: int = 3, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "INVALID"
    return f"{value:.{precision}f}{suffix}"


class AlignmentWindow(QMainWindow):
    def __init__(self, profile: HardwareProfile):
        super().__init__()
        self.profile = profile
        self.camera = CameraController()
        self.camera_api_initialized = False
        self.acquisition_worker: AcquisitionWorker | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.dark_reference: np.ndarray | None = None
        self._dark_average: np.ndarray | None = None
        self._dark_target = 0
        self._dark_count = 0
        self.last_raw_frame: np.ndarray | None = None
        self.last_result = None
        self.last_frame_shape: tuple[int, int] | None = None
        self.history = MetricHistory(maximum_samples=600)
        self.session = AlignmentSession(profile)

        self.setWindowTitle("Raman Alignment Monitor — Phase 1")
        self.resize(1700, 980)
        self._build_ui()
        self._connect_controls()
        self.analysis_worker = AnalysisWorker(self._current_config())
        self.analysis_worker.result_ready.connect(self._on_analysis_result)
        self.analysis_worker.analysis_error.connect(self._on_analysis_error)
        self.analysis_worker.start()
        self._update_profile_text()

        self.temperature_timer = QTimer(self)
        self.temperature_timer.timeout.connect(self._update_temperature)
        self.temperature_timer.start(1000)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        self.status_banner = QLabel("STOPPED — Connect the camera or start simulation")
        self.status_banner.setStyleSheet(
            "background:#2b2b2b;color:#f0f0f0;padding:7px;font-weight:bold;border-radius:4px;"
        )
        outer.addWidget(self.status_banner)

        horizontal = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(horizontal, stretch=1)
        horizontal.addWidget(self._build_control_panel())
        horizontal.addWidget(self._build_monitor_panel())
        horizontal.setSizes([340, 1360])

    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        content = QWidget()
        layout = QVBoxLayout(content)

        profile_group = QGroupBox("Hardware profile")
        profile_layout = QVBoxLayout(profile_group)
        self.profile_label = QLabel()
        self.profile_label.setWordWrap(True)
        profile_layout.addWidget(self.profile_label)
        layout.addWidget(profile_group)

        camera_group = QGroupBox("Camera / acquisition")
        camera_layout = QFormLayout(camera_group)
        self.btn_connect = QPushButton("Connect ORCA-Quest 2")
        self.btn_live = QPushButton("Start live")
        self.btn_live.setStyleSheet("background:#2e7d32;color:white;font-weight:bold;padding:6px;")
        self.exposure_ms = _spinbox(0.01, 1_800_000.0, 10.0, 3)
        self.binning = QComboBox()
        self.binning.addItems(["1", "2", "4"])
        self.output_bits = QComboBox()
        self.output_bits.addItems(["16", "12", "8"])
        self.roi_y = QSpinBox()
        self.roi_y.setRange(0, self.profile.sensor_height_pixels - 1)
        self.roi_y.setValue(0)
        self.roi_height = QSpinBox()
        self.roi_height.setRange(4, self.profile.sensor_height_pixels)
        self.roi_height.setValue(self.profile.sensor_height_pixels)
        self.btn_apply_camera = QPushButton("Apply camera settings")
        self.temperature_label = QLabel("Sensor: -- °C")
        camera_layout.addRow(self.btn_connect)
        camera_layout.addRow(self.btn_live)
        camera_layout.addRow("Exposure (ms)", self.exposure_ms)
        camera_layout.addRow("Binning", self.binning)
        camera_layout.addRow("Digital output (bit)", self.output_bits)
        camera_layout.addRow("Vertical ROI start", self.roi_y)
        camera_layout.addRow("Vertical ROI height", self.roi_height)
        camera_layout.addRow(self.btn_apply_camera)
        camera_layout.addRow(self.temperature_label)
        layout.addWidget(camera_group)

        reference_group = QGroupBox("Si reference / analysis")
        reference_layout = QFormLayout(reference_group)
        self.reference_label = QLabel(
            f"{self.profile.reference_name}: {self.profile.reference_raman_shift_cm1:.1f} cm⁻¹"
        )
        self.reference_label.setWordWrap(True)
        self.peak_center = _spinbox(0.0, 100_000.0, self.profile.sensor_width_pixels / 2, 2)
        self.peak_half_window = QSpinBox()
        self.peak_half_window.setRange(12, 600)
        self.peak_half_window.setValue(60)
        self.cm_per_pixel = _spinbox(0.0, 100.0, 0.0, 6)
        self.cm_per_pixel.setSpecialValueText("Not calibrated")
        self.row_step = QSpinBox()
        self.row_step.setRange(1, 64)
        self.row_step.setValue(4)
        self.spatial_minimum_snr = _spinbox(1.0, 100.0, 5.0, 1)
        self.spatial_minimum_snr.setToolTip(
            "A detector column is used only when its vertical maximum exceeds this SNR."
        )
        self.spatial_signal_threshold = _spinbox(0.0, 1_000_000.0, 0.0, 1)
        self.spatial_signal_threshold.setSpecialValueText("SNR only")
        self.spatial_signal_threshold.setToolTip(
            "Optional minimum background-subtracted maximum intensity for each column."
        )
        self.btn_auto_peak = QPushButton("Find strongest peak")
        reference_layout.addRow(self.reference_label)
        reference_layout.addRow("Peak center (px)", self.peak_center)
        reference_layout.addRow("Half window (px)", self.peak_half_window)
        reference_layout.addRow("Raman shift / pixel", self.cm_per_pixel)
        reference_layout.addRow("Sample every Nth row/column", self.row_step)
        reference_layout.addRow("Column maximum minimum SNR", self.spatial_minimum_snr)
        reference_layout.addRow("Column signal threshold", self.spatial_signal_threshold)
        reference_layout.addRow(self.btn_auto_peak)
        layout.addWidget(reference_group)

        dark_group = QGroupBox("Dark / display")
        dark_layout = QFormLayout(dark_group)
        self.btn_dark = QPushButton("Capture 20-frame dark average")
        self.btn_clear_dark = QPushButton("Clear dark")
        self.dark_label = QLabel("Dark: none")
        self.auto_levels = QCheckBox("Auto levels (exploration only)")
        self.level_min = _spinbox(-1_000_000.0, 1_000_000.0, 0.0, 1)
        self.level_max = _spinbox(-1_000_000.0, 1_000_000.0, 3000.0, 1)
        dark_layout.addRow(self.btn_dark)
        dark_layout.addRow(self.btn_clear_dark)
        dark_layout.addRow(self.dark_label)
        dark_layout.addRow(self.auto_levels)
        dark_layout.addRow("Fixed level min", self.level_min)
        dark_layout.addRow("Fixed level max", self.level_max)
        layout.addWidget(dark_group)

        session_group = QGroupBox("Compare / session")
        session_layout = QVBoxLayout(session_group)
        marks = QHBoxLayout()
        self.btn_mark_a = QPushButton("Mark A")
        self.btn_mark_b = QPushButton("Mark B")
        self.btn_mark_best = QPushButton("Set Best")
        marks.addWidget(self.btn_mark_a)
        marks.addWidget(self.btn_mark_b)
        marks.addWidget(self.btn_mark_best)
        self.notes = QTextEdit()
        self.notes.setPlaceholderText("Alignment notes: grating/cylindrical/qCMOS adjustment...")
        self.notes.setMaximumHeight(80)
        self.btn_save_session = QPushButton("Save session")
        self.btn_new_session = QPushButton("New session")
        session_layout.addLayout(marks)
        session_layout.addWidget(self.notes)
        session_layout.addWidget(self.btn_save_session)
        session_layout.addWidget(self.btn_new_session)
        layout.addWidget(session_group)
        layout.addStretch()
        scroll.setWidget(content)
        return scroll

    def _build_monitor_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        top = QSplitter(Qt.Orientation.Horizontal)

        image_group = QGroupBox("2D detector geometry")
        image_layout = QVBoxLayout(image_group)
        self.image_plot = pg.PlotWidget()
        self.image_plot.setLabel("bottom", "Dispersion axis (pixel)")
        self.image_plot.setLabel("left", "Spatial axis (row)")
        self.image_plot.invertY(True)
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)
        self.image_plot.addLegend()
        self.trace_curve = self.image_plot.plot(
            pen=pg.mkPen("#00e5ff", width=2), symbol="o", symbolSize=3,
            symbolBrush="#00e5ff", name="Peak trace x(y)"
        )
        self.spatial_trace_curve = self.image_plot.plot(
            pen=pg.mkPen("#ff4081", width=2), name="Peak-center alignment line"
        )
        self.alignment_peak_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen("#ffffff", width=1), brush=pg.mkBrush("#ff1744"), size=9,
            name="Accepted column maxima"
        )
        self.image_plot.addItem(self.alignment_peak_scatter)
        self.saturation_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 80, 180), size=4
        )
        self.image_plot.addItem(self.saturation_scatter)
        center = self.profile.sensor_width_pixels / 2
        self.peak_region = pg.LinearRegionItem(
            [center - 60, center + 60], orientation=pg.LinearRegionItem.Vertical,
            brush=QColor(255, 193, 7, 35), pen=pg.mkPen("#ffc107", width=1)
        )
        self.image_plot.addItem(self.peak_region)
        image_layout.addWidget(self.image_plot)
        top.addWidget(image_group)

        spectra_splitter = QSplitter(Qt.Orientation.Vertical)
        peak_group = QGroupBox("1D raw / rectified diagnostic / fit")
        peak_layout = QVBoxLayout(peak_group)
        self.peak_plot = pg.PlotWidget()
        self.peak_plot.setLabel("bottom", "Detector pixel")
        self.peak_plot.setLabel("left", "Summed intensity")
        self.peak_plot.showGrid(x=True, y=True, alpha=0.25)
        self.raw_curve = self.peak_plot.plot(pen=pg.mkPen("w", width=2), name="Raw sum")
        self.rectified_curve = self.peak_plot.plot(
            pen=pg.mkPen("#42a5f5", width=1, style=Qt.PenStyle.DashLine), name="Rectified"
        )
        self.fit_curve = self.peak_plot.plot(pen=pg.mkPen("#ffca28", width=2), name="Peak fit")
        self.peak_plot.addLegend()
        peak_layout.addWidget(self.peak_plot)
        spectra_splitter.addWidget(peak_group)

        derivative_group = QGroupBox("-d²I/dx² symmetry inspector")
        derivative_layout = QVBoxLayout(derivative_group)
        self.derivative_plot = pg.PlotWidget()
        self.derivative_plot.setLabel("bottom", "Detector pixel")
        self.derivative_plot.setLabel("left", "-d²I/dx²")
        self.derivative_plot.showGrid(x=True, y=True, alpha=0.25)
        self.derivative_curve = self.derivative_plot.plot(pen=pg.mkPen("#ef5350", width=2))
        self.derivative_plot.addLine(y=0, pen=pg.mkPen("#777777"))
        derivative_layout.addWidget(self.derivative_plot)
        spectra_splitter.addWidget(derivative_group)
        top.addWidget(spectra_splitter)
        top.setSizes([760, 620])
        layout.addWidget(top, stretch=3)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        metrics_group = QGroupBox("Live quantitative metrics")
        metrics_layout = QGridLayout(metrics_group)
        self.metric_labels: dict[str, QLabel] = {}
        metric_names = [
            "Status", "Saturation", "Headroom", "Peak SNR", "Peak area / s",
            "Peak FWHM", "Width symmetry", "Area asymmetry", "Mirror NRMSE",
            "D2 amplitude balance", "D2 position balance", "D2 area balance",
            "Trace tilt", "Trace center drift", "Trace curvature RMS", "Row FWHM",
            "Row FWHM CV", "Projection broadening", "Valid-row ratio",
            "Vertical center", "Vertical FWHM", "Vertical edge margin",
            "Spectrum horizontal tilt", "Spectrum vertical drift", "Spatial fit RMS",
            "Valid column count",
            "Center jitter", "FWHM stability CV", "Area stability CV", "Analysis latency",
        ]
        for index, name in enumerate(metric_names):
            row = index % 12
            column = (index // 12) * 2
            title = QLabel(name)
            value = QLabel("--")
            value.setStyleSheet("color:#00e5ff;font-weight:bold;")
            metrics_layout.addWidget(title, row, column)
            metrics_layout.addWidget(value, row, column + 1)
            self.metric_labels[name] = value
        bottom.addWidget(metrics_group)

        trend_group = QGroupBox("Recent trend")
        trend_layout = QVBoxLayout(trend_group)
        self.trend_metric = QComboBox()
        self.trend_metric.addItems(
            [
                "Peak FWHM (px)", "Width symmetry", "D2 amplitude balance",
                "Trace tilt (px/100 rows)", "Trace curvature RMS (px)",
                "Projection broadening", "Peak area", "Peak SNR",
                "Vertical centroid (px)", "Vertical clipping margin (px)",
                "Spectrum tilt (rows/100 columns)", "Spectrum vertical drift (px)",
            ]
        )
        self.trend_plot = pg.PlotWidget()
        self.trend_plot.setLabel("bottom", "Time (s)")
        self.trend_plot.showGrid(x=True, y=True, alpha=0.25)
        self.trend_curve = self.trend_plot.plot(pen=pg.mkPen("#66bb6a", width=2))
        trend_layout.addWidget(self.trend_metric)
        trend_layout.addWidget(self.trend_plot)
        bottom.addWidget(trend_group)
        bottom.setSizes([850, 510])
        layout.addWidget(bottom, stretch=2)
        return container

    def _connect_controls(self) -> None:
        self.btn_connect.clicked.connect(self._toggle_camera)
        self.btn_live.clicked.connect(self._toggle_live)
        self.btn_apply_camera.clicked.connect(self._apply_camera_settings)
        self.btn_auto_peak.clicked.connect(self._find_peak)
        self.btn_dark.clicked.connect(self._start_dark_capture)
        self.btn_clear_dark.clicked.connect(self._clear_dark)
        self.btn_mark_a.clicked.connect(lambda: self._mark("A"))
        self.btn_mark_b.clicked.connect(lambda: self._mark("B"))
        self.btn_mark_best.clicked.connect(lambda: self._mark("Best"))
        self.btn_save_session.clicked.connect(self._save_session)
        self.btn_new_session.clicked.connect(self._new_session)
        self.peak_center.valueChanged.connect(self._settings_changed)
        self.peak_half_window.valueChanged.connect(self._settings_changed)
        self.cm_per_pixel.valueChanged.connect(self._settings_changed)
        self.row_step.valueChanged.connect(self._settings_changed)
        self.spatial_minimum_snr.valueChanged.connect(self._settings_changed)
        self.spatial_signal_threshold.valueChanged.connect(self._settings_changed)
        self.output_bits.currentTextChanged.connect(self._settings_changed)
        self.peak_region.sigRegionChangeFinished.connect(self._region_changed)
        self.trend_metric.currentTextChanged.connect(self._update_trend)

    # -------------------------------------------------------- configuration
    def _current_config(self) -> AlignmentConfig:
        bit_depth = int(self.output_bits.currentText())
        scale = float(self.cm_per_pixel.value())
        return AlignmentConfig(
            peak_center_pixel=float(self.peak_center.value()),
            peak_half_window_pixels=int(self.peak_half_window.value()),
            dispersion_axis=1,
            detector_maximum=float((1 << bit_depth) - 1),
            saturation_level=0.98,
            minimum_peak_snr=5.0,
            minimum_row_snr=4.0,
            spatial_minimum_snr=float(self.spatial_minimum_snr.value()),
            spatial_signal_threshold=float(self.spatial_signal_threshold.value()),
            minimum_valid_row_ratio=0.15,
            row_step=int(self.row_step.value()),
            reference_raman_shift_cm1=self.profile.reference_raman_shift_cm1,
            raman_shift_per_pixel=scale if scale > 0 else None,
        )

    def _settings_changed(self) -> None:
        center = float(self.peak_center.value())
        half = int(self.peak_half_window.value())
        self.peak_region.setRegion([center - half, center + half])
        if self.analysis_worker is not None:
            self.analysis_worker.update_settings(self._current_config(), self.dark_reference)

    def _region_changed(self) -> None:
        start, stop = self.peak_region.getRegion()
        center = 0.5 * (start + stop)
        half = max(12, int(round(0.5 * (stop - start))))
        self.peak_center.blockSignals(True)
        self.peak_half_window.blockSignals(True)
        self.peak_center.setValue(center)
        self.peak_half_window.setValue(half)
        self.peak_center.blockSignals(False)
        self.peak_half_window.blockSignals(False)
        if self.analysis_worker is not None:
            self.analysis_worker.update_settings(self._current_config(), self.dark_reference)

    def _update_profile_text(self) -> None:
        self.profile_label.setText(
            f"{self.profile.camera_model}\n"
            f"{self.profile.sensor_width_pixels}×{self.profile.sensor_height_pixels}, "
            f"{self.profile.pixel_size_um:.1f} µm pixel\n"
            f"Cylindrical: dispersion {self.profile.cylindrical_dispersion_focal_length_mm:.0f} mm / "
            f"perpendicular {self.profile.cylindrical_spatial_focal_length_mm:.0f} mm\n"
            f"Grating: {self.profile.grating_grooves_per_mm:.0f} grooves/mm, "
            f"{self.profile.grating_blaze_nm:.0f} nm blaze"
        )

    # ------------------------------------------------------------- hardware
    def _toggle_camera(self) -> None:
        if self.camera.is_connected:
            if self.acquisition_worker is not None:
                self._stop_live()
            self.camera.disconnect()
            self.btn_connect.setText("Connect ORCA-Quest 2")
            return
        if not self.camera_api_initialized:
            self.camera_api_initialized = bool(self.camera.initialize_dcam())
        if self.camera_api_initialized and self.camera.connect_first_available_camera():
            self.btn_connect.setText("Disconnect camera")
            self._apply_camera_settings()
            self.status_banner.setText("CONNECTED — Apply settings, capture dark, then start live")
        else:
            QMessageBox.warning(
                self, "Camera connection", "ORCA-Quest 2 connection failed. Simulation remains available."
            )

    def _apply_camera_settings(self) -> None:
        if not self.camera.is_connected:
            return
        if self.acquisition_worker is not None:
            QMessageBox.information(self, "Stop live first", "Stop live capture before changing camera settings.")
            return
        try:
            self.camera.set_exposure_time(self.exposure_ms.value() / 1000.0)
            self.camera.set_binning(int(self.binning.currentText()))
            self.camera.set_roi(self.roi_y.value(), self.roi_height.value())
            self._clear_dark()
            self.status_banner.setText("CAMERA SETTINGS APPLIED — Dark reference was cleared")
        except Exception as exc:
            QMessageBox.critical(self, "Camera settings", str(exc))

    def _toggle_live(self) -> None:
        if self.acquisition_worker is None:
            self._start_live()
        else:
            self._stop_live()

    def _start_live(self) -> None:
        simulate = not self.camera.is_connected
        self.acquisition_worker = AcquisitionWorker(self.camera, simulate=simulate)
        self.acquisition_worker.frame_ready.connect(self._on_frame)
        self.acquisition_worker.status_changed.connect(self._on_acquisition_status)
        self.acquisition_worker.acquisition_error.connect(self._on_acquisition_error)
        self.acquisition_worker.finished.connect(self._acquisition_finished)
        self.acquisition_worker.start()
        self.btn_live.setText("Stop live")
        self.btn_live.setStyleSheet("background:#c62828;color:white;font-weight:bold;padding:6px;")

    def _stop_live(self) -> None:
        worker = self.acquisition_worker
        if worker is None:
            return
        worker.stop()
        if not worker.wait(2500):
            self.status_banner.setText("WAITING — Camera driver has not returned from frame wait yet")
            return
        self.acquisition_worker = None
        self.btn_live.setText("Start live")
        self.btn_live.setStyleSheet("background:#2e7d32;color:white;font-weight:bold;padding:6px;")
        self.status_banner.setText("STOPPED")

    def _acquisition_finished(self) -> None:
        if self.acquisition_worker is not None and not self.acquisition_worker.isRunning():
            self.acquisition_worker = None
            self.btn_live.setText("Start live")
            self.btn_live.setStyleSheet("background:#2e7d32;color:white;font-weight:bold;padding:6px;")

    # -------------------------------------------------------------- analysis
    def _on_frame(self, frame: np.ndarray, timestamp: float) -> None:
        shape = tuple(frame.shape)
        if self.last_frame_shape != shape:
            self.last_frame_shape = shape
            self.dark_reference = None
            self.dark_label.setText("Dark: none (frame shape changed)")
            if self.peak_center.value() >= shape[1]:
                self.peak_center.setValue(0.5 * (shape[1] - 1))
            self.image_plot.setXRange(0, shape[1], padding=0.01)
            self.image_plot.setYRange(0, shape[0], padding=0.01)

        if self._dark_target > 0:
            self._dark_count += 1
            sample = frame.astype(np.float32)
            if self._dark_average is None:
                self._dark_average = sample.copy()
            else:
                self._dark_average += (sample - self._dark_average) / self._dark_count
            self.dark_label.setText(f"Dark: capturing {self._dark_count}/{self._dark_target}")
            if self._dark_count >= self._dark_target:
                self.dark_reference = self._dark_average
                self._dark_target = 0
                self.dark_label.setText(f"Dark: {self._dark_count}-frame average ready")
                if self.analysis_worker is not None:
                    self.analysis_worker.update_settings(self._current_config(), self.dark_reference)

        if self.analysis_worker is not None:
            self.analysis_worker.submit(frame, timestamp)

    def _on_analysis_result(self, raw_frame: np.ndarray, result) -> None:
        self.last_raw_frame = raw_frame
        self.last_result = result
        self.history.append(result)
        self.session.record(result)
        self._update_image(raw_frame, result)
        self._update_spectrum(result)
        self._update_metrics(result)
        self._update_trend()

    def _update_image(self, raw_frame: np.ndarray, result) -> None:
        if self.auto_levels.isChecked():
            self.image_item.setImage(result.corrected_frame, autoLevels=True)
        else:
            low, high = self.level_min.value(), self.level_max.value()
            if high <= low:
                high = low + 1.0
            self.image_item.setImage(result.corrected_frame, autoLevels=False, levels=(low, high))
        self.trace_curve.setData(result.trace.centers, result.trace.sampled_rows)
        self.spatial_trace_curve.setData(
            result.trace.spatial_fit_columns, result.trace.spatial_fit_centers
        )
        self.alignment_peak_scatter.setData(
            result.trace.spatial_sampled_columns, result.trace.spatial_centers
        )

        threshold = result.quality.sensor_maximum * self._current_config().saturation_level
        oriented_raw = raw_frame if self._current_config().dispersion_axis == 1 else raw_frame.T
        flat_indices = np.flatnonzero(oriented_raw.ravel() >= threshold)
        if flat_indices.size:
            flat_indices = flat_indices[:2000]
            rows, columns = np.unravel_index(flat_indices, oriented_raw.shape)
            self.saturation_scatter.setData(columns, rows)
        else:
            self.saturation_scatter.setData([], [])

    def _update_spectrum(self, result) -> None:
        x = np.arange(result.raw_projection.size, dtype=np.float64)
        self.raw_curve.setData(x, result.raw_projection)
        if result.trace.rectified_projection.size == x.size:
            raw_max = float(np.max(result.raw_projection))
            rectified_max = float(np.max(result.trace.rectified_projection))
            scale = raw_max / rectified_max if rectified_max > 0 else 1.0
            self.rectified_curve.setData(x, result.trace.rectified_projection * scale)
        else:
            self.rectified_curve.setData([], [])
        self.fit_curve.setData(result.peak.roi_x, result.peak.fitted_y)
        self.derivative_curve.setData(
            result.peak.roi_x, result.peak.negative_second_derivative
        )

    def _update_metrics(self, result) -> None:
        stability = self.history.stability(30)
        exposure_seconds = max(self.exposure_ms.value() / 1000.0, 1e-12)
        shift_per_pixel = self._current_config().raman_shift_per_pixel
        fwhm_text = _format(result.peak.fwhm, 3, " px")
        if shift_per_pixel is not None and np.isfinite(result.peak.fwhm):
            fwhm_text += f" / {result.peak.fwhm * shift_per_pixel:.3f} cm⁻¹"
        status = "VALID"
        if result.quality.saturated:
            status = "INVALID: SATURATED"
        elif not result.peak.valid:
            status = f"INVALID PEAK: {result.peak.reason}"
        elif not result.trace.valid:
            status = f"INVALID TRACE: {result.trace.reason}"
        values = {
            "Status": status,
            "Saturation": _format(100.0 * result.quality.saturation_fraction, 5, " %"),
            "Headroom": _format(100.0 * result.quality.headroom_fraction, 1, " %"),
            "Peak SNR": _format(result.peak.snr, 1),
            "Peak area / s": _format(result.peak.area / exposure_seconds, 2),
            "Peak FWHM": fwhm_text,
            "Width symmetry": _format(result.peak.width_symmetry, 4),
            "Area asymmetry": _format(result.peak.area_asymmetry, 4),
            "Mirror NRMSE": _format(result.peak.mirror_nrmse, 4),
            "D2 amplitude balance": _format(result.peak.derivative_amplitude_balance, 4),
            "D2 position balance": _format(result.peak.derivative_position_balance, 4),
            "D2 area balance": _format(result.peak.derivative_area_balance, 4),
            "Trace tilt": _format(result.trace.trace_tilt_px_per_100_rows, 4, " px/100 rows"),
            "Trace center drift": _format(result.trace.trace_center_drift_px, 3, " px"),
            "Trace curvature RMS": _format(result.trace.trace_curvature_rms_px, 4, " px"),
            "Row FWHM": _format(result.trace.median_row_fwhm_px, 3, " px"),
            "Row FWHM CV": _format(100.0 * result.trace.row_fwhm_cv, 2, " %"),
            "Projection broadening": _format(100.0 * result.trace.projection_broadening, 2, " %"),
            "Valid-row ratio": _format(100.0 * result.trace.valid_row_ratio, 1, " %"),
            "Vertical center": _format(result.trace.vertical_centroid_px, 2, " px"),
            "Vertical FWHM": _format(result.trace.vertical_fwhm_px, 2, " px"),
            "Vertical edge margin": _format(result.trace.vertical_clipping_margin_px, 1, " px"),
            "Spectrum horizontal tilt": _format(
                result.trace.spectrum_tilt_rows_per_100_columns, 4, " rows/100 cols"
            ),
            "Spectrum vertical drift": _format(result.trace.spectrum_vertical_drift_px, 3, " px"),
            "Spatial fit RMS": _format(result.trace.spectrum_center_residual_rms_px, 3, " px"),
            "Valid column count": str(result.trace.spatial_valid_column_count),
            "Center jitter": _format(stability.center_jitter_rms_px, 4, " px"),
            "FWHM stability CV": _format(100.0 * stability.fwhm_cv, 2, " %"),
            "Area stability CV": _format(100.0 * stability.area_cv, 2, " %"),
            "Analysis latency": _format(result.analysis_ms, 1, " ms"),
        }
        for name, text in values.items():
            self.metric_labels[name].setText(text)
        status_label = self.metric_labels["Status"]
        status_label.setStyleSheet(
            "color:#66bb6a;font-weight:bold;" if status == "VALID" else "color:#ef5350;font-weight:bold;"
        )
        self.status_banner.setText(
            f"{status} — Si {self.profile.reference_raman_shift_cm1:.1f} cm⁻¹ end-to-end reference — "
            f"analysis {result.analysis_ms:.1f} ms"
        )

    def _update_trend(self) -> None:
        times, values = self.history.series(self.trend_metric.currentText())
        self.trend_curve.setData(times, values)
        self.trend_plot.setLabel("left", self.trend_metric.currentText())

    # --------------------------------------------------------------- actions
    def _find_peak(self) -> None:
        if self.last_result is None:
            QMessageBox.information(self, "Peak search", "Start live capture first.")
            return
        center = detect_strongest_peak(self.last_result.raw_projection)
        self.peak_center.setValue(center)

    def _start_dark_capture(self) -> None:
        if self.acquisition_worker is None:
            QMessageBox.information(self, "Dark capture", "Start live capture with the light blocked first.")
            return
        self._dark_target = 20
        self._dark_count = 0
        self._dark_average = None
        self.dark_label.setText("Dark: capturing 0/20")

    def _clear_dark(self) -> None:
        self.dark_reference = None
        self._dark_average = None
        self._dark_target = 0
        self._dark_count = 0
        self.dark_label.setText("Dark: none")
        if self.analysis_worker is not None:
            self.analysis_worker.update_settings(self._current_config(), None)

    def _mark(self, label: str) -> None:
        if self.last_raw_frame is None or self.last_result is None:
            QMessageBox.information(self, "Snapshot", "No analyzed frame is available yet.")
            return
        self.session.mark(label, self.last_raw_frame, self.last_result)
        self.statusBar().showMessage(f"Snapshot {label} marked", 3000)

    def _save_session(self) -> None:
        parent = QFileDialog.getExistingDirectory(self, "Select alignment session output directory")
        if not parent:
            return
        try:
            output = self.session.save(
                Path(parent), self._current_config(), self.dark_reference, self.notes.toPlainText()
            )
            QMessageBox.information(self, "Session saved", str(output))
        except Exception as exc:
            QMessageBox.critical(self, "Session save failed", str(exc))

    def _new_session(self) -> None:
        self.session.clear()
        self.history.clear()
        self.notes.clear()
        self.trend_curve.setData([], [])
        self.statusBar().showMessage("New alignment session started", 3000)

    # -------------------------------------------------------------- feedback
    def _on_acquisition_status(self, message: str) -> None:
        self.status_banner.setText(message)

    def _on_acquisition_error(self, message: str) -> None:
        self.status_banner.setText(f"ACQUISITION ERROR — {message}")

    def _on_analysis_error(self, message: str) -> None:
        self.status_banner.setText(f"ANALYSIS ERROR — {message}")

    def _update_temperature(self) -> None:
        temperature = self.camera.get_temperature() if self.camera.is_connected else None
        self.temperature_label.setText(
            f"Sensor: {temperature:.1f} °C" if temperature is not None else "Sensor: -- °C"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        if self.acquisition_worker is not None:
            self.acquisition_worker.stop()
            self.acquisition_worker.wait(2500)
        if self.analysis_worker is not None:
            self.analysis_worker.stop()
            self.analysis_worker.wait(5000)
        if self.camera.is_connected:
            self.camera.disconnect()
        if self.camera_api_initialized:
            self.camera.uninitialize_dcam()
        event.accept()


__all__ = ["AlignmentWindow"]
