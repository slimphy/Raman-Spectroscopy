"""Standalone GUI test for DFK 37AUX290 vision and Thorlabs CSN210.

Run with:
    python vision_hardware_test.py

The application is intentionally independent from main.py.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCloseEvent,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vision_test_support import (
    CALIBRATION_PROFILES,
    CSN210Worker,
    CalibrationStore,
    CameraWorker,
    IC4Runtime,
    csn210_vendor_app_running,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_PATH = APP_DIR / "vision_calibration.json"


class ImageCanvas(QWidget):
    """Camera display with source-pixel-coordinate measurement overlays."""

    calibration_line_finished = pyqtSignal(float)
    roi_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(720, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._image = QImage()
        self._source_size = (0, 0)
        self._show_crosshair = True
        self._interaction_mode = "none"
        self._dragging = False
        self._cal_start: QPointF | None = None
        self._cal_end: QPointF | None = None
        self._roi_start: QPointF | None = None
        self._roi_end: QPointF | None = None

    @property
    def source_size(self) -> tuple[int, int]:
        return self._source_size

    @property
    def roi(self) -> tuple[float, float, float, float] | None:
        if self._roi_start is None or self._roi_end is None:
            return None
        x0, x1 = sorted((self._roi_start.x(), self._roi_end.x()))
        y0, y1 = sorted((self._roi_start.y(), self._roi_end.y()))
        return x0, y0, x1, y1

    def set_frame(self, rgb_frame: np.ndarray) -> None:
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
            return
        rgb = np.ascontiguousarray(rgb_frame[:, :, :3], dtype=np.uint8)
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        )
        self._image = image.copy()
        self._source_size = (width, height)
        self.update()

    def set_crosshair_visible(self, visible: bool) -> None:
        self._show_crosshair = bool(visible)
        self.update()

    def set_interaction_mode(self, mode: str) -> None:
        if mode not in ("none", "calibration", "roi"):
            raise ValueError(f"Unknown interaction mode: {mode}")
        self._interaction_mode = mode
        self._dragging = False
        if mode == "none":
            self.unsetCursor()
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def clear_roi(self) -> None:
        self._roi_start = None
        self._roi_end = None
        self.roi_changed.emit(None)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming convention
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11151a"))
        if self._image.isNull():
            painter.setPen(QColor("#8f9ba8"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "카메라를 연결하거나 Simulation을 시작하세요",
            )
            return

        image_rect = self._image_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(image_rect, self._image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._show_crosshair:
            self._draw_crosshair(painter, image_rect)
        self._draw_calibration_line(painter)
        self._draw_roi(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton:
            self._dragging = False
            if self._interaction_mode == "calibration":
                self._cal_start = None
                self._cal_end = None
            self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton or self._image.isNull():
            return
        source_point = self._widget_to_source(event.position())
        if source_point is None:
            return
        if self._interaction_mode == "calibration":
            self._cal_start = source_point
            self._cal_end = source_point
            self._dragging = True
        elif self._interaction_mode == "roi":
            self._roi_start = source_point
            self._roi_end = source_point
            self._dragging = True
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._dragging:
            return
        source_point = self._widget_to_source(event.position(), clamp=True)
        if source_point is None:
            return
        if self._interaction_mode == "calibration":
            self._cal_end = source_point
        elif self._interaction_mode == "roi":
            self._roi_end = source_point
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        source_point = self._widget_to_source(event.position(), clamp=True)
        if source_point is None:
            return

        if self._interaction_mode == "calibration" and self._cal_start is not None:
            self._cal_end = source_point
            distance = math.hypot(
                self._cal_end.x() - self._cal_start.x(),
                self._cal_end.y() - self._cal_start.y(),
            )
            if distance >= 2.0:
                self.calibration_line_finished.emit(distance)
        elif self._interaction_mode == "roi" and self._roi_start is not None:
            self._roi_end = source_point
            roi = self.roi
            if roi is not None and (roi[2] - roi[0]) >= 2 and (roi[3] - roi[1]) >= 2:
                self.roi_changed.emit(roi)
            else:
                self.clear_roi()
        self.update()

    def _image_rect(self) -> QRectF:
        if self._image.isNull():
            return QRectF()
        image_width, image_height = self._source_size
        scale = min(self.width() / image_width, self.height() / image_height)
        draw_width = image_width * scale
        draw_height = image_height * scale
        return QRectF(
            (self.width() - draw_width) / 2.0,
            (self.height() - draw_height) / 2.0,
            draw_width,
            draw_height,
        )

    def _widget_to_source(self, point: QPointF, clamp: bool = False) -> QPointF | None:
        image_rect = self._image_rect()
        if image_rect.isEmpty():
            return None
        x, y = point.x(), point.y()
        if clamp:
            x = min(max(x, image_rect.left()), image_rect.right())
            y = min(max(y, image_rect.top()), image_rect.bottom())
        elif not image_rect.contains(point):
            return None
        image_width, image_height = self._source_size
        return QPointF(
            (x - image_rect.left()) * image_width / image_rect.width(),
            (y - image_rect.top()) * image_height / image_rect.height(),
        )

    def _source_to_widget(self, point: QPointF) -> QPointF:
        image_rect = self._image_rect()
        image_width, image_height = self._source_size
        return QPointF(
            image_rect.left() + point.x() * image_rect.width() / image_width,
            image_rect.top() + point.y() * image_rect.height() / image_height,
        )

    def _draw_crosshair(self, painter: QPainter, image_rect: QRectF) -> None:
        center = image_rect.center()
        painter.setPen(QPen(QColor(40, 255, 120, 210), 1.0))
        painter.drawLine(QPointF(image_rect.left(), center.y()), QPointF(image_rect.right(), center.y()))
        painter.drawLine(QPointF(center.x(), image_rect.top()), QPointF(center.x(), image_rect.bottom()))
        painter.setPen(QPen(QColor(40, 255, 120), 2.0))
        painter.drawEllipse(center, 6.0, 6.0)

    def _draw_calibration_line(self, painter: QPainter) -> None:
        if self._cal_start is None or self._cal_end is None:
            return
        start = self._source_to_widget(self._cal_start)
        end = self._source_to_widget(self._cal_end)
        painter.setPen(QPen(QColor("#25d9ff"), 2.0))
        painter.drawLine(start, end)
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        screen_length = math.hypot(dx, dy)
        if screen_length > 0:
            # Fixed-size end caps stay perpendicular even for a diagonal
            # measurement, producing an H-shaped measurement marker.
            cap_half_length = 12.0
            normal = QPointF(
                -dy / screen_length * cap_half_length,
                dx / screen_length * cap_half_length,
            )
            painter.drawLine(start - normal, start + normal)
            painter.drawLine(end - normal, end + normal)
        distance = math.hypot(
            self._cal_end.x() - self._cal_start.x(),
            self._cal_end.y() - self._cal_start.y(),
        )
        self._draw_overlay_text(painter, (start + end) / 2.0, f"{distance:.2f} px", QColor("#25d9ff"))

    def _draw_roi(self, painter: QPainter) -> None:
        roi = self.roi
        if roi is None:
            return
        top_left = self._source_to_widget(QPointF(roi[0], roi[1]))
        bottom_right = self._source_to_widget(QPointF(roi[2], roi[3]))
        rect = QRectF(top_left, bottom_right).normalized()
        painter.setPen(QPen(QColor("#ffd84a"), 2.0))
        painter.setBrush(QColor(255, 216, 74, 30))
        painter.drawRect(rect)
        center = rect.center()
        painter.setBrush(QColor("#ffd84a"))
        painter.drawEllipse(center, 4.0, 4.0)
        self._draw_overlay_text(
            painter,
            QPointF(rect.left(), rect.top()),
            f"ROI {roi[2] - roi[0]:.1f} x {roi[3] - roi[1]:.1f} px",
            QColor("#ffd84a"),
        )

    @staticmethod
    def _draw_overlay_text(painter: QPainter, anchor: QPointF, text: str, color: QColor) -> None:
        painter.save()
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(text).adjusted(-5, -3, 5, 3)
        text_rect.moveBottomLeft(anchor.toPoint() + QPointF(7, -7).toPoint())
        painter.fillRect(text_rect, QColor(0, 0, 0, 175))
        painter.setPen(color)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class VisionHardwareTestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision CCD + CSN210 Hardware Test")
        self.resize(1500, 900)

        self.ic4_runtime = IC4Runtime()
        self.ic4_runtime.initialize()
        self.camera_worker: CameraWorker | None = None
        self.frame_times: deque[float] = deque(maxlen=60)

        self.calibration_store = CalibrationStore(DEFAULT_CALIBRATION_PATH)
        self.current_roi: tuple[float, float, float, float] | None = None

        self.csn_connected = False
        self.csn_snapshot = None
        self.csn_poll_failures = 0
        self.csn_worker = CSN210Worker(parent=self)

        self._build_ui()
        self._connect_signals()
        self._refresh_camera_devices()
        self._refresh_calibration_view()
        self._set_csn_controls(False)

        if self.calibration_store.load_error:
            self.statusBar().showMessage(
                f"Calibration file load warning: {self.calibration_store.load_error}",
                10000,
            )

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = ImageCanvas()
        splitter.addWidget(self.canvas)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(420)
        side_scroll.setMaximumWidth(560)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(10)

        side_layout.addWidget(self._build_camera_group())
        side_layout.addWidget(self._build_calibration_group())
        side_layout.addWidget(self._build_roi_group())
        side_layout.addWidget(self._build_csn_group())
        side_layout.addStretch(1)
        side_scroll.setWidget(side)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1050, 450])
        self.setCentralWidget(splitter)

        self.setStyleSheet(
            """
            QMainWindow { background: #20252b; }
            QGroupBox { font-weight: 600; border: 1px solid #56606a; border-radius: 5px;
                        margin-top: 9px; padding-top: 9px; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }
            QPushButton { min-height: 28px; }
            QPushButton:checked { background: #167b9b; color: white; }
            QLabel#valueLabel { color: #61dafb; font-family: Consolas; }
            QLabel#warningLabel { color: #ff725e; font-weight: 600; }
            """
        )

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Vision camera — DFK 37AUX290")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.camera_combo.setMinimumContentsLength(18)
        self.camera_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.camera_refresh_button = QPushButton("재검색")
        self.camera_refresh_button.setFixedWidth(64)
        row.addWidget(self.camera_combo, 1)
        row.addWidget(self.camera_refresh_button)
        layout.addLayout(row)

        self.camera_connect_button = QPushButton("카메라 연결")
        layout.addWidget(self.camera_connect_button)

        options = QHBoxLayout()
        self.auto_brightness_check = QCheckBox("Auto brightness (노출 + 게인)")
        self.auto_brightness_check.setChecked(True)
        self.auto_brightness_check.setToolTip(
            "카메라의 ExposureAuto와 GainAuto를 Continuous/Off로 전환합니다."
        )
        self.crosshair_check = QCheckBox("중앙 십자선")
        self.crosshair_check.setChecked(True)
        options.addWidget(self.auto_brightness_check)
        options.addWidget(self.crosshair_check)
        layout.addLayout(options)

        self.camera_status_label = self._value_label("Disconnected")
        self.frame_info_label = self._value_label("—")
        form = QFormLayout()
        form.addRow("상태", self.camera_status_label)
        form.addRow("프레임", self.frame_info_label)
        layout.addLayout(form)
        return group

    def _build_calibration_group(self) -> QGroupBox:
        group = QGroupBox("Vision calibration")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(CALIBRATION_PROFILES)
        self.known_length_spin = QDoubleSpinBox()
        self.known_length_spin.setRange(0.001, 1_000_000.0)
        self.known_length_spin.setDecimals(3)
        self.known_length_spin.setValue(100.0)
        self.known_length_spin.setSuffix(" µm")
        form.addRow("Objective", self.profile_combo)
        form.addRow("알고 있는 길이", self.known_length_spin)
        layout.addLayout(form)

        self.calibration_tool_button = QPushButton("선 길이 측정 모드")
        self.calibration_tool_button.setCheckable(True)
        layout.addWidget(self.calibration_tool_button)
        hint = QLabel("표준물의 같은 길이를 드래그할 때마다 샘플이 추가됩니다. 우클릭하면 현재 선을 취소합니다.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aeb8c2;")
        layout.addWidget(hint)

        self.calibration_table = QTableWidget(0, 3)
        self.calibration_table.setHorizontalHeaderLabels(["#", "길이 (px)", "µm/px"])
        self.calibration_table.verticalHeader().setVisible(False)
        self.calibration_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.calibration_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.calibration_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.calibration_table.setMaximumHeight(160)
        layout.addWidget(self.calibration_table)

        self.calibration_average_label = self._value_label("샘플 없음")
        layout.addWidget(self.calibration_average_label)

        buttons = QGridLayout()
        self.remove_sample_button = QPushButton("마지막 샘플 제거")
        self.clear_samples_button = QPushButton("현재 배율 초기화")
        self.reload_calibration_button = QPushButton("파일 다시 불러오기")
        self.save_calibration_button = QPushButton("JSON 저장")
        buttons.addWidget(self.remove_sample_button, 0, 0)
        buttons.addWidget(self.clear_samples_button, 0, 1)
        buttons.addWidget(self.reload_calibration_button, 1, 0)
        buttons.addWidget(self.save_calibration_button, 1, 1)
        layout.addLayout(buttons)

        self.calibration_path_label = QLabel(DEFAULT_CALIBRATION_PATH.name)
        self.calibration_path_label.setToolTip(str(DEFAULT_CALIBRATION_PATH))
        self.calibration_path_label.setWordWrap(True)
        self.calibration_path_label.setStyleSheet("color: #89949f; font-size: 10px;")
        layout.addWidget(self.calibration_path_label)
        return group

    def _build_roi_group(self) -> QGroupBox:
        group = QGroupBox("Mapping ROI")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.roi_tool_button = QPushButton("사각형 영역 선택")
        self.roi_tool_button.setCheckable(True)
        self.clear_roi_button = QPushButton("영역 지우기")
        row.addWidget(self.roi_tool_button)
        row.addWidget(self.clear_roi_button)
        layout.addLayout(row)

        self.roi_center_label = self._value_label("—")
        self.roi_size_px_label = self._value_label("—")
        self.roi_size_um_label = self._value_label("—")
        form = QFormLayout()
        form.addRow("Center (px)", self.roi_center_label)
        form.addRow("X × Y (px)", self.roi_size_px_label)
        form.addRow("X × Y (µm)", self.roi_size_um_label)
        layout.addLayout(form)
        self.copy_roi_button = QPushButton("ROI JSON 복사")
        self.copy_roi_button.setEnabled(False)
        layout.addWidget(self.copy_roi_button)
        return group

    def _build_csn_group(self) -> QGroupBox:
        group = QGroupBox("Thorlabs CSN210 objective changer")
        layout = QVBoxLayout(group)
        self.csn_connect_button = QPushButton("CSN210 연결")
        layout.addWidget(self.csn_connect_button)

        self.csn_status_label = self._value_label("Disconnected")
        self.csn_homed_label = self._value_label("—")
        self.csn_collision_label = QLabel("—")
        self.csn_collision_label.setObjectName("valueLabel")
        self.csn_device_label = self._value_label("—")
        form = QFormLayout()
        form.addRow("현재 위치", self.csn_status_label)
        form.addRow("Homed", self.csn_homed_label)
        form.addRow("Collision", self.csn_collision_label)
        form.addRow("장치", self.csn_device_label)
        layout.addLayout(form)

        row = QGridLayout()
        self.csn_home_button = QPushButton("Home")
        self.csn_position1_button = QPushButton("Position 1 (20X)")
        self.csn_position2_button = QPushButton("Position 2 (100X)")
        self.csn_stop_button = QPushButton("STOP")
        self.csn_stop_button.setStyleSheet("background: #a93232; color: white; font-weight: 700;")
        row.addWidget(self.csn_home_button, 0, 0)
        row.addWidget(self.csn_stop_button, 0, 1)
        row.addWidget(self.csn_position1_button, 1, 0)
        row.addWidget(self.csn_position2_button, 1, 1)
        layout.addLayout(row)

        note = QLabel("안전 순서: 연결 → Home 완료 → Position 1/2. 충돌 검출 후에는 다시 Home 하세요.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #f0b35a;")
        layout.addWidget(note)
        return group

    def _connect_signals(self) -> None:
        self.camera_refresh_button.clicked.connect(self._refresh_camera_devices)
        self.camera_connect_button.clicked.connect(self._toggle_camera)
        self.auto_brightness_check.toggled.connect(self._auto_brightness_toggled)
        self.crosshair_check.toggled.connect(self.canvas.set_crosshair_visible)

        self.profile_combo.currentTextChanged.connect(self._profile_changed)
        self.calibration_tool_button.toggled.connect(self._calibration_tool_toggled)
        self.canvas.calibration_line_finished.connect(self._add_calibration_sample)
        self.remove_sample_button.clicked.connect(self._remove_last_sample)
        self.clear_samples_button.clicked.connect(self._clear_samples)
        self.reload_calibration_button.clicked.connect(self._reload_calibration)
        self.save_calibration_button.clicked.connect(self._save_calibration)

        self.roi_tool_button.toggled.connect(self._roi_tool_toggled)
        self.clear_roi_button.clicked.connect(self.canvas.clear_roi)
        self.canvas.roi_changed.connect(self._on_roi_changed)
        self.copy_roi_button.clicked.connect(self._copy_roi_json)

        self.csn_connect_button.clicked.connect(self._toggle_csn)
        self.csn_home_button.clicked.connect(lambda: self._command_csn("home"))
        self.csn_position1_button.clicked.connect(lambda: self._command_csn("position1"))
        self.csn_position2_button.clicked.connect(lambda: self._command_csn("position2"))
        self.csn_stop_button.clicked.connect(lambda: self._command_csn("stop"))
        self.csn_worker.connection_changed.connect(self._on_csn_connection_changed)
        self.csn_worker.snapshot_ready.connect(self._on_csn_snapshot)
        self.csn_worker.command_started.connect(self._on_csn_command_started)
        self.csn_worker.command_finished.connect(self._on_csn_command_finished)

    @staticmethod
    def _value_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("valueLabel")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    # Camera ---------------------------------------------------------------

    def _refresh_camera_devices(self) -> None:
        if self.camera_worker is not None and self.camera_worker.isRunning():
            return
        self.camera_combo.clear()
        devices = self.ic4_runtime.enumerate_devices()
        preferred_index = -1
        for index, device in enumerate(devices):
            label = f"{device['model']}  |  S/N {device['serial']}"
            self.camera_combo.addItem(
                label,
                {"mode": "ic4", "serial": device["serial"]},
            )
            if "DFK 37AUX290" in device["model"].upper():
                preferred_index = index
        self.camera_combo.addItem(
            "Simulation (장비 없이 GUI 테스트)",
            {"mode": "simulation", "serial": ""},
        )
        if preferred_index >= 0:
            self.camera_combo.setCurrentIndex(preferred_index)
        elif devices:
            self.camera_combo.setCurrentIndex(0)
        else:
            self.camera_combo.setCurrentIndex(self.camera_combo.count() - 1)
            detail = self.ic4_runtime.error or "IC4 장치가 검색되지 않았습니다."
            self.camera_status_label.setText(detail)

    def _toggle_camera(self) -> None:
        if self.camera_worker is not None and self.camera_worker.isRunning():
            self._stop_camera()
            return
        selection = self.camera_combo.currentData()
        if not isinstance(selection, dict):
            return
        self.frame_times.clear()
        worker = CameraWorker(
            selection["mode"],
            selection["serial"],
            auto_brightness=self.auto_brightness_check.isChecked(),
            parent=self,
        )
        worker.frame_ready.connect(self._on_frame)
        worker.connection_changed.connect(self._on_camera_connection_changed)
        worker.acquisition_error.connect(self._on_camera_error)
        worker.configuration_changed.connect(self._on_camera_configuration_changed)
        worker.finished.connect(self._camera_worker_finished)
        self.camera_worker = worker
        self.camera_combo.setEnabled(False)
        self.camera_refresh_button.setEnabled(False)
        self.camera_connect_button.setText("카메라 연결 해제")
        self.camera_status_label.setText("Connecting…")
        worker.start()

    def _stop_camera(self) -> None:
        worker = self.camera_worker
        if worker is None:
            return
        self.camera_connect_button.setEnabled(False)
        worker.request_stop()
        if not worker.wait(1800):
            self.camera_status_label.setText("Disconnecting…")
        self.camera_connect_button.setEnabled(True)

    def _camera_worker_finished(self) -> None:
        self.camera_combo.setEnabled(True)
        self.camera_refresh_button.setEnabled(True)
        self.camera_connect_button.setEnabled(True)
        self.camera_connect_button.setText("카메라 연결")
        worker = self.sender()
        if worker is self.camera_worker:
            self.camera_worker = None
        if hasattr(worker, "deleteLater"):
            worker.deleteLater()

    def _on_camera_connection_changed(self, connected: bool, detail: str) -> None:
        self.camera_status_label.setText(detail)
        if not connected and (self.camera_worker is None or not self.camera_worker.isRunning()):
            self.camera_connect_button.setText("카메라 연결")

    def _on_camera_error(self, message: str) -> None:
        self.camera_status_label.setText(f"Error: {message}")
        self.statusBar().showMessage(f"Camera error: {message}", 15000)

    def _on_camera_configuration_changed(self, success: bool, message: str) -> None:
        if not success:
            self.auto_brightness_check.blockSignals(True)
            self.auto_brightness_check.setChecked(False)
            self.auto_brightness_check.blockSignals(False)
        self.statusBar().showMessage(message, 10000)

    def _auto_brightness_toggled(self, enabled: bool) -> None:
        worker = self.camera_worker
        if worker is not None and worker.isRunning():
            worker.set_auto_brightness(enabled)
        else:
            state = "ON" if enabled else "OFF"
            self.statusBar().showMessage(
                f"Auto brightness {state}: 다음 카메라 연결 시 적용됩니다.",
                5000,
            )

    def _on_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        self.frame_times.append(now)
        self.canvas.set_frame(frame)
        height, width = frame.shape[:2]
        fps = 0.0
        if len(self.frame_times) >= 2:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            if elapsed > 0:
                fps = (len(self.frame_times) - 1) / elapsed
        self.frame_info_label.setText(f"{width} × {height}  |  {fps:.1f} FPS")

    # Calibration ----------------------------------------------------------

    def _profile_changed(self, _profile: str) -> None:
        self._refresh_calibration_view()
        self._update_roi_labels()

    def _calibration_tool_toggled(self, checked: bool) -> None:
        if checked:
            self.roi_tool_button.blockSignals(True)
            self.roi_tool_button.setChecked(False)
            self.roi_tool_button.blockSignals(False)
            self.canvas.set_interaction_mode("calibration")
        elif not self.roi_tool_button.isChecked():
            self.canvas.set_interaction_mode("none")

    def _roi_tool_toggled(self, checked: bool) -> None:
        if checked:
            self.calibration_tool_button.blockSignals(True)
            self.calibration_tool_button.setChecked(False)
            self.calibration_tool_button.blockSignals(False)
            self.canvas.set_interaction_mode("roi")
        elif not self.calibration_tool_button.isChecked():
            self.canvas.set_interaction_mode("none")

    def _add_calibration_sample(self, pixel_length: float) -> None:
        profile = self.profile_combo.currentText()
        try:
            self.calibration_store.add_sample(
                profile,
                self.known_length_spin.value(),
                pixel_length,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        self._refresh_calibration_view()
        self._update_roi_labels()
        self.statusBar().showMessage(
            f"{profile}: calibration sample #{len(self.calibration_store.profiles[profile])} added",
            4000,
        )

    def _remove_last_sample(self) -> None:
        self.calibration_store.remove_last(self.profile_combo.currentText())
        self._refresh_calibration_view()
        self._update_roi_labels()

    def _clear_samples(self) -> None:
        profile = self.profile_combo.currentText()
        if not self.calibration_store.profiles[profile]:
            return
        answer = QMessageBox.question(
            self,
            "Calibration 초기화",
            f"{profile}의 현재 샘플을 모두 지울까요? 저장 버튼을 누르기 전까지 파일은 바뀌지 않습니다.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.calibration_store.clear(profile)
        self._refresh_calibration_view()
        self._update_roi_labels()

    def _reload_calibration(self) -> None:
        self.calibration_store = CalibrationStore(self.calibration_store.path)
        if self.calibration_store.load_error:
            QMessageBox.warning(
                self,
                "Calibration file",
                self.calibration_store.load_error,
            )
        self._refresh_calibration_view()
        self._update_roi_labels()

    def _save_calibration(self) -> None:
        try:
            self.calibration_store.save()
        except OSError as exc:
            QMessageBox.critical(self, "Calibration save failed", str(exc))
            return
        self.statusBar().showMessage(
            f"Saved: {self.calibration_store.path}",
            7000,
        )

    def _refresh_calibration_view(self) -> None:
        profile = self.profile_combo.currentText() or CALIBRATION_PROFILES[0]
        samples = self.calibration_store.profiles[profile]
        self.calibration_table.setRowCount(len(samples))
        for row, sample in enumerate(samples):
            values = (str(row + 1), f"{sample.pixel_length:.3f}", f"{sample.um_per_pixel:.6g}")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.calibration_table.setItem(row, column, item)
        average = self.calibration_store.average_scale(profile)
        if average is None:
            self.calibration_average_label.setText(f"{profile}: 샘플 없음")
        else:
            self.calibration_average_label.setText(
                f"{profile}: 평균 {average:.6g} µm/px  (n={len(samples)})"
            )
        self.remove_sample_button.setEnabled(bool(samples))
        self.clear_samples_button.setEnabled(bool(samples))

    # ROI ------------------------------------------------------------------

    def _on_roi_changed(self, roi) -> None:
        self.current_roi = tuple(roi) if roi is not None else None
        self._update_roi_labels()

    def _update_roi_labels(self) -> None:
        if self.current_roi is None:
            self.roi_center_label.setText("—")
            self.roi_size_px_label.setText("—")
            self.roi_size_um_label.setText("—")
            self.copy_roi_button.setEnabled(False)
            return
        x0, y0, x1, y1 = self.current_roi
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        width = x1 - x0
        height = y1 - y0
        self.roi_center_label.setText(f"({center_x:.2f}, {center_y:.2f})")
        self.roi_size_px_label.setText(f"{width:.2f} × {height:.2f}")
        scale = self.calibration_store.average_scale(self.profile_combo.currentText())
        if scale is None:
            self.roi_size_um_label.setText("캘리브레이션 필요")
        else:
            self.roi_size_um_label.setText(f"{width * scale:.3f} × {height * scale:.3f}")
        self.copy_roi_button.setEnabled(True)

    def _roi_payload(self) -> dict | None:
        if self.current_roi is None:
            return None
        x0, y0, x1, y1 = self.current_roi
        profile = self.profile_combo.currentText()
        scale = self.calibration_store.average_scale(profile)
        payload = {
            "coordinate_system": "camera_source_pixels",
            "profile": profile,
            "top_left_px": {"x": x0, "y": y0},
            "bottom_right_px": {"x": x1, "y": y1},
            "center_px": {"x": (x0 + x1) / 2.0, "y": (y0 + y1) / 2.0},
            "size_px": {"x": x1 - x0, "y": y1 - y0},
            "um_per_pixel": scale,
        }
        if scale is not None:
            payload["size_um"] = {"x": (x1 - x0) * scale, "y": (y1 - y0) * scale}
        return payload

    def _copy_roi_json(self) -> None:
        payload = self._roi_payload()
        if payload is None:
            return
        QApplication.clipboard().setText(json.dumps(payload, indent=2, ensure_ascii=False))
        self.statusBar().showMessage("ROI JSON을 클립보드에 복사했습니다.", 4000)

    # CSN210 ---------------------------------------------------------------

    def _toggle_csn(self) -> None:
        if self.csn_connected:
            return
        if csn210_vendor_app_running():
            QMessageBox.warning(
                self,
                "CSN210 is already in use",
                "Thorlabs CSN210_Control.exe가 실행 중입니다. 제조사 프로그램에서 장치 연결을 해제하고 "
                "프로그램을 닫은 뒤 다시 시도하세요.",
            )
            return
        self.csn_connect_button.setEnabled(False)
        self.csn_status_label.setText("Connecting in background…")
        self.csn_worker.request("connect")

    def _command_csn(self, command: str) -> None:
        if not self.csn_connected:
            return
        if command == "position1":
            self.profile_combo.setCurrentText("20X")
        elif command == "position2":
            self.profile_combo.setCurrentText("100X")
        self._set_csn_controls(False)
        self.csn_worker.request(command)

    def _on_csn_connection_changed(self, connected: bool, info: object) -> None:
        self.csn_connected = connected
        self.csn_connect_button.setEnabled(True)
        if connected:
            details = info if isinstance(info, dict) else {}
            self.csn_connect_button.setText("CSN210 연결됨")
            self.csn_connect_button.setEnabled(False)
            self.csn_device_label.setText(
                f"S/N {details.get('serial') or '?'} | FW {details.get('firmware') or '?'} | "
                f"{details.get('count', '?')} device(s)"
            )
            self.csn_status_label.setText("Connected — reading status…")
            self._set_csn_controls(True)
        else:
            self.csn_snapshot = None
            self.csn_status_label.setText("Disconnected")
            self.csn_homed_label.setText("—")
            self.csn_collision_label.setText("—")
            self.csn_device_label.setText("—")
            self.csn_connect_button.setText("CSN210 연결")
            self._set_csn_controls(False)

    def _on_csn_command_started(self, command: str) -> None:
        labels = {
            "connect": "Connecting in background…",
            "disconnect": "Disconnecting…",
            "home": "Home command…",
            "position1": "Position 1 command…",
            "position2": "Position 2 command…",
            "stop": "STOP command…",
        }
        if command in labels:
            self.csn_status_label.setText(labels[command])

    def _on_csn_command_finished(self, command: str, success: bool, message: str) -> None:
        if success:
            if command in ("home", "position1", "position2", "stop"):
                self.csn_status_label.setText(f"{command} command sent — reading status…")
            return
        if command == "poll":
            self.csn_poll_failures += 1
            self.csn_status_label.setText(f"Read error: {message}")
            if self.csn_poll_failures >= 3:
                self._set_csn_controls(False)
            return
        self.csn_connect_button.setEnabled(True)
        self._set_csn_controls(self.csn_connected)
        QMessageBox.critical(self, "CSN210 command failed", message)

    def _on_csn_snapshot(self, snapshot) -> None:
        if not self.csn_connected:
            return
        self.csn_poll_failures = 0

        self.csn_snapshot = snapshot
        self.csn_status_label.setText(snapshot.position_text)
        self.csn_homed_label.setText("Yes" if snapshot.homed else "No — Home required")
        self.csn_collision_label.setText("DETECTED — Home required" if snapshot.collision else "No")
        self.csn_collision_label.setObjectName("warningLabel" if snapshot.collision else "valueLabel")
        self.csn_collision_label.style().unpolish(self.csn_collision_label)
        self.csn_collision_label.style().polish(self.csn_collision_label)

        moving = snapshot.position_code in (3, 4, 5)
        can_select_position = snapshot.homed and not snapshot.collision and not moving
        self.csn_home_button.setEnabled(not moving)
        self.csn_position1_button.setEnabled(can_select_position)
        self.csn_position2_button.setEnabled(can_select_position)
        self.csn_stop_button.setEnabled(moving)

    def _set_csn_controls(self, connected: bool) -> None:
        self.csn_home_button.setEnabled(connected)
        self.csn_position1_button.setEnabled(False)
        self.csn_position2_button.setEnabled(False)
        self.csn_stop_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.camera_worker is not None and self.camera_worker.isRunning():
            self.camera_worker.request_stop()
            self.camera_worker.wait(2000)
        self.csn_worker.shutdown()
        self.ic4_runtime.close()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Vision and CSN210 Hardware Test")
    app.setStyle("Fusion")
    window = VisionHardwareTestWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
