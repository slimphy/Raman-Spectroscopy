"""Large CSV hyperspectral-map viewer.

Run with::

    python hyperspectral_viewer.py [Raw_Hyperspectral_Data_....csv]

The CSV is indexed sequentially instead of being loaded as one enormous matrix.
Only coordinates, file offsets, and a few map metrics remain in memory; a full
spectrum is read from disk when a point in the map is clicked.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QThread, Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


def read_header(path: str) -> tuple[list[str], int, np.ndarray]:
    """Return headers, first spectral column, and numeric spectral axis."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        headers = next(csv.reader(handle))
    spectral_start = None
    for i, value in enumerate(headers):
        try:
            float(value)
        except ValueError:
            continue
        spectral_start = i
        break
    if spectral_start is None:
        raise ValueError("숫자로 된 Raman shift 열을 찾을 수 없습니다.")
    axis = np.asarray(headers[spectral_start:], dtype=np.float64)
    return headers, spectral_start, axis


@dataclass
class DataIndex:
    path: str
    headers: list[str]
    spectral_start: int
    axis: np.ndarray
    xyz: np.ndarray
    offsets: np.ndarray
    total: np.ndarray
    maximum: np.ndarray
    peak_position: np.ndarray
    temperature: np.ndarray
    anti_snr: np.ndarray


@dataclass(frozen=True)
class TemperatureSettings:
    anti_center: float = -520.0
    anti_width: float = 30.0
    stokes_center: float = 520.0
    stokes_width: float = 30.0
    bg_center: float = 0.0
    bg_width: float = 50.0
    laser_nm: float = 532.0
    correction: float = 1.0
    anti_snr_min: float = 3.0


def calculate_temperature_and_snr(
    axis: np.ndarray, spectrum: np.ndarray, settings: TemperatureSettings
) -> tuple[float, float]:
    """Return temperature and anti-Stokes SNR using ``main.TempPair`` regions."""
    def region(center: float, width: float) -> np.ndarray:
        return (axis >= center - width / 2) & (axis <= center + width / 2)

    anti_mask = region(settings.anti_center, settings.anti_width)
    stokes_mask = region(settings.stokes_center, settings.stokes_width)
    bg_mask = region(settings.bg_center, settings.bg_width)
    if not (anti_mask.any() and stokes_mask.any() and bg_mask.any()):
        return np.nan, np.nan
    background = spectrum[bg_mask]
    baseline = np.nanmedian(background)
    anti_net = np.nanmean(spectrum[anti_mask]) - baseline
    stokes_net = np.nanmean(spectrum[stokes_mask]) - baseline
    noise = np.nanstd(background, ddof=1) if np.count_nonzero(np.isfinite(background)) > 1 else np.nan
    if noise > 0:
        anti_snr = float(anti_net / noise)
    elif anti_net > 0 and noise == 0:
        anti_snr = np.inf
    else:
        anti_snr = np.nan
    if not np.isfinite(anti_snr) and anti_snr != np.inf:
        return np.nan, anti_snr
    if anti_snr < settings.anti_snr_min:
        return np.nan, anti_snr
    if not (anti_net > 0 and stokes_net > 0):
        return np.nan, anti_snr
    v0 = 1e7 / settings.laser_nm
    vm = abs(settings.stokes_center)
    ratio = anti_net / stokes_net
    k_factor = ((v0 + vm) / (v0 - vm)) ** 4
    ln_term = settings.correction * k_factor / ratio
    if not (ln_term > 1.0):
        return np.nan, anti_snr
    temperature = float((1.43877 * vm) / np.log(ln_term) - 273.15)
    return temperature, anti_snr


def calculate_temperature(axis: np.ndarray, spectrum: np.ndarray, settings: TemperatureSettings) -> float:
    """Compatibility wrapper returning only temperature in degrees Celsius."""
    return calculate_temperature_and_snr(axis, spectrum, settings)[0]


class IndexWorker(QThread):
    progress = pyqtSignal(int)
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: str, temperature_settings: TemperatureSettings):
        super().__init__()
        self.path = path
        self.temperature_settings = temperature_settings

    def run(self) -> None:
        try:
            headers, start, axis = read_header(self.path)
            lower = [h.strip().lower() for h in headers[:start]]

            def coordinate_index(letter: str) -> int:
                for i, name in enumerate(lower):
                    if name == letter or name.startswith(letter + "("):
                        return i
                raise ValueError(f"{letter.upper()} 좌표 열을 찾을 수 없습니다.")

            xi, yi, zi = (coordinate_index(c) for c in "xyz")
            size = max(os.path.getsize(self.path), 1)
            coords, offsets, totals, maxima, peaks, temperatures, anti_snrs = [], [], [], [], [], [], []
            last_percent = -1
            with open(self.path, "rb") as handle:
                handle.readline()  # header
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    values = np.fromstring(line, sep=",")
                    if values.size != len(headers):
                        continue
                    spectrum = values[start:]
                    finite = np.isfinite(spectrum)
                    if finite.any():
                        safe = np.where(finite, spectrum, -np.inf)
                        peak_i = int(np.argmax(safe))
                        totals.append(float(np.nansum(spectrum)))
                        maxima.append(float(safe[peak_i]))
                        peaks.append(float(axis[peak_i]))
                    else:
                        totals.append(np.nan)
                        maxima.append(np.nan)
                        peaks.append(np.nan)
                    temperature, anti_snr = calculate_temperature_and_snr(
                        axis, spectrum, self.temperature_settings
                    )
                    temperatures.append(temperature)
                    anti_snrs.append(anti_snr)
                    coords.append((values[xi], values[yi], values[zi]))
                    offsets.append(offset)
                    percent = min(99, int(handle.tell() * 100 / size))
                    if percent != last_percent:
                        self.progress.emit(percent)
                        last_percent = percent
            if not coords:
                raise ValueError("유효한 데이터 행이 없습니다.")
            result = DataIndex(
                self.path, headers, start, axis,
                np.asarray(coords), np.asarray(offsets, dtype=np.int64),
                np.asarray(totals), np.asarray(maxima), np.asarray(peaks),
                np.asarray(temperatures), np.asarray(anti_snrs),
            )
            self.progress.emit(100)
            self.loaded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


def median_step(values: np.ndarray) -> float:
    unique = np.unique(values)
    if len(unique) < 2:
        return 1.0
    return float(np.median(np.diff(unique)))


class HyperspectralViewer(QMainWindow):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Hyperspectral CSV Heatmap & Spectrum Viewer")
        self.resize(1450, 850)
        self.data: DataIndex | None = None
        self.worker: IndexWorker | None = None
        self.current_rows: dict[tuple[float, float], int] = {}
        self.current_image: np.ndarray | None = None

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        controls = QHBoxLayout()
        self.open_button = QPushButton("CSV 열기")
        self.open_button.clicked.connect(self.choose_file)
        self.file_label = QLabel("파일을 선택하세요")
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "전체 intensity (합)", "Peak intensity", "Peak Raman shift",
            "Temperature (°C)", "Anti-Stokes SNR",
        ])
        self.metric_combo.currentIndexChanged.connect(self.refresh_map)
        self.z_combo = QComboBox()
        self.z_combo.currentIndexChanged.connect(self.refresh_map)
        self.row_shift_check = QCheckBox("교대 행 X shift 보정")
        self.row_shift_check.setToolTip(
            "Y를 오름차순으로 정렬한 뒤 선택한 한쪽 scan line만 이동합니다."
        )
        self.row_shift_check.setChecked(True)
        self.row_shift_check.stateChanged.connect(self.refresh_map)
        self.row_shift_mode = QComboBox()
        self.row_shift_mode.addItems(["홀수 행만 −1칸", "짝수 행만 +1칸"])
        self.row_shift_mode.setToolTip("정렬 기준으로 유지할 행의 반대쪽을 선택해 이동하세요.")
        self.row_shift_mode.currentIndexChanged.connect(self.refresh_map)
        self.row_shift_check.toggled.connect(self.row_shift_mode.setEnabled)
        self.color_autoscale_button = QPushButton("색상 Z축 Auto scale")
        self.color_autoscale_button.setToolTip("유효한 heatmap 값의 1–99 percentile로 색상 범위를 재설정합니다.")
        self.color_autoscale_button.clicked.connect(self.autoscale_color)
        self.view_autoscale_button = QPushButton("화면 Auto fit")
        self.view_autoscale_button.clicked.connect(self.autoscale_views)
        controls.addWidget(self.open_button)
        controls.addWidget(self.file_label, 1)
        controls.addWidget(QLabel("Heatmap:"))
        controls.addWidget(self.metric_combo)
        controls.addWidget(QLabel("Z layer:"))
        controls.addWidget(self.z_combo)
        controls.addWidget(self.row_shift_check)
        controls.addWidget(self.row_shift_mode)
        controls.addWidget(self.color_autoscale_button)
        controls.addWidget(self.view_autoscale_button)
        outer.addLayout(controls)

        temp_controls = QHBoxLayout()
        temp_controls.addWidget(QLabel("온도 계산 — Anti 중심/폭:"))
        self.anti_center = self.make_spin(-4000, 4000, -520, 1)
        self.anti_width = self.make_spin(1, 1000, 30, 1)
        temp_controls.addWidget(self.anti_center)
        temp_controls.addWidget(self.anti_width)
        temp_controls.addWidget(QLabel("Stokes 중심/폭:"))
        self.stokes_center = self.make_spin(-4000, 4000, 520, 1)
        self.stokes_width = self.make_spin(1, 1000, 30, 1)
        temp_controls.addWidget(self.stokes_center)
        temp_controls.addWidget(self.stokes_width)
        temp_controls.addWidget(QLabel("BG 중심/폭:"))
        self.bg_center = self.make_spin(-4000, 4000, 0, 1)
        self.bg_width = self.make_spin(1, 1000, 50, 1)
        temp_controls.addWidget(self.bg_center)
        temp_controls.addWidget(self.bg_width)
        temp_controls.addWidget(QLabel("Laser nm:"))
        self.laser_nm = self.make_spin(200, 2000, 532, 2)
        temp_controls.addWidget(self.laser_nm)
        temp_controls.addWidget(QLabel("C:"))
        self.correction = self.make_spin(0.001, 100, 1, 3)
        temp_controls.addWidget(self.correction)
        temp_controls.addWidget(QLabel("최소 Anti SNR:"))
        self.anti_snr_min = self.make_spin(0, 1000, 3, 2)
        self.anti_snr_min.setToolTip("Anti net intensity / background 표준편차. 0이면 SNR 필터를 끕니다.")
        temp_controls.addWidget(self.anti_snr_min)
        self.recalculate_button = QPushButton("온도 다시 계산")
        self.recalculate_button.clicked.connect(self.recalculate_temperature)
        temp_controls.addWidget(self.recalculate_button)
        outer.addLayout(temp_controls)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.map_plot = pg.PlotWidget(title="XY Heatmap")
        self.map_plot.setLabel("bottom", "X", units="um")
        self.map_plot.setLabel("left", "Y", units="um")
        self.map_plot.setAspectLocked(False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.map_plot.addItem(self.image)
        self.colorbar = pg.ColorBarItem(colorMap=pg.colormap.get("CET-L9"), interactive=True)
        self.colorbar.setImageItem(self.image, insert_in=self.map_plot.getPlotItem())
        self.marker = pg.ScatterPlotItem(size=13, pen=pg.mkPen("w", width=2), brush=QColor(255, 80, 80, 120))
        self.map_plot.addItem(self.marker)
        self.map_plot.scene().sigMouseClicked.connect(self.map_clicked)

        self.spectrum_plot = pg.PlotWidget(title="클릭 지점 스펙트럼")
        self.spectrum_plot.setLabel("bottom", "Raman shift", units="cm⁻¹")
        self.spectrum_plot.setLabel("left", "Intensity", units="counts")
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen("#33d6ff", width=1.5))
        splitter.addWidget(self.map_plot)
        splitter.addWidget(self.spectrum_plot)
        splitter.setSizes([700, 700])
        outer.addWidget(splitter, 1)

        self.status = QLabel("대용량 파일은 최초 인덱싱에 시간이 걸릴 수 있습니다.")
        outer.addWidget(self.status)
        if initial_path:
            self.load_file(initial_path)

    @staticmethod
    def make_spin(minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def temperature_settings(self) -> TemperatureSettings:
        return TemperatureSettings(
            self.anti_center.value(), self.anti_width.value(),
            self.stokes_center.value(), self.stokes_width.value(),
            self.bg_center.value(), self.bg_width.value(),
            self.laser_nm.value(), self.correction.value(), self.anti_snr_min.value(),
        )

    def recalculate_temperature(self) -> None:
        if self.data is not None:
            self.load_file(self.data.path)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Hyperspectral CSV 선택", "", "CSV files (*.csv);;All files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        if self.worker and self.worker.isRunning():
            return
        self.data = None
        self.open_button.setEnabled(False)
        self.recalculate_button.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.file_label.setText(Path(path).name)
        self.status.setText("파일을 스트리밍으로 인덱싱하는 중…")
        self.worker = IndexWorker(path, self.temperature_settings())
        self.worker.progress.connect(self.progress.setValue)
        self.worker.loaded.connect(self.index_ready)
        self.worker.failed.connect(self.index_failed)
        self.worker.start()

    def index_ready(self, data: DataIndex) -> None:
        self.data = data
        self.open_button.setEnabled(True)
        self.recalculate_button.setEnabled(True)
        self.progress.setVisible(False)
        self.z_combo.blockSignals(True)
        self.z_combo.clear()
        for z in np.unique(data.xyz[:, 2]):
            self.z_combo.addItem(f"{z:g} µm", float(z))
        self.z_combo.blockSignals(False)
        metadata = ", ".join(data.headers[:data.spectral_start])
        self.status.setText(
            f"{len(data.xyz):,} spectra · {len(data.axis):,} channels · metadata: {metadata} "
            "(temperature/channel metadata 열은 이 파일에 없음)"
        )
        self.refresh_map()

    def index_failed(self, message: str) -> None:
        self.open_button.setEnabled(True)
        self.recalculate_button.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "CSV 읽기 실패", message)
        self.status.setText("파일을 읽지 못했습니다.")

    def refresh_map(self) -> None:
        if self.data is None or self.z_combo.count() == 0:
            return
        z = float(self.z_combo.currentData())
        indices = np.flatnonzero(np.isclose(self.data.xyz[:, 2], z, rtol=0, atol=1e-9))
        metric = (
            self.data.total, self.data.maximum, self.data.peak_position,
            self.data.temperature, self.data.anti_snr,
        )[self.metric_combo.currentIndex()]
        ys = np.unique(self.data.xyz[indices, 1])
        raw_xs = np.unique(self.data.xyz[indices, 0])
        dx = median_step(raw_xs)
        y_lookup = {float(v): i for i, v in enumerate(ys)}

        def display_x(row: int) -> float:
            raw_x, raw_y = map(float, self.data.xyz[row, :2])
            if not self.row_shift_check.isChecked():
                return raw_x
            is_odd_line = y_lookup[raw_y] % 2 == 0  # scan-line number is 1-based
            if self.row_shift_mode.currentIndex() == 0:
                return raw_x - dx if is_odd_line else raw_x
            return raw_x + dx if not is_odd_line else raw_x

        displayed_x = np.asarray([display_x(int(row)) for row in indices])
        xs = np.unique(displayed_x)
        x_lookup = {float(v): i for i, v in enumerate(xs)}
        image = np.full((len(ys), len(xs)), np.nan)
        self.current_rows.clear()
        # Later duplicate points replace earlier ones; acquisition files normally have one per XYZ.
        for row in indices:
            x = display_x(int(row))
            y = float(self.data.xyz[row, 1])
            image[y_lookup[y], x_lookup[x]] = metric[row]
            self.current_rows[(x, y)] = int(row)
        dx, dy = median_step(xs), median_step(ys)
        rect = QRectF(float(xs.min() - dx / 2), float(ys.min() - dy / 2), float(np.ptp(xs) + dx), float(np.ptp(ys) + dy))
        self.image.setImage(image, autoLevels=True)
        self.image.setRect(rect)
        self.current_image = image
        self.autoscale_color()
        self.map_plot.autoRange()
        self.marker.clear()

    def autoscale_color(self) -> None:
        """Reset the heatmap color (Z-value) range while rejecting outliers."""
        if self.current_image is None:
            return
        finite = self.current_image[np.isfinite(self.current_image)]
        if finite.size == 0:
            self.status.setText("Auto scale할 유효 heatmap 값이 없습니다.")
            return
        low, high = np.percentile(finite, [1.0, 99.0])
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if high <= low:
            margin = max(abs(float(low)) * 0.01, 1.0)
            low, high = low - margin, high + margin
        self.image.setLevels((float(low), float(high)))
        self.colorbar.setLevels((float(low), float(high)))

    def autoscale_views(self) -> None:
        """Fit both map XY axes and spectrum XY axes to their current data."""
        self.map_plot.enableAutoRange()
        self.map_plot.autoRange()
        self.spectrum_plot.enableAutoRange()
        self.spectrum_plot.autoRange()

    def map_clicked(self, event) -> None:
        if self.data is None or not self.current_rows or event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.map_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.map_plot.getPlotItem().vb.mapSceneToView(event.scenePos())
        keys = np.asarray(list(self.current_rows.keys()), dtype=float)
        # Normalize distance so anisotropic scan steps do not bias point selection.
        scale = np.array([median_step(keys[:, 0]), median_step(keys[:, 1])])
        selected = int(np.argmin(np.sum(((keys - [point.x(), point.y()]) / scale) ** 2, axis=1)))
        map_x, map_y = map(float, keys[selected])
        row = self.current_rows[(map_x, map_y)]
        spectrum = self.read_spectrum(row)
        self.spectrum_curve.setData(self.data.axis, spectrum)
        self.spectrum_plot.enableAutoRange()
        self.marker.setData([map_x], [map_y])
        raw_x, raw_y, z = self.data.xyz[row]
        self.spectrum_plot.setTitle(f"Spectrum at X={raw_x:g}, Y={raw_y:g}, Z={z:g} µm")
        corrected = (
            f" · corrected map X={map_x:g} ({self.row_shift_mode.currentText()})"
            if self.row_shift_check.isChecked() else ""
        )
        snr = self.data.anti_snr[row]
        temperature = self.data.temperature[row]
        temp_text = f"{temperature:.2f} °C" if np.isfinite(temperature) else "None (SNR/ratio 조건 미달)"
        self.status.setText(
            f"선택 원본: X={raw_x:g}, Y={raw_y:g}, Z={z:g} µm{corrected} · "
            f"Anti SNR={snr:.2f} · Temperature={temp_text} · row {row + 2:,}"
        )

    def read_spectrum(self, row: int) -> np.ndarray:
        assert self.data is not None
        with open(self.data.path, "rb") as handle:
            handle.seek(int(self.data.offsets[row]))
            values = np.fromstring(handle.readline(), sep=",")
        return values[self.data.spectral_start:]


def main() -> int:
    pg.setConfigOptions(antialias=True, background="#17191d", foreground="#e5e7eb")
    app = QApplication(sys.argv)
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    if initial and not os.path.isfile(initial):
        initial = None
    window = HyperspectralViewer(initial)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
