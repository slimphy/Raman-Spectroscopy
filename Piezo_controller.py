import sys
import re
import time
import threading
import serial
import serial.tools.list_ports
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QGroupBox, QLineEdit, QComboBox, QRadioButton,
                             QDoubleSpinBox, QMessageBox, QCheckBox)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal


# ==========================================
# 1. 고속 맵핑 워커 스레드 (ㄹ자 스캔)
# ==========================================
class HighSpeedMappingWorker(QThread):
    progress_signal = pyqtSignal(int, int)  # 현재 포인트, 총 포인트
    finished_signal = pyqtSignal()

    def __init__(self, stage, params):
        super().__init__()
        self.stage = stage
        self.params = params
        self.is_running = True

    def run(self):
        xi, xf, dx = self.params['xi'], self.params['xf'], self.params['dx']
        yi, yf, dy = self.params['yi'], self.params['yf'], self.params['dy']
        zi, zf, dz = self.params['zi'], self.params['zf'], self.params['dz']

        # 스텝이 0이거나 음수인 경우를 방지하여 좌표 배열 생성
        x_points = np.arange(xi, xf + dx / 2, dx) if dx > 0 else np.array([xi])
        y_points = np.arange(yi, yf + dy / 2, dy) if dy > 0 else np.array([yi])
        z_points = np.arange(zi, zf + dz / 2, dz) if dz > 0 else np.array([zi])

        total_points = len(x_points) * len(y_points) * len(z_points)
        current_pt = 0

        print(f"[Mapping] 총 {total_points} 포인트 초고속 스캔 시작...")

        # 맵핑 루프 시작
        for z in z_points:
            if not self.is_running: break

            for yi_idx, y in enumerate(y_points):
                if not self.is_running: break

                # ㄹ자 스캔(Snake Scan): 짝수 행은 정방향, 홀수 행은 역방향
                current_x_points = x_points if yi_idx % 2 == 0 else x_points[::-1]

                for x in current_x_points:
                    if not self.is_running: break

                    # 고속 이동 명령 전송 (Ack 대기 생략으로 속도 극대화)
                    if self.stage.is_connected:
                        self.stage.move_to_logical(x, y, z, wait_ack=False)

                    current_pt += 1

                    # UI 갱신 (너무 자주 갱신하면 렉이 걸리므로 10포인트 단위로 emit)
                    if current_pt % 10 == 0 or current_pt == total_points:
                        self.progress_signal.emit(current_pt, total_points)

                    # [추가된 기능] 이동 후 10ms 측정 시간 (측정 시뮬레이션 및 딜레이 확보)
                    time.sleep(0.01)

                    # [추가된 기능] 사용자가 중단하지 않고 맵핑이 정상적으로 끝난 경우 영점 복귀
        if self.is_running and self.stage.is_connected:
            print("[Mapping] 스캔 완료. 시작점(0, 0, 0)으로 복귀합니다.")
            self.stage.move_to_logical(0.0, 0.0, 0.0, wait_ack=True)
            time.sleep(0.5)  # 안전한 복귀를 위한 약간의 여유 시간

        self.finished_signal.emit()

    def stop(self):
        self.is_running = False


# ==========================================
# 2. Piezo Stage 통신 및 제어 클래스
# ==========================================
class PiezoController:
    def __init__(self, port='COM3', baudrate=115200, simulate=False):
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate
        self.serial_conn = None
        self.is_connected = False

        self.lock = threading.RLock()

        # 오프셋 및 하드웨어/논리 좌표 변수
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0

        self.hard_x = 0.0
        self.hard_y = 0.0
        self.hard_z = 0.0

        self.logical_x = 0.0
        self.logical_y = 0.0
        self.logical_z = 0.0

    def connect(self, port_name):
        self.port = port_name
        if self.simulate:
            self.is_connected = True
            print("[Simulate] 가상 연결됨")
            return True
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1, write_timeout=0.5)
            self.is_connected = True
            time.sleep(0.1)
            if self.serial_conn.in_waiting:
                self.serial_conn.read_all()
            print(f"Connected to {self.port}")
            self.read_position()
            return True
        except Exception as e:
            print(f"Connection Error: {e}")
            return False

    def disconnect(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        print("Disconnected")

    def send_command(self, command, wait_for_ack=True):
        if not self.simulate and self.is_connected:
            with self.lock:
                try:
                    self.serial_conn.reset_input_buffer()
                    self.serial_conn.reset_output_buffer()

                    cmd_str = f"{command}\r\n"
                    self.serial_conn.write(cmd_str.encode('ascii'))
                    self.serial_conn.flush()

                    if wait_for_ack:
                        start_time = time.time()
                        response = ""
                        while (time.time() - start_time) < 0.15:
                            if self.serial_conn.in_waiting:
                                chunk = self.serial_conn.read(self.serial_conn.in_waiting).decode('ascii',
                                                                                                  errors='ignore')
                                response += chunk
                                if '\n' in chunk or '\r' in chunk:
                                    break
                            time.sleep(0.005)
                        return response.strip()
                except Exception as e:
                    print(f"Command Error: {e}")
        return ""

    def check_is_indexed(self):
        res = self.send_command("RI x y z", wait_for_ack=True)
        if res:
            match = re.search(r'x(\d).*y(\d).*z(\d)', res, re.IGNORECASE)
            if match:
                status = [int(v) for v in match.groups()]
                return all(s == 1 for s in status)
        return False

    def find_index(self):
        self.send_command("IN x y z", wait_for_ack=True)
        print("[Stage] Index(물리적 영점) 찾기 동작 시작...")

    def read_position(self):
        """X, Y, Z를 개별적으로 요청하여 통신 누락 방지"""
        self.hard_x = self._query_single_axis('x')
        self.hard_y = self._query_single_axis('y')
        self.hard_z = self._query_single_axis('z')
        return self.get_logical_position()

    def _query_single_axis(self, axis_label):
        cmd = f"RP {axis_label}"
        response = self.send_command(cmd, wait_for_ack=True)
        if response:
            match = re.search(rf'{axis_label}[:\s]*([-\d\.]+)', response, re.IGNORECASE)
            if match:
                return float(match.group(1))
            else:
                nums = re.findall(r'[-\d\.]+', response)
                for n in nums:
                    try:
                        if n.count('.') <= 1 and n != '.':
                            return float(n)
                    except ValueError:
                        continue
        return getattr(self, f'hard_{axis_label}', 0.0)

    def get_logical_position(self):
        self.logical_x = self.hard_x - self.offset_x
        self.logical_y = self.hard_y - self.offset_y
        self.logical_z = self.hard_z - self.offset_z
        return (self.logical_x, self.logical_y, self.logical_z)

    def set_zero(self):
        self.read_position()
        self.offset_x = self.hard_x
        self.offset_y = self.hard_y
        self.offset_z = self.hard_z

    def set_speed(self, speed):
        self.send_command(f"SP x{speed:.3f} y{speed:.3f} z{speed:.3f}")

    def move_relative(self, dx, dy, dz):
        parts = ["MR"]
        if dx != 0: parts.append(f"x{dx:.3f}")
        if dy != 0: parts.append(f"y{dy:.3f}")
        if dz != 0: parts.append(f"z{dz:.3f}")
        if len(parts) > 1:
            self.send_command(" ".join(parts))

    def move_absolute(self, x, y, z):
        abs_x = x + self.offset_x
        abs_y = y + self.offset_y
        abs_z = z + self.offset_z
        self.send_command(f"GT x{abs_x:.3f} y{abs_y:.3f} z{abs_z:.3f}")

    def move_to_logical(self, target_x, target_y, target_z, wait_ack=False):
        """고속 맵핑을 위한 가변 패킷 이동 함수"""
        abs_target_x = target_x + self.offset_x
        abs_target_y = target_y + self.offset_y
        abs_target_z = target_z + self.offset_z

        parts = ["GT"]
        if target_x != self.logical_x: parts.append(f"x{abs_target_x:.3f}")
        if target_y != self.logical_y: parts.append(f"y{abs_target_y:.3f}")
        if target_z != self.logical_z: parts.append(f"z{abs_target_z:.3f}")

        self.logical_x = target_x
        self.logical_y = target_y
        self.logical_z = target_z

        if len(parts) > 1:
            cmd = " ".join(parts)
            self.send_command(cmd, wait_for_ack=wait_ack)

    def stop_motion(self):
        self.send_command("SH x y z")


# ==========================================
# 3. PyQt6 테스트 UI 클래스
# ==========================================
class PiezoTestUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Precibeo GO Controller Test")
        self.stage = PiezoController()
        self._is_updating = False

        self.init_ui()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.auto_update_indicator)
        self.update_timer.start(100)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # ----- 상단: 통신 연결 -----
        conn_group = QGroupBox("1. Initialize (Connection)")
        conn_layout = QHBoxLayout()
        self.combo_port = QComboBox()
        self.combo_port.addItems([p.device for p in serial.tools.list_ports.comports()] or ["COM3"])

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.toggle_connection)

        self.btn_find_index = QPushButton("Find Index (하드웨어 영점 찾기)")
        self.btn_find_index.setStyleSheet("background-color: #FFE082; font-weight: bold;")
        self.btn_find_index.clicked.connect(self.run_find_index)
        self.btn_find_index.setEnabled(False)

        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.combo_port)
        conn_layout.addWidget(self.btn_connect)
        conn_layout.addWidget(self.btn_find_index)
        conn_layout.addStretch()
        conn_group.setLayout(conn_layout)
        main_layout.addWidget(conn_group)

        # ----- 중앙 패널 -----
        center_layout = QHBoxLayout()

        # [왼쪽 패널]
        left_panel = QVBoxLayout()

        # 1. Current Position
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
        left_panel.addWidget(pos_group)

        # 2. Go To
        go_group = QGroupBox("Saved / Target Position (Go To)")
        go_layout = QGridLayout()

        self.radio_go_rel = QRadioButton("Relative Move")
        self.radio_go_abs = QRadioButton("Absolute Move")
        self.radio_go_abs.setChecked(True)

        self.input_go_x = QDoubleSpinBox();
        self.input_go_x.setRange(-10000, 10000);
        self.input_go_x.setDecimals(3)
        self.input_go_y = QDoubleSpinBox();
        self.input_go_y.setRange(-10000, 10000);
        self.input_go_y.setDecimals(3)
        self.input_go_z = QDoubleSpinBox();
        self.input_go_z.setRange(-10000, 10000);
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
        left_panel.addWidget(go_group)

        center_layout.addLayout(left_panel)

        # [오른쪽 패널]
        right_panel = QVBoxLayout()

        # 3. Motion Control
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
        right_panel.addWidget(ctrl_group)

        # 4. Jog
        jog_group = QGroupBox("Relative Motion (Jog)")
        jog_layout = QGridLayout()

        self.btn_y_plus = QPushButton("∧\n+y")
        self.btn_y_minus = QPushButton("∨\n-y")
        self.btn_x_minus = QPushButton("<\n-x")
        self.btn_x_plus = QPushButton(">\n+x")
        self.btn_z_plus = QPushButton("∧\nctrl-up (+z)")
        self.btn_z_minus = QPushButton("∨\nctrl-down (-z)")
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("background-color: #EF9A9A; font-weight: bold;")
        self.btn_stop.clicked.connect(lambda: self.stage.stop_motion())

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
        right_panel.addWidget(jog_group)

        # 5. High-Speed Mapping
        map_group = QGroupBox("High-Speed Mapping (ㄹ-Scan)")
        map_layout = QGridLayout()

        self.spin_xi = QDoubleSpinBox();
        self.spin_xi.setRange(-10000, 10000);
        self.spin_xi.setValue(0.0)
        self.spin_xf = QDoubleSpinBox();
        self.spin_xf.setRange(-10000, 10000);
        self.spin_xf.setValue(10.0)
        self.spin_dx = QDoubleSpinBox();
        self.spin_dx.setRange(0.001, 1000);
        self.spin_dx.setValue(1.0)

        self.spin_yi = QDoubleSpinBox();
        self.spin_yi.setRange(-10000, 10000);
        self.spin_yi.setValue(0.0)
        self.spin_yf = QDoubleSpinBox();
        self.spin_yf.setRange(-10000, 10000);
        self.spin_yf.setValue(10.0)
        self.spin_dy = QDoubleSpinBox();
        self.spin_dy.setRange(0.001, 1000);
        self.spin_dy.setValue(1.0)

        self.spin_zi = QDoubleSpinBox();
        self.spin_zi.setRange(-10000, 10000);
        self.spin_zi.setValue(0.0)
        self.spin_zf = QDoubleSpinBox();
        self.spin_zf.setRange(-10000, 10000);
        self.spin_zf.setValue(0.0)
        self.spin_dz = QDoubleSpinBox();
        self.spin_dz.setRange(0.001, 1000);
        self.spin_dz.setValue(1.0)

        map_layout.addWidget(QLabel("Axis"), 0, 0)
        map_layout.addWidget(QLabel("Start (i)"), 0, 1)
        map_layout.addWidget(QLabel("End (f)"), 0, 2)
        map_layout.addWidget(QLabel("Step (d)"), 0, 3)

        map_layout.addWidget(QLabel("X:"), 1, 0);
        map_layout.addWidget(self.spin_xi, 1, 1);
        map_layout.addWidget(self.spin_xf, 1, 2);
        map_layout.addWidget(self.spin_dx, 1, 3)
        map_layout.addWidget(QLabel("Y:"), 2, 0);
        map_layout.addWidget(self.spin_yi, 2, 1);
        map_layout.addWidget(self.spin_yf, 2, 2);
        map_layout.addWidget(self.spin_dy, 2, 3)
        map_layout.addWidget(QLabel("Z:"), 3, 0);
        map_layout.addWidget(self.spin_zi, 3, 1);
        map_layout.addWidget(self.spin_zf, 3, 2);
        map_layout.addWidget(self.spin_dz, 3, 3)

        self.lbl_map_status = QLabel("Ready")
        self.lbl_map_status.setStyleSheet("color: #1976D2; font-weight: bold;")

        self.btn_map_start = QPushButton("▶ Start Mapping")
        self.btn_map_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_map_start.clicked.connect(self.toggle_mapping)

        map_layout.addWidget(self.lbl_map_status, 4, 0, 1, 2)
        map_layout.addWidget(self.btn_map_start, 4, 2, 1, 2)
        map_group.setLayout(map_layout)
        right_panel.addWidget(map_group)

        center_layout.addLayout(right_panel)
        main_layout.addLayout(center_layout)

    # ---------------- UI Action Methods ----------------
    def toggle_connection(self):
        if not self.stage.is_connected:
            port = self.combo_port.currentText()
            if self.stage.connect(port):
                self.btn_connect.setText("Disconnect")
                self.btn_connect.setStyleSheet("background-color: #EF9A9A;")
                self.btn_find_index.setEnabled(True)
                self.apply_speed()

                if not self.stage.check_is_indexed():
                    reply = QMessageBox.question(
                        self,
                        'Hardware Index Not Found',
                        '장비의 기계적 절대 영점(Index)이 설정되어 있지 않습니다.\n정확한 맵핑과 이동을 위해 지금 영점(INdex) 찾기를 수행하시겠습니까?',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.run_find_index()
            else:
                QMessageBox.warning(self, "Error", f"{port} 연결 실패")
        else:
            self.stage.disconnect()
            self.btn_connect.setText("Connect")
            self.btn_connect.setStyleSheet("")
            self.btn_find_index.setEnabled(False)

    def run_find_index(self):
        if self.stage.is_connected:
            self.stage.find_index()
            QMessageBox.information(
                self,
                "진행 중",
                "하드웨어 영점(Index) 탐색을 시작했습니다.\n모든 축이 기준점으로 이동할 때까지 잠시 기다려주세요."
            )

    def toggle_auto_update(self, state):
        if state == Qt.CheckState.Checked.value:
            self.update_timer.start(100)
        else:
            self.update_timer.stop()

    def auto_update_indicator(self):
        if getattr(self, '_is_updating', False):
            return
        self._is_updating = True
        try:
            self._update_ui_labels()
        except Exception as e:
            print(f"[UI] Update Error: {e}")
        finally:
            self._is_updating = False

    def manual_update_indicator(self):
        if not self.stage.is_connected:
            QMessageBox.warning(self, "오류", "스테이지가 연결되어 있지 않습니다.")
            return
        self._update_ui_labels()
        print("[UI] 위치 수동 업데이트 완료")

    def _update_ui_labels(self):
        if self.stage.is_connected:
            log_x, log_y, log_z = self.stage.read_position()

            if self.chk_absolute.isChecked():
                self.lbl_x.setText(f"X: {self.stage.hard_x:.3f} µm (Abs)")
                self.lbl_y.setText(f"Y: {self.stage.hard_y:.3f} µm (Abs)")
                self.lbl_z.setText(f"Z: {self.stage.hard_z:.3f} µm (Abs)")
            else:
                self.lbl_x.setText(f"X: {log_x:.3f} µm")
                self.lbl_y.setText(f"Y: {log_y:.3f} µm")
                self.lbl_z.setText(f"Z: {log_z:.3f} µm")

    def set_zero(self):
        if self.stage.is_connected:
            self.stage.set_zero()
            self._update_ui_labels()

    def apply_speed(self):
        if self.stage.is_connected:
            self.stage.set_speed(self.spin_speed.value())

    def go_target(self):
        if not self.stage.is_connected: return
        x = self.input_go_x.value()
        y = self.input_go_y.value()
        z = self.input_go_z.value()

        if self.radio_go_abs.isChecked():
            self.stage.move_absolute(x, y, z)
        else:
            self.stage.move_relative(x, y, z)

        QTimer.singleShot(100, self._update_ui_labels)

    def setup_jog_btn(self, btn, dx, dy, dz):
        btn.pressed.connect(lambda: self.jog_start(dx, dy, dz))
        btn.released.connect(self.jog_stop)

    def jog_start(self, dx, dy, dz):
        if not self.stage.is_connected: return

        if self.radio_jog_step.isChecked():
            step = self.spin_step.value()
            self.stage.move_relative(dx * step, dy * step, dz * step)
            QTimer.singleShot(100, self._update_ui_labels)
        else:
            large_step = 10000.0
            self.stage.move_relative(dx * large_step, dy * large_step, dz * large_step)

    def jog_stop(self):
        if not self.stage.is_connected: return
        if self.radio_jog_cont.isChecked():
            self.stage.stop_motion()
            QTimer.singleShot(200, self._update_ui_labels)

    # ---------------- Mapping Action Methods ----------------
    def toggle_mapping(self):
        if not hasattr(self, 'map_worker') or not self.map_worker.isRunning():
            if not self.stage.is_connected:
                QMessageBox.warning(self, "오류", "스테이지가 연결되어 있지 않습니다.")
                return

            params = {
                'xi': self.spin_xi.value(), 'xf': self.spin_xf.value(), 'dx': self.spin_dx.value(),
                'yi': self.spin_yi.value(), 'yf': self.spin_yf.value(), 'dy': self.spin_dy.value(),
                'zi': self.spin_zi.value(), 'zf': self.spin_zf.value(), 'dz': self.spin_dz.value()
            }

            self.map_worker = HighSpeedMappingWorker(self.stage, params)
            self.map_worker.progress_signal.connect(self.update_mapping_progress)
            self.map_worker.finished_signal.connect(self.mapping_finished)

            self.btn_map_start.setText("⏹ Stop Mapping")
            self.btn_map_start.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")

            # 자동 업데이트 타이머 끄기 (통신 간섭 방지)
            if self.chk_auto_update.isChecked():
                self.update_timer.stop()

            self.map_worker.start()
        else:
            # 사용자가 강제 중단 시에만 Stop 명령 전송
            self.map_worker.stop()
            self.stage.stop_motion()

    def update_mapping_progress(self, current, total):
        self.lbl_map_status.setText(f"Mapping... {current} / {total}")

    def mapping_finished(self):
        self.btn_map_start.setText("▶ Start Mapping")
        self.btn_map_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.lbl_map_status.setText("Finished")

        # 맵핑 종료 후 자동 업데이트 타이머 복구
        if self.chk_auto_update.isChecked():
            self.update_timer.start(100)

        # 위치 강제 동기화
        self._update_ui_labels()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PiezoTestUI()
    window.show()
    sys.exit(app.exec())