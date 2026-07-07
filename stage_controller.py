import serial
import time
import threading
import re


class PiezoController:
    def __init__(self, port='COM3', baudrate=115200, simulate=False):
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate
        self.serial_conn = None
        self.is_connected = False

        self.lock = threading.RLock()

        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0

        self.hard_x = 0.0
        self.hard_y = 0.0
        self.hard_z = 0.0

        self.logical_x = 0.0
        self.logical_y = 0.0
        self.logical_z = 0.0

    def connect(self, port_name=None):
        if port_name:
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
        if not self.simulate and self.serial_conn and self.serial_conn.is_open:
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
                                chunk = self.serial_conn.read(self.serial_conn.in_waiting).decode('ascii', errors='ignore')
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

    def set_trigger_out(self, axis='x', value='0.0'):
        cmd = f"TO {axis}{value}"
        self.send_command(cmd, wait_for_ack=True)
        print(f"[Stage] Trigger Out 설정 완료: {cmd}")

    def stop_motion(self):
        self.send_command("SH x y z")

    def get_position(self):
        return self.get_logical_position()