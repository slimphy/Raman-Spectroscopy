import serial
import time
import threading


class PiezoController:
    def __init__(self, port='COM3', baudrate=115200, simulate=False):
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate
        self.serial_conn = None
        self.is_connected = False

        # 🚨 [수정 1] Lock 대신 RLock을 사용하여 중복 잠금으로 인한 멈춤 방지
        self.lock = threading.RLock()

        # 파이썬 내부에서 추적하는 논리적 타겟 위치
        self.logical_x = 0.0
        self.logical_y = 0.0
        self.logical_z = 0.0

        # 🚨 대물렌즈 충돌 방지를 위한 소프트웨어 절대 Z 리미트 (단위: µm)
        self.Z_MAX_LIMIT = 50.0

        # [여기에 추가!] 하드웨어 절대 좌표와 논리 좌표의 차이를 저장하는 오프셋 변수
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0

        # 🚨 대물렌즈 충돌 방지를 위한 소프트웨어 절대 Z 리미트 (단위: µm)
        self.Z_MAX_LIMIT = 50.0

    def connect(self, port_name=None):
        if port_name:
            self.port = port_name

        if self.simulate:
            self.is_connected = True
            print("[Simulation] 스테이지 가상 연결됨")
            return True

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.is_connected = True

            # 연결 직후 쌓여있던 쓰레기 버퍼 비우기
            time.sleep(0.1)
            if self.serial_conn.in_waiting:
                self.serial_conn.read_all()

            print(f"Piezo Stage Connected on {self.port} (Baudrate: {self.baudrate})")

            # 🚨 [신규 추가] 장비 연결 즉시, 현재 허공에 떠있는 그 위치를 안전한 영점으로 자동 세팅합니다!
            time.sleep(0.1)
            self.set_zero()

            return True

        except serial.SerialException as e:
            print(f"Error connecting to stage on {self.port}: {e}")
            return False

    def disconnect(self):
        if not self.simulate and self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        print("Stage Disconnected")

    def send_command(self, command, wait_for_ack=True):
        if not self.simulate and self.is_connected:
            # 🚨 [수정 2] 모든 통신(Jog 포함)이 꼬이지 않도록 이 관문 자체에 Lock을 겁니다.
            with self.lock:
                try:
                    cmd_str = f"{command}\r\n"
                    self.serial_conn.write(cmd_str.encode('ascii'))
                    self.serial_conn.flush()

                    if wait_for_ack:
                        start_time = time.time()
                        response = ""
                        while (time.time() - start_time) < 0.05:
                            if self.serial_conn.in_waiting:
                                chunk = self.serial_conn.read(self.serial_conn.in_waiting).decode('ascii',
                                                                                                  errors='ignore')
                                response += chunk
                                if '\n' in chunk or '\r' in chunk:
                                    break
                            time.sleep(0.001)
                        return response.strip()

                except serial.SerialException as e:
                    # 🚨 [핵심 수정] PermissionError(13) 발생 시 좀비 포트를 강제로 죽이고 재연결 시도
                    print(f"\n🚨 [Stage 통신 치명적 에러] 포트 접근 거부됨: {e}")
                    print("🔄 드라이버 충돌 감지! 포트를 초기화하고 자동 재연결을 시도합니다...")

                    self.is_connected = False
                    try:
                        # 꼬여있는 시리얼 포트를 강제로 닫음
                        if self.serial_conn and self.serial_conn.is_open:
                            self.serial_conn.close()
                    except:
                        pass

                    # 하드웨어가 정신을 차릴 수 있도록 1.5초 대기 후 다시 연결 시도
                    time.sleep(1.5)
                    self.connect()

                except Exception as e:
                    print(f"Stage Command Error: {e}")

        return ""

    def set_zero(self):
        """현재 하드웨어의 진짜 절대 위치를 통신으로 읽어와 논리적 영점(Offset)으로 꽉 고정합니다."""
        if not self.is_connected and not self.simulate:
            print("[Stage WARNING] 연결되지 않아 영점을 설정할 수 없습니다.")
            return

        # 🚨 [핵심] 우리가 완벽하게 고친 함수를 호출해서 진짜 절대 위치를 가져옴
        abs_x, abs_y, abs_z = self.get_hardware_position()

        # 그 진짜 위치를 오프셋(기준점)으로 삼음
        self.offset_x = abs_x
        self.offset_y = abs_y
        self.offset_z = abs_z

        # 소프트웨어 화면상의 좌표는 0,0,0으로 초기화
        self.logical_x = 0.0
        self.logical_y = 0.0
        self.logical_z = 0.0

        print(f"[Stage INFO] 📍 영점 설정 완벽 동기화! (하드웨어 실제 Z:{abs_z:.3f} 위치가 이제 0으로 인식됩니다)")

    def move_to_logical(self, target_x, target_y, target_z, wait_ack=False):
        """논리 좌표에 오프셋을 더해 하드웨어를 움직입니다. (초고속 맵핑 최적화 적용)"""
        abs_target_x = target_x + getattr(self, 'offset_x', 0.0)
        abs_target_y = target_y + getattr(self, 'offset_y', 0.0)
        abs_target_z = target_z + getattr(self, 'offset_z', 0.0)

        # 🚀 [최적화 1] 변경된 축(Axis)만 골라서 전송명령어 조립
        # (X, Y가 가만히 있고 Z만 움직일 때는 "GT z0.100" 처럼 단일 축 명령어만 생성)
        parts = ["GT"]
        if target_x != self.logical_x:
            parts.append(f"x{abs_target_x:.3f}")
        if target_y != self.logical_y:
            parts.append(f"y{abs_target_y:.3f}")
        if target_z != self.logical_z:
            parts.append(f"z{abs_target_z:.3f}")

        # 현재 좌표 업데이트
        self.logical_x = target_x
        self.logical_y = target_y
        self.logical_z = target_z

        # 🚀 [최적화 2] 목표 위치가 바뀌어 전송할 명령어가 있을 때만 통신
        if len(parts) > 1:
            cmd = " ".join(parts)

            # 1. 하나의 Lock 블록 안에서 모든 처리를 끝냅니다.
            with self.lock:
                try:
                    # 2. 인자로 전달받은 wait_ack 값을 사용하여 통신합니다.
                    # 맵핑 시에는 False가 전달되어 즉시 반환, UI 조작 시에는 True가 전달됨
                    self.send_command(cmd, wait_for_ack=wait_ack)
                except Exception as e:
                    print(f"Stage Command Error: {e}")

    def move_relative(self, dx, dy, dz):
        """
        상대 좌표 이동 (MR - Move Relative)
        주로 사용자가 화살표 버튼(조그)을 눌러서 수동으로 이동할 때 사용합니다.
        """
        if dx == 0 and dy == 0 and dz == 0:
            return

        target_z = self.logical_z + dz
        if target_z > self.Z_MAX_LIMIT:
            print(f"🚨 [안전 경고] Z축 상대 이동 시 제한선({self.Z_MAX_LIMIT}µm) 초과 위험!")
            dz = self.Z_MAX_LIMIT - self.logical_z

        self.logical_x += dx
        self.logical_y += dy
        self.logical_z += dz

        if not self.simulate and self.is_connected:
            # 매뉴얼 6.1.13 참조: "MR x50 y60.001"
            parts = ["MR"]
            if dx != 0: parts.append(f"x{dx:.3f}")
            if dy != 0: parts.append(f"y{dy:.3f}")
            if dz != 0: parts.append(f"z{dz:.3f}")

            cmd = " ".join(parts)
            self.send_command(cmd, wait_for_ack=True)

    def read_actual_position(self):
        """
        실제 하드웨어의 위치를 RP 명령어로 읽어옵니다. (디버깅/동기화 용도)
        """
        if not self.simulate and self.is_connected:
            # 매뉴얼 6.3.1 참조: "RP x y z"
            res = self.send_command("RP x y z", wait_for_ack=True)
            return res
        return f"Simulated: x{self.logical_x} y{self.logical_y} z{self.logical_z}"

    def set_speed(self, speed):
        """
        [매뉴얼 6.1.2 SPeed (SP)]
        스테이지의 이동 속도를 설정합니다 (단위: µm/s).
        """
        if not self.simulate and self.is_connected:
            # x, y, z 모든 축에 대해 동일한 속도 적용 (공백 필수)
            cmd = f"SP x{speed:.3f} y{speed:.3f} z{speed:.3f}"
            self.send_command(cmd, wait_for_ack=True)
        print(f"[Stage] 이동 속도가 {speed} µm/s 로 설정되었습니다.")

    def get_position(self):
        """
        현재 소프트웨어 기준의 x, y, z 좌표를 튜플 형태로 반환합니다.
        main.py의 UI 패널(현재 좌표 표시 등)에서 반복적으로 호출됩니다.
        """
        return self.logical_x, self.logical_y, self.logical_z

    def get_hardware_position(self):
        """컨트롤러에 X, Y, Z 각각 RP 명령을 보내 실제 하드웨어 절대 위치를 읽어옵니다."""
        import re

        def query_single_axis(axis_label):
            """특정 축(x, y, z) 하나만 개별적으로 장비에 물어보고 값을 추출하는 내부 함수"""
            # 예: "RP x" 전송
            cmd = f"RP {axis_label}"
            response = self.send_command(cmd, wait_for_ack=True)

            if not response:
                return 0.0

            if isinstance(response, bytes):
                response = response.decode('utf-8', errors='ignore')

            raw_str = response.strip()

            # 장비가 보낸 "$RP x1000.00" 같은 문자열에서 해당 축의 숫자만 추출
            match = re.search(f'{axis_label}[:\\s]*([-\\d\\.]+)', raw_str, re.IGNORECASE)

            if match:
                return float(match.group(1))
            else:
                # 라벨 매칭 실패 시, 부호와 소수점이 포함된 첫 번째 숫자를 강제 추출 (폴백)
                nums = re.findall(r'[-\d\.]+', raw_str)
                for n in nums:
                    try:
                        if n.count('.') <= 1 and n != '.':
                            return float(n)
                    except ValueError:
                        continue
            return 0.0

        # X, Y, Z를 순서대로 하나씩 물어봅니다.
        try:
            abs_x = query_single_axis('x')
            abs_y = query_single_axis('y')
            abs_z = query_single_axis('z')

            print(f"[Stage INFO] 절대 좌표 개별 수신 성공 -> X:{abs_x}, Y:{abs_y}, Z:{abs_z}")
            return abs_x, abs_y, abs_z

        except Exception as e:
            print(f"[Stage ERROR] 절대 위치 읽기 실패: {e}")
            return 0.0, 0.0, 0.0

    def move_hardware_absolute(self, x, y, z):
        """소프트웨어 오프셋 없이 기계적 절대 좌표로 이동(GT) 명령을 보냅니다."""
        cmd = f"GT x{x:.3f} y{y:.3f} z{z:.3f}"

        # 🚨 [수정] send_command로 변경
        self.send_command(cmd, wait_for_ack=True)

        # 이동 후 현재 논리 좌표 UI 싱크 연산
        self.logical_x = x - getattr(self, 'offset_x', 0.0)
        self.logical_y = y - getattr(self, 'offset_y', 0.0)
        self.logical_z = z - getattr(self, 'offset_z', 0.0)