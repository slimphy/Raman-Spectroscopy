import numpy as np
import time

try:
    from dcam import *
    from dcamcon import *

    DCAM_AVAILABLE = True
except ImportError:
    DCAM_AVAILABLE = False
    print("⚠️ [Error] dcam.py 또는 dcamcon.py 파일을 찾을 수 없습니다.")


class CameraController:
    def __init__(self):
        self.is_connected = False
        self.dcam = None

    def initialize_dcam(self):
        if not DCAM_AVAILABLE:
            return False
        if Dcamapi.init():
            print("DCAM API Initialized Successfully")
            return True
        else:
            print("DCAM API Initialization Failed")
            return False

    def connect_first_available_camera(self):
        if not DCAM_AVAILABLE:
            return False

        self.dcam = Dcam(0)  # 첫 번째 카메라 연결
        if self.dcam.dev_open():
            self.is_connected = True
            print("Hamamatsu qCMOS Camera Connected!")
            return True
        return False

    def disconnect(self):
        if self.is_connected and self.dcam:
            self.dcam.dev_close()
            self.is_connected = False
            print("Camera Disconnected.")

    def uninitialize_dcam(self):
        if DCAM_AVAILABLE:
            Dcamapi.uninit()

    def start_capture(self):
        if self.is_connected:
            self.dcam.buf_alloc(3)  # 버퍼 할당
            self.dcam.cap_start()
            print("Live Capture Started")

    def stop_capture(self):
        if self.is_connected:
            self.dcam.cap_stop()
            self.dcam.buf_release()
            print("Live Capture Stopped")

    def grab_frame(self):
        if not self.is_connected:
            return None

        # 1초(1000ms) 대기하며 프레임이 준비되었는지 확인
        if self.dcam.wait_capevent_frameready(1000):
            frame = self.dcam.buf_getlastframedata()
            return frame
        return None

    # --- 카메라 파라미터 제어 ---
    def set_exposure_time(self, exp_time_sec):
        if self.is_connected:
            self.dcam.prop_setvalue(DCAM_IDPROP.EXPOSURETIME, exp_time_sec)

    def set_binning(self, bin_size):
        if self.is_connected:
            # bin_size는 1, 2, 4 등의 정수
            self.dcam.prop_setvalue(DCAM_IDPROP.BINNING, bin_size)

    def set_roi(self, offset_y, height):
        if self.is_connected:
            self.dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE, 2)  # ON
            self.dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVSIZE, height)
            self.dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVPOS, offset_y)

    def set_cooler(self, state):
        if not self.is_connected:
            return

        try:
            if state:
                # 1. 쿨러 켜기 (🚨 수냉식 전용 MAX 쿨링 모드: 4.0 적용)
                # 과거 공랭일 때 열폭주를 일으켰던 그 설정이, 수냉 환경에서는 정답입니다!
                self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLER, 4.0)

                # 2. 타겟 온도 수동 설정은 펌웨어에서 거부하므로 삭제
                # (4.0 모드가 들어가면 카메라가 알아서 타겟을 -35도로 세팅합니다)

                # 3. 방열 팬(Fan) 끄기 (수냉이므로 진동 방지를 위해 강제 종료)
                try:
                    self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLERFAN, 1.0)
                except Exception as fan_e:
                    pass

            else:
                # 쿨러 끄기 (1.0 = OFF)
                self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLER, 1.0)
                try:
                    # 팬 끄기 (1.0 = OFF)
                    self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLERFAN, 1.0)
                except:
                    pass

            print(f"Sensor Cooler & Fan set to: {'WATER COOLING MODE (Fan OFF)' if state else 'OFF'}")
        except Exception as e:
            print(f"⚠️ 쿨러/팬 제어 실패: {e}")

    def get_temperature(self):
        """센서의 현재 온도를 읽어옵니다."""
        if self.is_connected:
            try:
                # DCAM_IDPROP.SENSORTEMPERATURE 의 현재 값을 읽어옴 (단위: 섭씨)
                temp = self.dcam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE)
                return temp
            except Exception as e:
                return None
        return None