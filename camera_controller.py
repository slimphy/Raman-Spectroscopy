import time

try:
    from dcam import *
    from dcamcon import *

    DCAM_AVAILABLE = True
except (ImportError, OSError):
    DCAM_AVAILABLE = False
    print("[Camera] DCAM Python module or dcamapi.dll is unavailable. Simulation can still run.")


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

        self.dcam = Dcam(0)
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
            self.dcam.buf_alloc(3)
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

        if self.dcam.wait_capevent_frameready(1000):
            frame = self.dcam.buf_getlastframedata()
            return frame
        return None

    def set_exposure_time(self, exp_time_sec):
        if self.is_connected:
            self.dcam.prop_setvalue(DCAM_IDPROP.EXPOSURETIME, exp_time_sec)

    def set_binning(self, bin_size):
        if self.is_connected:
            self.dcam.prop_setvalue(DCAM_IDPROP.BINNING, bin_size)

    def set_roi(self, offset_y, height):
        if self.is_connected:
            self.dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE, 2)
            self.dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVSIZE, height)
            self.dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVPOS, offset_y)

    def set_cooler(self, state):
        if not self.is_connected:
            return
        try:
            if state:
                self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLER, 4.0)
                try:
                    self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLERFAN, 1.0)
                except Exception as fan_e:
                    pass
            else:
                self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLER, 1.0)
                try:
                    self.dcam.prop_setvalue(DCAM_IDPROP.SENSORCOOLERFAN, 1.0)
                except:
                    pass
            print(f"Sensor Cooler & Fan set to: {'WATER COOLING MODE (Fan OFF)' if state else 'OFF'}")
        except Exception as e:
            print(f"⚠️ 쿨러/팬 제어 실패: {e}")

    def get_temperature(self):
        if self.is_connected:
            try:
                temp = self.dcam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE)
                return temp
            except Exception as e:
                return None
        return None

    def set_trigger_mode(self, mode="INTERNAL"):
        if not self.is_connected:
            print("[Camera] 연결되어 있지 않아 트리거를 설정할 수 없습니다.")
            return False
        try:
            if mode == "EXTERNAL":
                self.dcam.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE, 2.0)
                try:
                    self.dcam.prop_setvalue(DCAM_IDPROP.TRIGGERPOLARITY, 1.0)
                except:
                    pass
                try:
                    self.dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE, 1.0)
                except:
                    pass

                time.sleep(0.5)

                actual_mode = self.dcam.prop_getvalue(DCAM_IDPROP.TRIGGERSOURCE)
                if actual_mode != 2.0:
                    print(f"⚠️ [Camera] 트리거 변경 거부됨! (현재 설정값: {actual_mode})")
                    return False

                print("[Camera] ⚡ 외부 하드웨어 트리거(EXTERNAL) 대기 모드 전환!")
                return True
            else:
                self.dcam.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE, 1.0)
                print("[Camera] 💻 내부 소프트웨어 트리거(INTERNAL) 모드 복귀!")
                return True
        except Exception as e:
            print(f"[Camera] 트리거 모드 설정 중 에러: {e}")
            return False
