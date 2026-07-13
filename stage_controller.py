"""Thread-safe serial controller for the piezo stage.

Drop-in replacement for the original ``stage_controller.py``. Connection,
command I/O, and disconnect are protected by one re-entrant lock so the serial
port cannot be closed while another thread is writing or waiting for a reply.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Optional, Tuple

try:
    import serial
except ImportError:  # Simulation and static checks can run without pyserial.
    serial = None  # type: ignore[assignment]

_SERIAL_EXCEPTION = getattr(serial, "SerialException", OSError)

_LOG = logging.getLogger(__name__)


class PiezoController:
    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        simulate: bool = False,
        ack_timeout: float = 0.25,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.simulate = bool(simulate)
        self.ack_timeout = max(0.02, float(ack_timeout))

        self.serial_conn: Optional[Any] = None
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
        self.last_error: Optional[str] = None

    def _set_error(self, message: str, exc: Optional[BaseException] = None) -> None:
        self.last_error = message
        if exc is None:
            _LOG.error(message)
        else:
            _LOG.exception(message, exc_info=exc)

    def connect(self, port_name: Optional[str] = None) -> bool:
        with self.lock:
            if port_name:
                self.port = port_name
            if self.is_connected:
                return True
            if self.simulate:
                self.is_connected = True
                _LOG.info("[Simulate] stage connected")
                return True

            try:
                if serial is None:
                    raise RuntimeError("pyserial is not installed; install requirements.txt")

                # Close a stale object left by a previous failed connection.
                if self.serial_conn is not None:
                    try:
                        self.serial_conn.close()
                    except Exception:
                        pass

                self.serial_conn = serial.Serial(
                    self.port,
                    self.baudrate,
                    timeout=0.05,
                    write_timeout=0.5,
                )
                time.sleep(0.10)
                if self.serial_conn.in_waiting:
                    self.serial_conn.read_all()
                self.is_connected = True
                self.last_error = None
                _LOG.info("Stage connected to %s", self.port)
                self.read_position()
                return True
            except Exception as exc:
                self.is_connected = False
                self._set_error(f"Stage connection failed: {self.port}", exc)
                try:
                    if self.serial_conn is not None:
                        self.serial_conn.close()
                except Exception:
                    pass
                self.serial_conn = None
                return False

    def disconnect(self) -> None:
        # The same lock is used by send_command, eliminating close-during-write.
        with self.lock:
            if not self.simulate and self.serial_conn is not None:
                try:
                    if self.serial_conn.is_open:
                        self.serial_conn.close()
                except Exception as exc:
                    self._set_error("Stage serial port close failed", exc)
            self.serial_conn = None
            self.is_connected = False
            _LOG.info("Stage disconnected")

    def send_command(
        self,
        command: str,
        wait_for_ack: bool = True,
        timeout: Optional[float] = None,
    ) -> str:
        """Write one complete command and optionally read one bounded response.

        For fire-and-forget commands the return value is ``"SENT"`` after a
        successful write. This lets movement methods update their cached logical
        position only when the serial write actually succeeded.
        """
        command = str(command).strip()
        if not command:
            return ""

        with self.lock:
            if self.simulate:
                return self._simulate_command(command)
            if not self.is_connected or self.serial_conn is None:
                return ""
            if not self.serial_conn.is_open:
                self.is_connected = False
                return ""

            try:
                # Remove only stale *input*. Resetting the output buffer before
                # every command can discard bytes still being transmitted.
                if self.serial_conn.in_waiting:
                    stale = self.serial_conn.read_all()
                    _LOG.debug("Discarded %d stale stage bytes", len(stale))

                payload = f"{command}\r\n".encode("ascii")
                written = self.serial_conn.write(payload)
                self.serial_conn.flush()
                if written != len(payload):
                    self._set_error(
                        f"Stage short write: {written}/{len(payload)} bytes for {command!r}"
                    )
                    return ""
                if not wait_for_ack:
                    return "SENT"

                deadline = time.monotonic() + max(0.02, float(timeout or self.ack_timeout))
                chunks: list[str] = []
                while time.monotonic() < deadline:
                    waiting = self.serial_conn.in_waiting
                    if waiting:
                        text = self.serial_conn.read(waiting).decode("ascii", errors="ignore")
                        chunks.append(text)
                        if "\n" in text or "\r" in text:
                            break
                    time.sleep(0.002)
                return "".join(chunks).strip()
            except (_SERIAL_EXCEPTION, OSError) as exc:
                self.is_connected = False
                self._set_error(f"Stage command I/O failed: {command}", exc)
                return ""
            except Exception as exc:
                self._set_error(f"Stage command failed: {command}", exc)
                return ""

    def _simulate_command(self, command: str) -> str:
        if command.startswith("RP "):
            axis = command.split()[-1].lower()
            return f"{axis}{getattr(self, f'hard_{axis}', 0.0):.3f}"
        if command.startswith("RI"):
            return "x1 y1 z1"
        return "SENT"

    def check_is_indexed(self) -> bool:
        response = self.send_command("RI x y z", wait_for_ack=True)
        match = re.search(r"x(\d).*y(\d).*z(\d)", response, re.IGNORECASE)
        return bool(match and all(int(value) == 1 for value in match.groups()))

    def find_index(self) -> bool:
        response = self.send_command("IN x y z", wait_for_ack=True)
        _LOG.info("Stage physical-index search started")
        return bool(response)

    def read_position(self) -> Tuple[float, float, float]:
        with self.lock:
            self.hard_x = self._query_single_axis("x")
            self.hard_y = self._query_single_axis("y")
            self.hard_z = self._query_single_axis("z")
            return self.get_logical_position()

    def _query_single_axis(self, axis_label: str) -> float:
        response = self.send_command(f"RP {axis_label}", wait_for_ack=True)
        if response:
            match = re.search(
                rf"{re.escape(axis_label)}[:\s]*([-+]?\d+(?:\.\d+)?)",
                response,
                re.IGNORECASE,
            )
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

            for token in re.findall(r"[-+]?\d+(?:\.\d+)?", response):
                try:
                    return float(token)
                except ValueError:
                    continue
        return float(getattr(self, f"hard_{axis_label}", 0.0))

    def get_logical_position(self) -> Tuple[float, float, float]:
        with self.lock:
            self.logical_x = self.hard_x - self.offset_x
            self.logical_y = self.hard_y - self.offset_y
            self.logical_z = self.hard_z - self.offset_z
            return self.logical_x, self.logical_y, self.logical_z

    def set_zero(self) -> None:
        with self.lock:
            self.read_position()
            self.offset_x = self.hard_x
            self.offset_y = self.hard_y
            self.offset_z = self.hard_z
            self.logical_x = self.logical_y = self.logical_z = 0.0

    def set_speed(self, speed: float) -> bool:
        speed = float(speed)
        return bool(
            self.send_command(
                f"SP x{speed:.3f} y{speed:.3f} z{speed:.3f}",
                wait_for_ack=True,
            )
        )

    def move_relative(self, dx: float, dy: float, dz: float) -> bool:
        with self.lock:
            parts = ["MR"]
            if dx != 0:
                parts.append(f"x{dx:.3f}")
            if dy != 0:
                parts.append(f"y{dy:.3f}")
            if dz != 0:
                parts.append(f"z{dz:.3f}")
            if len(parts) == 1:
                return True

            sent = bool(self.send_command(" ".join(parts), wait_for_ack=True))
            if sent:
                self.logical_x += float(dx)
                self.logical_y += float(dy)
                self.logical_z += float(dz)
                self.hard_x = self.logical_x + self.offset_x
                self.hard_y = self.logical_y + self.offset_y
                self.hard_z = self.logical_z + self.offset_z
            return sent

    def move_absolute(self, x: float, y: float, z: float) -> bool:
        return self.move_to_logical(x, y, z, wait_ack=True)

    def move_to_logical(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        wait_ack: bool = False,
    ) -> bool:
        with self.lock:
            target_x = float(target_x)
            target_y = float(target_y)
            target_z = float(target_z)
            abs_target_x = target_x + self.offset_x
            abs_target_y = target_y + self.offset_y
            abs_target_z = target_z + self.offset_z

            parts = ["GT"]
            tolerance = 1e-9
            if abs(target_x - self.logical_x) > tolerance:
                parts.append(f"x{abs_target_x:.3f}")
            if abs(target_y - self.logical_y) > tolerance:
                parts.append(f"y{abs_target_y:.3f}")
            if abs(target_z - self.logical_z) > tolerance:
                parts.append(f"z{abs_target_z:.3f}")
            if len(parts) == 1:
                return True

            result = self.send_command(" ".join(parts), wait_for_ack=wait_ack)
            if not result:
                return False

            # Update cached coordinates only after a successful serial write.
            self.logical_x = target_x
            self.logical_y = target_y
            self.logical_z = target_z
            self.hard_x = abs_target_x
            self.hard_y = abs_target_y
            self.hard_z = abs_target_z
            return True

    def set_trigger_out(self, axis: str = "x", value="0.0") -> bool:
        axis = str(axis).lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"Invalid trigger axis: {axis}")
        command = f"TO {axis}{value}"
        ok = bool(self.send_command(command, wait_for_ack=True))
        if ok:
            _LOG.info("Stage trigger output configured: %s", command)
        return ok

    def stop_motion(self) -> bool:
        # An ACK is useful here: cleanup code knows whether the stop command was
        # at least accepted before resources are closed.
        return bool(self.send_command("SH x y z", wait_for_ack=True, timeout=0.5))

    def get_position(self) -> Tuple[float, float, float]:
        return self.get_logical_position()
