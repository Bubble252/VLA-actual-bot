"""U-Arm master arm servo reader (Zhonglin PWM serial protocol)."""

from __future__ import annotations

import re
import threading
import time
from typing import List, Optional

import serial


class UArmServoReader:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200) -> None:
        self.port = port
        self.baudrate = baudrate
        self._ser = serial.Serial(port, baudrate, timeout=0.1)
        self._zero_angles = [0.0] * 7
        self._current_angles = [0.0] * 7
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._stop = threading.Event()
        self._init_servos()
        print(f"[UArmServoReader] Opened {port}")

    def _send_command(self, cmd: str) -> str:
        with self._io_lock:
            if self._stop.is_set() or not self._ser.is_open:
                return ""
            try:
                self._ser.write(cmd.encode("ascii"))
                time.sleep(0.008)
                return self._ser.read_all().decode("ascii", errors="ignore")
            except (serial.SerialException, TypeError, OSError, ValueError):
                return ""

    @staticmethod
    def _pwm_to_angle(
        response_str: str,
        servo_num: int,
        pwm_min: int = 500,
        pwm_max: int = 2500,
        angle_range: float = 270.0,
    ) -> Optional[float]:
        pattern = f"#{servo_num:03d}P(\\d{{4}})"
        match = re.search(pattern, response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        return (pwm_val - pwm_min) / pwm_span * angle_range

    def _init_servos(self, release_torque: bool = True) -> None:
        self._send_command("#000PVER!")
        for i in range(7):
            self._send_command("#000PCSK!")
            if release_torque:
                self._send_command(f"#{i:03d}PULK!")
            response = self._send_command(f"#{i:03d}PRAD!")
            angle = self._pwm_to_angle(response.strip(), i)
            self._zero_angles[i] = angle if angle is not None else 0.0
        print("[UArmServoReader] Zero calibration done")

    def release_torque(self, servo_ids: Optional[List[int]] = None) -> None:
        """Release servo torque (Zhonglin PULK). Arm can be moved by hand."""
        ids = list(range(7)) if servo_ids is None else servo_ids
        for i in ids:
            self._send_command("#000PCSK!")
            self._send_command(f"#{i:03d}PULK!")
        print(f"[UArmServoReader] Torque released on servos {ids}")

    def recalibrate_zero(self, servo_ids: Optional[List[int]] = None) -> List[float]:
        """Re-capture current pose as software zero (official servo_reader init logic)."""
        ids = list(range(7)) if servo_ids is None else servo_ids
        for i in ids:
            response = self._send_command(f"#{i:03d}PRAD!")
            angle = self._pwm_to_angle(response.strip(), i)
            if angle is not None:
                self._zero_angles[i] = angle
        with self._lock:
            self._current_angles = [0.0] * 7
        zeros = [self._zero_angles[i] for i in ids]
        print(f"[UArmServoReader] Re-zeroed servos {ids}: {[f'{z:.1f}' for z in zeros]}")
        return list(self._zero_angles)

    def read_loop(self, hz: float = 50.0) -> None:
        dt = 1.0 / max(hz, 1.0)
        while not self._stop.is_set():
            try:
                new_angles = [0.0] * 7
                for i in range(7):
                    response = self._send_command(f"#{i:03d}PRAD!")
                    angle = self._pwm_to_angle(response.strip(), i)
                    if angle is not None:
                        new_angles[i] = angle - self._zero_angles[i]
                with self._lock:
                    self._current_angles = new_angles
            except Exception as exc:
                print(f"[UArmServoReader] read_loop error: {exc}", flush=True)
            time.sleep(dt)

    def get_angles_deg(self) -> List[float]:
        with self._lock:
            return list(self._current_angles)

    def close(self) -> None:
        self._stop.set()
        with self._io_lock:
            if self._ser.is_open:
                self._ser.close()
