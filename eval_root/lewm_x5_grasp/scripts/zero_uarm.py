#!/usr/bin/env python3
"""U-Arm Zhonglin servo zero / diagnostic tool.

Ported from official LeRobot-Anything-U-Arm:
  src/uarm/scripts/Uarm_teleop/Zhonglin_servo/servo_zero.py

Official workflow (howtoplay.md):
  1. Run this script to verify servo readings (torque released, arm movable by hand).
  2. Place U-Arm in reference pose; teleop_record re-zeros at startup (servo_reader logic).

Modes:
  monitor (default)  - release torque, print raw + relative angles at ~10 Hz
  calibrate          - capture current pose as software zero and exit

Example:
  python3 scripts/zero_uarm.py --port /dev/ttyUSB0
  python3 scripts/zero_uarm.py --calibrate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from lewm_x5_grasp.uarm_servo import UArmServoReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U-Arm zero / diagnostic (Zhonglin servos).")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--calibrate", action="store_true", help="Set current pose as software zero and exit.")
    p.add_argument(
        "--no-release",
        action="store_true",
        help="Do not send PULK (keep torque); default releases torque like official servo_zero.py.",
    )
    p.add_argument("--hz", type=float, default=10.0)
    return p.parse_args()


def read_raw_and_rel(reader: UArmServoReader) -> tuple[List[float], List[float]]:
    raw: List[float] = []
    rel: List[float] = []
    for i in range(7):
        resp = reader._send_command(f"#{i:03d}PRAD!")
        a = reader._pwm_to_angle(resp.strip(), i)
        if a is None:
            raw.append(float("nan"))
            rel.append(float("nan"))
        else:
            raw.append(a)
            rel.append(a - reader._zero_angles[i])
    return raw, rel


def main() -> None:
    args = parse_args()
    reader = UArmServoReader(port=args.port, baudrate=args.baudrate)

    if not args.no_release:
        reader.release_torque()

    if args.calibrate:
        print("Place U-Arm in reference pose (elastic gripper naturally open), then press Enter…")
        try:
            input()
        except KeyboardInterrupt:
            reader.close()
            return
        reader.recalibrate_zero()
        _, rel = read_raw_and_rel(reader)
        print("Software zero captured. Relative angles:")
        print("  ", "  ".join(f"s{i}:{rel[i]:+6.1f}°" for i in range(7)))
        print("Note: teleop_record.py re-zeros again at its own startup.")
        reader.close()
        return

    dt = 1.0 / max(args.hz, 1.0)
    print("Monitoring U-Arm (official servo_zero.py style). Ctrl+C to quit.")
    print("  raw = PWM angle (deg)   rel = raw - software_zero")
    print("  Press Enter anytime to re-capture software zero at current pose.")
    print("-" * 72)
    try:
        while True:
            raw, rel = read_raw_and_rel(reader)
            line = "  ".join(
                f"s{i}: {raw[i]:+6.1f}/{rel[i]:+6.1f}" for i in range(7)
            )
            print(f"\r{line}", end="", flush=True)
            # Non-blocking check for Enter to re-zero (simple poll with timeout)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
