#!/usr/bin/env python3
"""Print live U-Arm servo angles to find master_joint_indices mapping."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from lewm_x5_grasp.uarm_servo import UArmServoReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug U-Arm servo angles (move one joint at a time).")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--log", default=None, help="Append one line per sample to this file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reader = UArmServoReader(port=args.port, baudrate=args.baudrate)
    import threading

    threading.Thread(target=reader.read_loop, kwargs={"hz": 50.0}, daemon=True).start()
    dt = 1.0 / max(args.hz, 1.0)

    print("Move ONE U-Arm joint at a time and watch which servo index changes.")
    print("ARX joint index (bottom=0): map master_joint_indices[j] = that servo index.")
    print("Ctrl+C to exit.\n")

    log_path = Path(args.log).resolve() if args.log else None

    try:
        while True:
            angles = reader.get_angles_deg()
            parts = " ".join(f"s{i}:{angles[i]:+7.2f}" for i in range(7))
            print(f"\r{parts}", end="", flush=True)
            if log_path is not None:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(parts + "\n")
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
