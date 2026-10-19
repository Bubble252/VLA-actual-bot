#!/usr/bin/env python3
"""Interactive U-Arm servo -> ARX joint mapping calibration.

Move ONE U-Arm joint at a time (from bottom: 1st, 2nd, ... 6th), press Enter after each.
The script detects which servo changed most and prints master_joint_indices for config JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from lewm_x5_grasp.uarm_servo import UArmServoReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate master_joint_indices for U-Arm teleop.")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--dof", type=int, default=6)
    p.add_argument(
        "--config",
        default=str(PROJECT / "config" / "teleop_uarm.example.json"),
        help="Write master_joint_indices into this JSON (backup .bak first).",
    )
    p.add_argument("--write", action="store_true", help="Apply result to --config.")
    return p.parse_args()


def read_stable(reader: UArmServoReader, samples: int = 15, dt: float = 0.05) -> List[float]:
    acc = [0.0] * 7
    for _ in range(samples):
        angles = reader.get_angles_deg()
        for i in range(7):
            acc[i] += angles[i]
        time.sleep(dt)
    return [v / samples for v in acc]


def max_delta(before: Sequence[float], after: Sequence[float]) -> tuple[int, float]:
    best_i = 0
    best_d = 0.0
    for i in range(7):
        d = abs(after[i] - before[i])
        if d > best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def main() -> None:
    args = parse_args()
    import threading

    reader = UArmServoReader(port=args.port, baudrate=args.baudrate)
    threading.Thread(target=reader.read_loop, kwargs={"hz": 50.0}, daemon=True).start()
    time.sleep(0.4)

    mapping: List[int] = []
    labels = ["最底下第1关节", "第2关节", "第3关节", "第4关节", "第5关节", "第6关节(腕部)"]

    print("=" * 60)
    print("U-Arm -> ARX 关节映射标定")
    print("每次只动 U-Arm 的一个关节（尽量大幅度），其他关节保持不动。")
    print("=" * 60)

    try:
        for j in range(args.dof):
            label = labels[j] if j < len(labels) else f"joint {j}"
            input(f"\n[{j}] 请只动 U-Arm {label}，准备好后按 Enter 记录零位...")
            before = read_stable(reader)
            print(f"    零位: {[f's{i}:{before[i]:+.1f}' for i in range(7)]}")
            input(f"    现在大幅度来回移动 {label}，按 Enter 结束...")
            after = read_stable(reader)
            servo, delta = max_delta(before, after)
            print(f"    变化: {[f's{i}:{after[i]:+.1f}' for i in range(7)]}")
            print(f"    => ARX joint{j} 对应 servo s{servo} (Δ={delta:.1f}°)")
            if delta < 3.0:
                print("    [警告] 变化 <3°，可能动错了关节或舵机未接好，请重试此步。")
            mapping.append(servo)

        print("\n" + "=" * 60)
        print("标定结果 master_joint_indices:")
        print(json.dumps(mapping))
        print("\n写入 config/teleop_uarm.example.json 的 mapping 段:")
        print(f'  "master_joint_indices": {json.dumps(mapping)},')
        print("=" * 60)

        if args.write:
            cfg_path = Path(args.config)
            backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            backup.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            data.setdefault("mapping", {})["master_joint_indices"] = mapping
            cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"已写入 {cfg_path}（备份 {backup}）")

    except KeyboardInterrupt:
        print("\n已取消。")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
