#!/usr/bin/env python3
"""手动关节点动工具：单独转某个关节到指定角度（mode=5 位置控制）。

用途：排查 go_home 后 J5 没到位（-96.7° 而非 0°）的问题。
先确认 CAN + ros2 launch arx_x5_controller open_single_arm.launch.py 已起。

用法：
  # 查看当前关节角（不发指令）
  python3 jog_joint.py --show

  # 把 J5（索引 5，第 6 关节）转到 0°，其余关节保持当前
  python3 jog_joint.py --joint 5 --deg 0

  # 把 J5 转到 0°，同时其余 5 关节也回 go_home [90,0,0,0,0,*]
  python3 jog_joint.py --joint 5 --deg 0 --home-others

  # 全部回 go_home [90,0,0,0,0,0]
  python3 jog_joint.py --go-home
"""
import argparse
import math
import sys
import time

import numpy as np
import rclpy

sys.path.insert(0, "/home/casia/project/ARX_X5_clean(7.12eval)/lewm_x5_grasp/src")
from lewm_x5_grasp.arm_ros2 import X5ArmBridge  # noqa: E402

GO_HOME_DEG = [90.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def show_state(node: X5ArmBridge, label: str = "") -> None:
    st = node.latest_state
    if st is None:
        print("[show] 还没有 /arm_status，确认控制器已启动")
        return
    deg = [math.degrees(v) for v in st.joint_pos[:6]]
    grip = float(st.joint_pos[6]) if st.joint_pos.shape[0] > 6 else float("nan")
    print(
        "[show]%s joint_deg=[%s] gripper=%.3f"
        % (label, ", ".join("%.2f" % d for d in deg), grip)
    )
    print("[show]%s joint_rad=[%s]" % (label, ", ".join("%.4f" % v for v in st.joint_pos[:6])))


def blend_to(node: X5ArmBridge, target_deg, gripper: float, hz: float = 30.0) -> None:
    """平滑插值到目标关节角（最短路径），避免突变。"""
    st = node.latest_state
    cur = np.asarray(st.joint_pos[:6], dtype=np.float64)
    tgt = np.array([math.radians(d) for d in target_deg], dtype=np.float64)
    delta = np.arctan2(np.sin(tgt - cur), np.cos(tgt - cur))
    dist = float(np.max(np.abs(delta)))
    if dist < 0.002:
        print("[blend] 已在目标位姿（差 %.4f rad）" % dist)
        return
    dur = min(4.0, max(0.5, dist / 0.6))  # 0.6 rad/s
    n = max(5, int(dur * hz))
    print("[blend] 移动 %s（max-delta %.1f°，%.2fs，%d 步）" % (
        [round(d, 1) for d in target_deg], math.degrees(dist), dur, n))
    period = 1.0 / hz
    for k in range(1, n + 1):
        t0 = time.perf_counter()
        a = 0.5 - 0.5 * math.cos(math.pi * k / n)  # smoothstep
        q = cur + a * delta
        node.publish_joint_command(q.tolist(), float(gripper), mode=5)
        rclpy.spin_once(node, timeout_sec=0.0)
        el = time.perf_counter() - t0
        if el < period:
            time.sleep(period - el)
    # 锁定
    for _ in range(int(1.0 * hz)):
        node.publish_joint_command(
            [math.radians(d) for d in target_deg], float(gripper), mode=5)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(period)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="只显示当前关节角")
    ap.add_argument("--joint", type=int, default=None, help="要点动的关节索引 0-5")
    ap.add_argument("--deg", type=float, default=None, help="目标角度（度）")
    ap.add_argument("--home-others", action="store_true",
                    help="点动某关节时，其余关节回 go_home")
    ap.add_argument("--go-home", action="store_true", help="全部回 go_home")
    ap.add_argument("--gripper", type=float, default=None,
                    help="夹爪指令（默认保持当前）")
    args = ap.parse_args()

    rclpy.init()
    node = X5ArmBridge()
    try:
        node.wait_for_state(timeout_sec=5.0)
    except RuntimeError as e:
        print("ERROR:", e)
        rclpy.shutdown()
        sys.exit(1)

    st = node.latest_state
    cur_deg = [math.degrees(v) for v in st.joint_pos[:6]]
    cur_grip = float(st.joint_pos[6]) if st.joint_pos.shape[0] > 6 else 0.0
    grip = args.gripper if args.gripper is not None else cur_grip

    show_state(node, "(起始) ")

    if args.show:
        rclpy.shutdown()
        return

    if args.go_home:
        target = GO_HOME_DEG
    elif args.joint is not None and args.deg is not None:
        target = list(cur_deg)
        if args.home_others:
            target = list(GO_HOME_DEG)
        target[args.joint] = args.deg
    else:
        print("ERROR: 用 --show / --go-home / --joint N --deg X")
        rclpy.shutdown()
        sys.exit(1)

    print("[plan] 目标 joint_deg=%s gripper=%.2f" % (
        [round(d, 1) for d in target], grip))
    blend_to(node, target, grip)
    time.sleep(0.3)
    rclpy.spin_once(node, timeout_sec=0.1)
    show_state(node, "(到位) ")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
