#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from lewm_x5_grasp.image_source import RealSenseD405ImageSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Intel RealSense D405 and write latest RGB frame.")
    parser.add_argument("--save-path", default="/tmp/latest_frame.npy", help="RGB frame .npy output path.")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    camera = RealSenseD405ImageSource(
        width=args.width,
        height=args.height,
        fps=args.fps,
        save_path=args.save_path,
        display=True,
        align_depth=True,
    )
    try:
        while True:
            camera.read()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()


if __name__ == "__main__":
    main()
