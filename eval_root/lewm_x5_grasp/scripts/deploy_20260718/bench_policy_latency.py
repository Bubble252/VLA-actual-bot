#!/usr/bin/env python3
"""Measure laptop ↔ GPU Policy Server communication / inference latency.

Splits:
  - T_ping_ms:   WebSocket ping RTT (almost pure network + tunnel + serialize)
  - T_infer_ms:  full predict_action RTT (network + GPU infer + serialize)
  - T_gpu_est:   max(0, T_infer - T_ping)  rough GPU share

Usage (tunnel already open):
  python3 bench_policy_latency.py --port 5694 --unnorm-key franka --n 30
  MODEL=1p bash bench_policy_latency.sh
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np


def _pct(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    a = np.sort(np.asarray(xs, dtype=np.float64))
    idx = int(np.clip(round((len(a) - 1) * p), 0, len(a) - 1))
    return float(a[idx])


def _summary(xs: Sequence[float]) -> dict:
    a = list(xs)
    return {
        "n": len(a),
        "mean": float(statistics.fmean(a)) if a else float("nan"),
        "p50": _pct(a, 0.50),
        "p95": _pct(a, 0.95),
        "min": float(min(a)) if a else float("nan"),
        "max": float(max(a)) if a else float("nan"),
    }


def _fmt(s: dict) -> str:
    return (
        f"n={s['n']} mean={s['mean']:.1f} p50={s['p50']:.1f} "
        f"p95={s['p95']:.1f} min={s['min']:.1f} max={s['max']:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("POLICY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POLICY_PORT", "5694")))
    parser.add_argument("--unnorm-key", default=os.environ.get("UNNORM_KEY", "franka"))
    parser.add_argument("--starvla-root", default=os.environ.get("STARVLA_ROOT", str(Path.home() / "project/starVLA")))
    parser.add_argument("--n", type=int, default=30, help="Timed rounds after warmup")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-ddim-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--task", default="pick and place the can")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON result path",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.starvla_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from deployment.model_server.tools import msgpack_numpy

    print(
        f"[bench] connect ws://{args.host}:{args.port} "
        f"unnorm={args.unnorm_key} n={args.n} warmup={args.warmup} "
        f"img={args.width}x{args.height}x2 ddim={args.num_ddim_steps}",
        flush=True,
    )
    t_conn0 = time.perf_counter()
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    t_conn = (time.perf_counter() - t_conn0) * 1000.0
    meta = client.get_server_metadata()
    print(f"[bench] connected in {t_conn:.1f} ms", flush=True)
    print(f"[bench] server ckpt={meta.get('ckpt_path', '?')}", flush=True)

    packer = msgpack_numpy.Packer()

    def one_ping(payload: dict) -> float:
        t0 = time.perf_counter()
        client._ws.send(packer.pack(payload))
        resp = client._ws.recv()
        dt = (time.perf_counter() - t0) * 1000.0
        if isinstance(resp, str):
            raise RuntimeError(resp)
        return dt

    # Dual RGB like real client (cam_high + cam_wrist)
    img0 = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    img1 = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    state = np.zeros((1, 7), dtype=np.float32)
    example = {"image": [img0, img1], "lang": args.task, "state": state}
    vla_input = {
        "examples": [example],
        "unnorm_key": args.unnorm_key,
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": int(args.num_ddim_steps),
    }
    packed = packer.pack(vla_input)
    payload_kb = len(packed) / 1024.0
    # Same-size upload but server only replies ping (no GPU) → transfer RTT
    heavy_ping = {
        "type": "ping",
        "request_id": "lat_heavy",
        "pad_images": [img0, img1],
    }
    heavy_kb = len(packer.pack(heavy_ping)) / 1024.0
    print(
        f"[bench] infer payload ≈ {payload_kb:.1f} KiB | "
        f"heavy-ping payload ≈ {heavy_kb:.1f} KiB",
        flush=True,
    )

    def one_infer() -> float:
        t0 = time.perf_counter()
        client.predict_action(vla_input)
        return (time.perf_counter() - t0) * 1000.0

    print("[bench] warmup...", flush=True)
    for _ in range(max(0, args.warmup)):
        one_ping({"type": "ping", "request_id": "warm"})
        one_ping(heavy_ping)
        one_infer()

    ping_ms: List[float] = []
    heavy_ms: List[float] = []
    infer_ms: List[float] = []
    print(f"[bench] timing {args.n} rounds...", flush=True)
    for i in range(args.n):
        ping_ms.append(one_ping({"type": "ping", "request_id": "lat"}))
        heavy_ms.append(one_ping(heavy_ping))
        infer_ms.append(one_infer())
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  [{i+1}/{args.n}] ping={ping_ms[-1]:.1f}ms "
                f"xfer={heavy_ms[-1]:.1f}ms infer={infer_ms[-1]:.1f}ms",
                flush=True,
            )

    gpu_est = [max(0.0, a - b) for a, b in zip(infer_ms, heavy_ms)]
    s_ping = _summary(ping_ms)
    s_heavy = _summary(heavy_ms)
    s_infer = _summary(infer_ms)
    s_gpu = _summary(gpu_est)

    print("", flush=True)
    print("======== RESULT (ms) ========", flush=True)
    print(f"T_ping      (小包通信RTT)     {_fmt(s_ping)}", flush=True)
    print(f"T_xfer      (带双图上传RTT)   {_fmt(s_heavy)}", flush=True)
    print(f"T_infer     (上传+GPU推理)    {_fmt(s_infer)}", flush=True)
    print(f"T_gpu_est   (infer-xfer)      {_fmt(s_gpu)}", flush=True)
    print(
        f"payload≈{payload_kb:.1f}KiB  connect={t_conn:.1f}ms  "
        f"path=ws://{args.host}:{args.port} (SSH -L 隧道)",
        flush=True,
    )
    print(
        "说明: T_ping=协议往返；T_xfer≈传图通信（无GPU）；"
        "T_infer=真机同量级端到端；T_gpu_est=粗估推理。",
        flush=True,
    )

    result = {
        "host": args.host,
        "port": args.port,
        "unnorm_key": args.unnorm_key,
        "ckpt_path": meta.get("ckpt_path"),
        "payload_kib": payload_kb,
        "heavy_ping_kib": heavy_kb,
        "connect_ms": t_conn,
        "image": [args.height, args.width, 3, 2],
        "num_ddim_steps": args.num_ddim_steps,
        "T_ping_ms": s_ping,
        "T_xfer_ms": s_heavy,
        "T_infer_ms": s_infer,
        "T_gpu_est_ms": s_gpu,
        "raw_ping_ms": ping_ms,
        "raw_xfer_ms": heavy_ms,
        "raw_infer_ms": infer_ms,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = args.out
    if out is None:
        out_dir = Path(__file__).resolve().parents[4] / "docs" / "latency_runs"
        # parents: deploy_20260718 -> scripts -> lewm_x5_grasp -> project root? 
        # __file__ = .../lewm_x5_grasp/scripts/deploy_20260718/bench_policy_latency.py
        # parents[0]=deploy, [1]=scripts, [2]=lewm_x5_grasp, [3]=project root
        out_dir = Path(__file__).resolve().parents[3] / "docs" / "latency_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = str(out_dir / f"latency_{args.port}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out}", flush=True)

    try:
        client.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
