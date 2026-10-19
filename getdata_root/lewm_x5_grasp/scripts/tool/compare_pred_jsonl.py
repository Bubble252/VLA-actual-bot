#!/usr/bin/env python3
"""Compare two --dump-pred-jsonl runs (offset=0 vs offset≠0).

Focus: state_raw / state_sent (①) and world_vector / clipped_dxyz (②).

Usage:
  python3 compare_pred_jsonl.py run_off0.jsonl run_off1.jsonl
  python3 compare_pred_jsonl.py a.jsonl b.jsonl --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional


def _load_preds(path: Path) -> tuple[Optional[dict], list[dict]]:
    meta: Optional[dict] = None
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("event") == "run_meta":
                meta = obj
                continue
            if obj.get("event") == "pred" or "world_vector" in obj or "state_raw" in obj:
                rows.append(obj)
            elif "step" in obj and "target_q" in obj:
                rows.append(obj)
    return meta, rows


def _vec(row: dict, key: str) -> Optional[list[float]]:
    v = row.get(key)
    if v is None:
        return None
    return [float(x) for x in v]


def _l2(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def _fmt(v: Optional[list[float]], nd: int = 4) -> str:
    if v is None:
        return "None"
    return "[" + ", ".join(f"{x:.{nd}f}" for x in v[:3]) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl_a", type=Path, help="Run A (e.g. offset=0)")
    ap.add_argument("jsonl_b", type=Path, help="Run B (e.g. offset on)")
    ap.add_argument("--max-steps", type=int, default=30, help="Compare first N pred rows")
    ap.add_argument(
        "--eps-state",
        type=float,
        default=1e-4,
        help="state xyz L2 below this → treat as identical",
    )
    ap.add_argument(
        "--eps-delta",
        type=float,
        default=1e-4,
        help="world_vector / clipped_dxyz L2 below this → identical",
    )
    args = ap.parse_args()

    meta_a, rows_a = _load_preds(args.jsonl_a)
    meta_b, rows_b = _load_preds(args.jsonl_b)
    n = min(len(rows_a), len(rows_b), max(0, int(args.max_steps)))
    if n == 0:
        raise SystemExit("No overlapping pred rows to compare.")

    print("=== run_meta ===")
    for label, m in (("A", meta_a), ("B", meta_b)):
        if m is None:
            print(f"  {label}: (no run_meta)")
        else:
            print(
                f"  {label}: offset={m.get('eef_xyz_offset_m')} "
                f"modality={m.get('state_modality')} yaml={m.get('eef_offset_yaml')}"
            )

    same_state = 0
    same_wv = 0
    same_cd = 0
    sum_ds = 0.0
    sum_dw = 0.0
    sum_dc = 0.0
    counted_s = counted_w = counted_c = 0

    print(f"\n=== first {n} pred rows (xyz only) ===")
    for i in range(n):
        a, b = rows_a[i], rows_b[i]
        sa, sb = _vec(a, "state_raw"), _vec(b, "state_raw")
        na, nb = _vec(a, "state_sent"), _vec(b, "state_sent")
        wa, wb = _vec(a, "world_vector"), _vec(b, "world_vector")
        ca, cb = _vec(a, "clipped_dxyz"), _vec(b, "clipped_dxyz")

        line = f"i={i} stepA={a.get('step')} stepB={b.get('step')}"
        if sa is not None and sb is not None:
            ds = _l2(sa[:3], sb[:3])
            sum_ds += ds
            counted_s += 1
            if ds < args.eps_state:
                same_state += 1
            line += f" | d(state_raw_xyz)={ds:.5f} A={_fmt(sa)} B={_fmt(sb)}"
            if na is not None and nb is not None:
                line += f" sentA={_fmt(na)} sentB={_fmt(nb)}"
        if wa is not None and wb is not None:
            dw = _l2(wa[:3], wb[:3])
            sum_dw += dw
            counted_w += 1
            if dw < args.eps_delta:
                same_wv += 1
            line += f" | d(world_vector)={dw:.5f}"
        if ca is not None and cb is not None:
            dc = _l2(ca[:3], cb[:3])
            sum_dc += dc
            counted_c += 1
            if dc < args.eps_delta:
                same_cd += 1
            line += f" | d(clipped_dxyz)={dc:.5f}"
        print(line)

    print("\n=== verdict hints ===")
    if counted_s:
        mean_s = sum_ds / counted_s
        print(
            f"① state_raw xyz: mean_L2={mean_s:.5f}  "
            f"near_identical={same_state}/{counted_s}"
        )
        if mean_s < args.eps_state:
            print("   → state 几乎没变：查 YAML/是否重启 / 是否真 --send-eef-state")
        else:
            print("   → state 有变（符合 offset）；继续看 ②")
    else:
        print("① 无 state_raw：确认 dump 来自新 client + --send-eef-state")

    if counted_w:
        mean_w = sum_dw / counted_w
        print(
            f"② world_vector: mean_L2={mean_w:.5f}  "
            f"near_identical={same_wv}/{counted_w}"
        )
        if mean_w < args.eps_delta:
            print("   → Δ 几乎不变：模型对 state 不敏感 / 塌缩（不是 offset 没接上）")
        else:
            print("   → Δ 有变：若胳膊仍一样，再查执行限幅/IK")
    else:
        print("② 无 world_vector 字段")

    if counted_c:
        mean_c = sum_dc / counted_c
        print(
            f"   clipped_dxyz: mean_L2={mean_c:.5f}  "
            f"near_identical={same_cd}/{counted_c}"
        )


if __name__ == "__main__":
    main()
