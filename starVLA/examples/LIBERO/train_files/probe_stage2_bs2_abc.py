#!/usr/bin/env python3
"""
阶段2全训 bs=2 NaN 探测：依次跑 A/B/C（建议子进程隔离，避免 CUDA 污染）。

A: bs=2 + VLM freeze(eval+no_grad) — 区分 batch>1 vs train+grad
B: bs=2 + 全训 + gradient_checkpointing=false
C: bs=2 + 全训 + use_liger_kernel=false（GC 保持 true）

用法:
  bash examples/LIBERO/train_files/run_probe_stage2_bs2_abc.sh
  # 或单实验:
  CUDA_VISIBLE_DEVICES=6 python examples/LIBERO/train_files/probe_stage2_bs2_abc.py --exp A
"""
import argparse
import os
import sys

STARVLA_ROOT = "/9950backfile/zhangjh/wh/starvla_wh/starVLA"
os.chdir(STARVLA_ROOT)
sys.path.insert(0, STARVLA_ROOT)

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from accelerate import PartialState
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

BS = 2
NUM_BATCHES = 3
CKPT = (
    f"{STARVLA_ROOT}/results/Checkpoints/"
    "20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_6000_pytorch_model.pt"
)

EXP_CONFIGS = {
    "A": {"freeze": "qwen_vl_interface", "grad_ckpt": False, "use_liger": True, "desc": "freeze eval+no_grad"},
    "B": {"freeze": "", "grad_ckpt": False, "use_liger": True, "desc": "full train, GC off"},
    "C": {"freeze": "", "grad_ckpt": True, "use_liger": False, "desc": "full train, liger off"},
}


def load_cfg(*, freeze: str, grad_ckpt: bool, use_liger: bool):
    cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    cfg = apply_config_compat(cfg)
    cfg = wrap_config(cfg)
    cfg.datasets.vla_data.per_device_batch_size = BS
    cfg.trainer.freeze_modules = freeze
    cfg.trainer.enable_gradient_checkpointing = grad_ckpt
    cfg.framework.qwenvl.use_liger_kernel = use_liger
    return cfg


def load_checkpoint(model, ckpt_path: str):
    print(f"loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected[:3]:
        print(f"  unexpected sample: {unexpected[:3]}")


def collate_batch(dataset, start: int):
    return [dataset[(start + i) % len(dataset)] for i in range(BS)]


def forward_stats(model, examples) -> dict:
    """单次 model.forward；hook sanitize 前统计 VLM hidden。"""
    acts = np.array([ex["action"] for ex in examples], dtype=np.float32)
    if not np.isfinite(acts).all():
        return {"error": "GT actions non-finite"}

    captured = {}

    orig_sanitize = model._sanitize_vlm_hidden

    def _hook_sanitize(last_hidden, context="train"):
        bad_mask = ~torch.isfinite(last_hidden)
        captured["shape"] = tuple(last_hidden.shape)
        captured["bad"] = int(bad_mask.sum().item())
        captured["nan"] = int(torch.isnan(last_hidden).sum().item())
        captured["total"] = last_hidden.numel()
        return orig_sanitize(last_hidden, context)

    model._sanitize_vlm_hidden = _hook_sanitize
    freeze_vlm = model._is_vlm_frozen()

    try:
        result = model.forward(examples)
        loss = float(result["action_loss"])
        loss_ok = np.isfinite(loss)
        result_err = None
    except Exception as e:
        loss = float("nan")
        loss_ok = False
        result_err = str(e)
    finally:
        model._sanitize_vlm_hidden = orig_sanitize

    if "bad" not in captured:
        return {"error": result_err or "sanitize hook not reached"}

    total = captured["total"]
    bad = captured["bad"]
    return {
        "freeze_vlm": freeze_vlm,
        "shape": captured["shape"],
        "bad": bad,
        "nan": captured["nan"],
        "total": total,
        "bad_pct": 100.0 * bad / max(total, 1),
        "loss": loss,
        "loss_ok": loss_ok,
        "result_err": result_err,
    }


def run_experiment(exp: str):
    cfg_dict = EXP_CONFIGS[exp]
    print(f"\n{'=' * 60}")
    print(
        f"[{exp}] {cfg_dict['desc']} | "
        f"freeze={cfg_dict['freeze']!r} grad_ckpt={cfg_dict['grad_ckpt']} "
        f"use_liger={cfg_dict['use_liger']}"
    )
    print("=" * 60)

    cfg = load_cfg(
        freeze=cfg_dict["freeze"],
        grad_ckpt=cfg_dict["grad_ckpt"],
        use_liger=cfg_dict["use_liger"],
    )
    if not torch.cuda.is_available():
        print("No CUDA, exit.")
        return 1

    torch.cuda.reset_peak_memory_stats()
    model = build_framework(cfg).cuda()
    load_checkpoint(model, CKPT)

    if cfg_dict["freeze"]:
        model.eval()
        for p in model.qwen_vl_interface.parameters():
            p.requires_grad = False
    else:
        model.train()

    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"model ready, cuda_allocated={mem_mb:.0f} MB")

    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    print(f"dataset_size={len(dataset)}")

    # bs=1 sanity
    ex1 = [dataset[0]]
    s1 = forward_stats(model, ex1)
    if "error" in s1:
        print(f"  sanity_bs1: ERROR {s1['error']}")
        return 1
    print(
        f"  sanity_bs1: VLM bad={s1['bad']}/{s1['total']} "
        f"loss={s1['loss']:.4f} freeze={s1['freeze_vlm']}"
    )

    all_ok = True
    for b in range(NUM_BATCHES):
        examples = collate_batch(dataset, b * BS)
        stats = forward_stats(model, examples)
        if "error" in stats:
            print(f"  batch{b}: ERROR {stats['error']}")
            all_ok = False
            continue

        vlm_status = "OK" if stats["bad"] == 0 else "BAD"
        loss_status = "OK" if stats["loss_ok"] else "BAD"
        print(
            f"  batch{b}: VLM=[{vlm_status}] shape={stats['shape']} "
            f"bad={stats['bad']}/{stats['total']} ({stats['bad_pct']:.2f}%) "
            f"freeze={stats['freeze_vlm']} | "
            f"loss=[{loss_status}] {stats['loss']:.4f}"
        )
        if stats["result_err"]:
            print(f"           forward_err: {stats['result_err']}")
        if stats["bad"] > 0 or not stats["loss_ok"]:
            all_ok = False

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"  peak_cuda_allocated={peak_mb:.0f} MB")
    print(f"[{exp}] verdict: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["A", "B", "C", "all"], default="all")
    args = parser.parse_args()

    PartialState()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[probe_stage2_bs2_abc] CUDA_VISIBLE_DEVICES={gpu} per_device_bs={BS}")

    exps = ["A", "B", "C"] if args.exp == "all" else [args.exp]
    rc = 0
    for exp in exps:
        r = run_experiment(exp)
        if r != 0:
            rc = r
        del r
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass

    print(f"\n[probe_stage2_bs2_abc] done. rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
