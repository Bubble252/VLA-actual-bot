#!/usr/bin/env python3
"""单卡快速定位 rank1 第一个 bs=32 batch 的 NaN 样本（train 模式）。"""
import os, sys
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
from torch.utils.data.distributed import DistributedSampler

from starVLA.model.framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def load_cfg(bs):
    cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    cfg = apply_config_compat(cfg)
    cfg = wrap_config(cfg)
    cfg.datasets.vla_data.per_device_batch_size = bs
    cfg.trainer.freeze_modules = "qwen_vl_interface"
    return cfg


def forward_loss(model, examples):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward(examples)
    return out["action_loss"]


def main():
    PartialState()
    bs = 32
    cfg = load_cfg(bs)
    model = build_framework(cfg).cuda().train()
    for p in model.qwen_vl_interface.parameters():
        p.requires_grad = False

    ds = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    sampler = DistributedSampler(ds, num_replicas=2, rank=1, shuffle=True, seed=42)
    sampler.set_epoch(0)
    idx = list(sampler)[:bs]
    print(f"rank1 first batch indices[:8]={idx[:8]}")

    batch = [ds[i] for i in idx]
    try:
        loss = forward_loss(model, batch)
        print(f"full batch: loss={float(loss):.4f} finite={torch.isfinite(loss).item()}")
    except RuntimeError as e:
        print(f"full batch: {e}")

    for j, i in enumerate(idx):
        ex = [ds[i]]
        try:
            loss = forward_loss(model, ex)
            ok = torch.isfinite(loss).item()
            flag = "OK" if ok else "NAN"
            print(f"  [{flag}] idx={i} loss={float(loss) if ok else 'nan'}")
            if not ok:
                a = np.array(ex[0]["action"], dtype=np.float32)
                print(f"       action [{a.min():.3f}, {a.max():.3f}]")
        except RuntimeError as e:
            print(f"  [ERR] idx={i} {e}")


if __name__ == "__main__":
    main()
