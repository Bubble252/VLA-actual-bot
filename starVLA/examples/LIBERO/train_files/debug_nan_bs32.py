#!/usr/bin/env python3
"""
单卡定位 bs=32 时 DiT action_loss NaN（无需第二张训练卡）。

用法（等 GPU 空闲后，约 5–10 分钟）:
  CUDA_VISIBLE_DEVICES=6 python examples/LIBERO/train_files/debug_nan_bs32.py

会模拟双卡时 rank0/rank1 各自第一个 micro-batch(32)，并对坏 batch 逐样本二分。
"""
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
from torch.utils.data.distributed import DistributedSampler

from starVLA.model.framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def load_cfg(per_device_bs: int):
    cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    cfg = apply_config_compat(cfg)
    cfg = wrap_config(cfg)
    cfg.datasets.vla_data.per_device_batch_size = per_device_bs
    cfg.trainer.freeze_modules = "qwen_vl_interface"
    return cfg


def collate_batch(dataset, indices):
    return [dataset[i] for i in indices]


def check_batch(model, examples, tag: str) -> str | None:
    batch_images = [ex["image"] for ex in examples]
    instructions = [ex["lang"] for ex in examples]
    actions = [ex["action"] for ex in examples]
    state = [ex["state"] for ex in examples] if "state" in examples[0] else None

    acts = np.array(actions, dtype=np.float32)
    if not np.isfinite(acts).all():
        return f"{tag}: GT actions non-finite"

    qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images=batch_images, instructions=instructions
    )
    model.qwen_vl_interface.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.qwen_vl_interface(**qwen_inputs, return_dict=True)
    lh = out.last_hidden_state
    if lh is None and getattr(out, "hidden_states", None):
        lh = out.hidden_states[-1]
    lh = torch.nan_to_num(lh, nan=0.0, posinf=1e4, neginf=-1e4)

    rep = model._get_repeated_diffusion_steps()
    am_dtype = next(model.action_model.parameters()).dtype
    device = lh.device
    actions_t = torch.as_tensor(acts, device=device)
    actions_target = actions_t[:, -model.action_horizon :, :]
    lh_rep = lh.repeat(rep, 1, 1).to(dtype=am_dtype)
    act_rep = actions_target.repeat(rep, 1, 1).to(dtype=am_dtype)
    mask = qwen_inputs.get("attention_mask")
    if mask is not None:
        mask = mask.repeat(rep, 1).to(dtype=torch.bool)
    state_rep = None
    if state is not None:
        st = torch.as_tensor(np.array(state, dtype=np.float32), device=device)
        if not torch.isfinite(st).all():
            return f"{tag}: GT state non-finite"
        state_rep = st.repeat(rep, 1, 1).to(dtype=am_dtype)

    with torch.autocast("cuda", dtype=torch.float32):
        loss = model.action_model(lh_rep, act_rep, state_rep, encoder_attention_mask=mask)
    if not torch.isfinite(loss):
        return f"{tag}: action_loss NaN/Inf (DiT B={lh_rep.shape[0]})"
    return f"{tag}: OK loss={float(loss):.4f} DiT_B={lh_rep.shape[0]}"


def bisect_bad_samples(model, dataset, indices):
    examples = collate_batch(dataset, indices)
    err = check_batch(model, examples, "full_batch")
    if "OK" in err:
        print(f"  {err}")
        return
    print(f"  {err} -> bisect {len(indices)} samples")
    for idx in indices:
        ex = collate_batch(dataset, [idx])
        e = check_batch(model, ex, f"idx={idx}")
        if "NaN" in e or "non-finite" in e:
            a = np.array(ex[0]["action"], dtype=np.float32)
            print(f"    BAD {e}")
            print(f"      action range [{np.nanmin(a):.4f}, {np.nanmax(a):.4f}]")
            if "state" in ex[0]:
                s = np.array(ex[0]["state"], dtype=np.float32)
                print(f"      state  range [{np.nanmin(s):.4f}, {np.nanmax(s):.4f}]")


def main():
    PartialState()  # satisfy accelerate.logging used by spikebrain
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[debug_nan_bs32] CUDA_VISIBLE_DEVICES={gpu}")

    for bs in (16, 24, 32):
        print(f"\n========== per_device_bs={bs} ==========")
        cfg = load_cfg(bs)
        if not torch.cuda.is_available():
            print("No CUDA, exit.")
            return
        model = build_framework(cfg).cuda().eval()
        for p in model.qwen_vl_interface.parameters():
            p.requires_grad = False

        dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
        rep = model._get_repeated_diffusion_steps()
        print(f"dataset_size={len(dataset)} repeated_diffusion_steps={rep} DiT_B={bs * rep}")

        for rank in (0, 1):
            sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=42)
            sampler.set_epoch(0)
            batch_idx = list(sampler)[:bs]
            print(f"\n--- simulate DDP rank={rank} first micro-batch (indices[:4]={batch_idx[:4]}...) ---")
            bisect_bad_samples(model, dataset, batch_idx)

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
