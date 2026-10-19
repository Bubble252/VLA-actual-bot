#!/usr/bin/env python3
"""
双卡 (GPU 6,7) 复现 bs=32 的 action_loss NaN，与正式训练相同的 Accelerator + DataLoader 分片。

用法:
  bash examples/LIBERO/train_files/debug_nan_bs32_dual.sh
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
import gc
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import wrap_config


def load_cfg(per_device_bs: int):
    cfg = OmegaConf.load("examples/LIBERO/train_files/starvla_cotrain_libero.yaml")
    cfg = apply_config_compat(cfg)
    cfg = wrap_config(cfg)
    cfg.datasets.vla_data.per_device_batch_size = per_device_bs
    cfg.trainer.freeze_modules = "qwen_vl_interface"
    chunk_size = os.environ.get("DEBUG_CHUNK_SIZE")
    if chunk_size is not None and chunk_size != "":
        cfg.framework.action_model.action_forward_chunk_size = int(chunk_size)
    repeat = os.environ.get("DEBUG_REPEAT")
    if repeat is not None and repeat != "":
        cfg.framework.action_model.repeated_diffusion_steps = int(repeat)
    cfg.output_dir = "./results/Checkpoints/debug_nan_bs32_dual67"
    return cfg


def check_examples(model, examples, tag: str) -> str:
    batch_images = [ex["image"] for ex in examples]
    instructions = [ex["lang"] for ex in examples]
    actions = [ex["action"] for ex in examples]
    state = [ex["state"] for ex in examples] if examples and "state" in examples[0] else None

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
    lh_f = lh.float()
    lh_stats = (
        f"lh_absmax={lh_f.abs().max().item():.4g} "
        f"lh_nan={torch.isnan(lh_f).any().item()} lh_inf={torch.isinf(lh_f).any().item()}"
    )
    lh = torch.nan_to_num(lh, nan=0.0, posinf=100.0, neginf=-100.0)

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
            return f"{tag}: GT state non-finite | {lh_stats}"
        state_rep = st.repeat(rep, 1, 1).to(dtype=am_dtype)

    with torch.autocast("cuda", dtype=torch.float32):
        loss = model.action_model(lh_rep, act_rep, state_rep, encoder_attention_mask=mask)
    if not torch.isfinite(loss):
        return f"{tag}: action_loss NaN/Inf DiT_B={lh_rep.shape[0]} | {lh_stats}"
    return f"{tag}: OK loss={float(loss):.4f} DiT_B={lh_rep.shape[0]} | {lh_stats}"


def bisect_rank_batch(model, batch, rank: int):
    import gc

    print(f"\n[RANK {rank}] bisect batch_size={len(batch)}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    msg = check_examples(model, batch, "full_batch")
    print(f"[RANK {rank}] {msg}")
    if "OK" in msg:
        return
    for i, ex in enumerate(batch):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        m = check_examples(model, ex, f"sample={i}")
        if "NaN" in m or "non-finite" in m:
            a = np.array(ex["action"], dtype=np.float32)
            ds_idx = ex.get("index", ex.get("dataset_index", "?"))
            print(f"[RANK {rank}] BAD {m}")
            print(f"[RANK {rank}]   dataset_index={ds_idx} action [{np.nanmin(a):.4f}, {np.nanmax(a):.4f}]")


def raw_model(model):
    return getattr(model, "module", model)


def gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "cuda=N/A"
    gb = 1024**3
    return (
        f"alloc={torch.cuda.memory_allocated()/gb:.2f}GB "
        f"reserved={torch.cuda.memory_reserved()/gb:.2f}GB "
        f"peak_alloc={torch.cuda.max_memory_allocated()/gb:.2f}GB"
    )


def main():
    per_device_bs = int(os.environ.get("DEBUG_BS", "32"))
    grad_accum = int(os.environ.get("DEBUG_GRAD_ACCUM", "2"))
    max_opt_steps = int(os.environ.get("DEBUG_OPT_STEPS", "2"))
    cfg = load_cfg(per_device_bs)
    cfg.trainer.gradient_accumulation_steps = grad_accum
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum)
    rank = accelerator.process_index
    set_seed(cfg.seed + rank)

    if rank == 0:
        chunk_size = getattr(cfg.framework.action_model, "action_forward_chunk_size", None)
        repeat = getattr(cfg.framework.action_model, "repeated_diffusion_steps", None)
        print(
            f"[debug_nan_bs32_dual] per_device_bs={per_device_bs} "
            f"grad_accum={grad_accum} opt_steps={max_opt_steps} "
            f"chunk_size={chunk_size} repeat={repeat} "
            f"world_size={accelerator.num_processes}"
        )

    model = build_framework(cfg)
    for p in model.qwen_vl_interface.parameters():
        p.requires_grad = False
    model.train()

    dl = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data_iter = iter(dl)
    completed_steps = 0
    micro_idx = 0

    while completed_steps < max_opt_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    out = model.forward(batch)
                    loss = out["action_loss"]
                except RuntimeError as e:
                    print(
                        f"[RANK {rank}] micro={micro_idx} opt_step={completed_steps} "
                        f"FORWARD_ERR={e} batch_len={len(batch)}"
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    bisect_rank_batch(raw_model(model), batch, rank)
                    break

            ok = torch.isfinite(loss).item()
            sync = accelerator.sync_gradients
            guard = {
                k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v)
                for k, v in out.items()
                if k not in ("action_loss",)
            }
            guard_str = " ".join(f"{k}={guard[k]}" for k in sorted(guard))
            print(
                f"[RANK {rank}] micro={micro_idx} opt_step={completed_steps} "
                f"loss={float(loss):.6f} finite={ok} sync_gradients={sync} "
                f"batch_len={len(batch)} {guard_str} {gpu_mem_str()}"
            )
            if not ok:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bisect_rank_batch(raw_model(model), batch, rank)
                break

            accelerator.backward(loss)
            optimizer.step()

        micro_idx += 1
        if accelerator.sync_gradients:
            completed_steps += 1
            if rank == 0:
                print(f"[debug_nan_bs32_dual] finished optimizer step {completed_steps}/{max_opt_steps}")

    accelerator.wait_for_everyone()
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"[RANK {rank}] SUMMARY micro_batches={micro_idx} peak_cuda_alloc={peak_gb:.2f}GB")


if __name__ == "__main__":
    main()
