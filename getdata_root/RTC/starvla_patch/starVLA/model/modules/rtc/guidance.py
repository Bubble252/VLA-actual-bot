"""RTC prefix guidance for StarVLA flow-matching (t: 0 → 1).

Ported from LeRobot ``RTCProcessor.denoise_step`` with the EVO1 time/velocity
mapping so it matches StarVLA's Euler convention:

    actions = actions + dt * v,  t = step / N

LeRobot computes velocity first (no graph through the network), then forms
``x1 = x - time * v`` and backprops only that linear map — we keep the same
pattern after the EVO1 sign/time mapping.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

# ΔEEF x,y,z,roll,pitch,yaw on; gripper off
RTC_DIM_MASK_DELTA_EEF: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)


def _prefix_weights_linear(start: int, end: int, total: int) -> Tensor:
    start = min(int(start), int(end))
    end = int(end)
    total = int(total)
    weights = torch.zeros(total)
    if start > 0:
        weights[: min(start, total)] = 1.0
    if end > start and total > start:
        linspace_steps = min(total, end) - start
        if linspace_steps > 0:
            mid = torch.linspace(1, 0, linspace_steps + 2)[1:-1]
            n = min(mid.numel(), total - start)
            weights[start : start + n] = mid[:n]
    return weights


def guide_starvla_velocity(
    x_t: Tensor,
    pred_velocity: Tensor,
    prev_chunk_left_over: Tensor,
    inference_delay: int,
    t_starvla: float,
    execution_horizon: int,
    *,
    max_guidance_weight: float = 10.0,
    dim_mask: Optional[Sequence[float]] = None,
) -> Tensor:
    """Apply RTC guidance to an already-computed StarVLA velocity.

    Args:
        x_t: Current Euler state ``(B,T,A)`` or ``(T,A)`` in normalized space.
        pred_velocity: DiT velocity for StarVLA (``actions += dt * v``).
        prev_chunk_left_over: Normalized leftover prefix.
        inference_delay: Frozen prefix steps.
        t_starvla: Continuous time in ``[0, 1)`` used by StarVLA Euler.
        execution_horizon: Soft-overlap end index.
    """
    if prev_chunk_left_over is None:
        return pred_velocity

    t = float(t_starvla)
    time_pi0 = 1.0 - t
    tau = t  # == 1 - time_pi0

    x_work = x_t.detach()
    v_star = pred_velocity.detach()
    squeezed = False
    if x_work.dim() < 3:
        x_work = x_work.unsqueeze(0)
        v_star = v_star.unsqueeze(0)
        squeezed = True

    prefix = prev_chunk_left_over
    if not torch.is_tensor(prefix):
        prefix = torch.as_tensor(prefix, dtype=x_work.dtype)
    if prefix.dim() < 3:
        prefix = prefix.unsqueeze(0)
    prefix = prefix.to(device=x_work.device, dtype=x_work.dtype)

    batch_size, chunk_size, action_dim = x_work.shape
    horizon = int(execution_horizon)
    if horizon > prefix.shape[1]:
        horizon = int(prefix.shape[1])
    delay = max(0, min(int(inference_delay), chunk_size))

    if prefix.shape[1] < chunk_size or prefix.shape[2] < action_dim:
        padded = torch.zeros(batch_size, chunk_size, action_dim, device=x_work.device, dtype=x_work.dtype)
        padded[:, : prefix.shape[1], : prefix.shape[2]] = prefix[
            :, : min(prefix.shape[1], chunk_size), : min(prefix.shape[2], action_dim)
        ]
        prefix = padded
    if prefix.shape[0] == 1 and batch_size > 1:
        prefix = prefix.expand(batch_size, -1, -1)

    weights = _prefix_weights_linear(delay, max(horizon, delay), chunk_size).to(x_work.device)
    weights = weights.view(1, chunk_size, 1)

    if dim_mask is None:
        mask_t = torch.ones(1, 1, action_dim, device=x_work.device, dtype=x_work.dtype)
        if action_dim >= 7:
            mask_t = mask_t.clone()
            mask_t[..., 6] = 0.0
    else:
        mask_list = list(dim_mask)
        if len(mask_list) < action_dim:
            mask_list = mask_list + [0.0] * (action_dim - len(mask_list))
        mask_t = torch.tensor(mask_list[:action_dim], device=x_work.device, dtype=x_work.dtype).view(1, 1, -1)
    weights = weights * mask_t

    # EVO1 map: processor sees v_pi0 = -v_starvla and time = 1 - t
    v_pi0 = -v_star

    with torch.enable_grad():
        x_req = x_work.detach().requires_grad_(True)
        x1_t = x_req - time_pi0 * v_pi0
        err = (prefix - x1_t) * weights
        correction = torch.autograd.grad(x1_t, x_req, err.detach(), retain_graph=False)[0]

    max_gw = torch.as_tensor(float(max_guidance_weight), device=x_work.device, dtype=x_work.dtype)
    tau_t = torch.as_tensor(tau, device=x_work.device, dtype=x_work.dtype)
    squared_one_minus_tau = (1.0 - tau_t) ** 2
    inv_r2 = (squared_one_minus_tau + tau_t**2) / torch.clamp(squared_one_minus_tau, min=1e-8)
    c = torch.nan_to_num((1.0 - tau_t) / torch.clamp(tau_t, min=1e-8), posinf=float(max_guidance_weight))
    guidance_weight = torch.minimum(
        torch.nan_to_num(c * inv_r2, posinf=float(max_guidance_weight)),
        max_gw,
    )

    guided_pi0 = v_pi0 - guidance_weight * correction.detach()
    result = -guided_pi0  # back to StarVLA velocity

    if not torch.isfinite(result).all():
        return pred_velocity.detach()

    if squeezed:
        result = result.squeeze(0)
    return result


def prepare_prefix_tensor(
    prefix: Union[Tensor, np.ndarray],
    *,
    batch_size: int,
    chunk_size: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not torch.is_tensor(prefix):
        prefix = torch.as_tensor(np.asarray(prefix), dtype=torch.float32)
    prefix = prefix.to(device=device, dtype=dtype)
    if prefix.dim() == 2:
        prefix = prefix.unsqueeze(0)
    if prefix.shape[-1] < action_dim:
        pad = torch.zeros(*prefix.shape[:-1], action_dim - prefix.shape[-1], device=device, dtype=dtype)
        prefix = torch.cat([prefix, pad], dim=-1)
    elif prefix.shape[-1] > action_dim:
        prefix = prefix[..., :action_dim]
    if prefix.shape[1] > chunk_size:
        prefix = prefix[:, :chunk_size]
    if prefix.shape[0] == 1 and batch_size > 1:
        prefix = prefix.expand(batch_size, -1, -1)
    return prefix
